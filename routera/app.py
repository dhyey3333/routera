"""ROUTERA control room: Sentinel + Shelf-Life + Planner + Approval & Audit, live in one app.

    python -m routera            # http://localhost:8000, plays the India-inside scenario

Everything from agents/sentinel/api.py (REST + the SSE stream) plus:
    GET  /                            the intro and sign-in page (name, role, team access code)
    GET  /room                        the ROUTERA control room, once signed in
    GET|POST|DELETE /api/session      who's on duty; set ROUTERA_ACCESS_CODE to require the team's code
    GET  /sentinel                    the Sentinel's own console
    GET  /api/state                   snapshot incl. fleet, plans, KPIs and the audit chain
    POST /api/plans/{id}/approve      {"approver": "...", "overrides": {container: route_id}}
    POST /api/plans/{id}/reject       {"approver": "...", "reason": "..."}
    GET  /api/plans/{id}/why?container=GRP-07
    GET  /api/audit                   full chain + verification
    POST /api/live                    Live mode: real weather, news and ships feed the Sentinel
The stream adds "routera" (fleet, plans, KPIs), "audit" (each new chain entry) and "feeds" events.
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
import webbrowser
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response

from agents.sentinel.api import STATIC_DIR as SENTINEL_STATIC
from agents.sentinel.api import SentinelService, create_app
from agents.sentinel.llm_client import LLM_MODES, build_llm_client
from agents.sentinel.replay import SCENARIO_NAMES, Rig, Scenario, make_rig
from routera.access import COOKIE, ROLES, SESSION_S, Access
from routera.audit import HashChainAudit
from routera.filing import build_zip, pdf_available, render_pdf
from routera.geo import load_geo
from routera.live_feeds import TrafficFeed, default_feeds, live_books
from routera.orchestrator import Orchestrator
from routera.phone import TelegramBot

STATIC = Path(__file__).parent / "static"
DEFAULT_PORT = 8000
OPTIONAL_BODY = Body(default={})
LIVE = "live"
LIVE_TICK_S = 60.0
REFRESH_MIN_S = 30.0  # the fastest one feed can be refreshed by hand, to protect free API limits
NO_STORE = {"Cache-Control": "no-store"}
SIGN_IN_PATHS = {"/", "/api/session"}  # all a visitor without the team code can reach
log = logging.getLogger("routera.live")


def live_scenario(now: datetime) -> Scenario:
    return Scenario(
        name=LIVE, title="LIVE: real weather, alerts, traffic, news and ships, on a simulated fleet",
        simulated=True, layer="india", start=now, container_ids=[], steps=[],
        disclaimer="Feeds are REAL (Open-Meteo, GDACS, Google News; AIS and TomTom with free keys). The fleet is the "
                   "SIMULATED book moved to today; bookings stay simulated.",
    )


class RouteraService(SentinelService):
    """The Sentinel service, with the orchestrator and a hash-chained audit on every rig."""

    def __init__(self, *args: Any, feeds: list[Any] | None = None, **kwargs: Any) -> None:
        self.feeds = feeds if feeds is not None else default_feeds()
        self._wake = threading.Event()
        self._due: dict[str, float] = {}
        self._refreshed: dict[str, float] = {}
        self.bot: TelegramBot | None = None
        super().__init__(*args, **kwargs)
        self.bot = TelegramBot.from_env(self)

    def _install(self, scenario: Scenario | None, rig: Rig) -> None:
        super()._install(scenario, rig)
        self.live: dict[str, Any] = {"on": False, "feeds": {}}

        def emit(kind: str, data: dict[str, Any]) -> None:
            if rig is self.rig:
                self._broadcast(kind, data)
                if kind == "routera" and self.bot:
                    self.bot.kick()

        chain = HashChainAudit(clock=rig.clock, on_append=lambda entry: emit("audit", entry))
        rig.audit.subscribe(lambda e: chain.log(e.actor, e.action, e.details, e.rationale))
        orchestrator = Orchestrator(rig.sentinel, chain, emit, llm=self.llm)
        rig.bus.subscribe(orchestrator.on_bus)
        self.chain, self.orchestrator = chain, orchestrator

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            phone = self.bot.status() if self.bot else {"state": "off"}
            return {**super().snapshot(), "routera": self.orchestrator.snapshot(), "live": self.live, "phone": phone}

    # --- Live mode: real feeds into the same Sentinel, planner and audit -------------------------

    def start_replay(self, name: str, speed: float) -> dict[str, Any]:
        return self.start_live() if name == LIVE else super().start_replay(name, speed)

    def start_live(self) -> dict[str, Any]:
        self.stop_replay()
        stop = threading.Event()
        now = datetime.now(UTC).replace(microsecond=0)
        books = live_books(now)
        for feed in self.feeds:  # the new Sentinel has seen nothing yet
            if hasattr(feed, "reset"):
                feed.reset()
        with self._lock:
            rig = make_rig(None, now, books=books, llm=self.llm, config=self.config)
            self._install(live_scenario(now), rig)
            self.live["on"] = True
            self.progress.update(running=True, speed=1)
            self._stop = stop
            self._broadcast("reset", self.snapshot())
        thread = threading.Thread(target=self._run_live, args=(rig, stop), name="live-feeds", daemon=True)
        self._thread = thread
        thread.start()
        return {"live": True, "feeds": [f.name for f in self.feeds], "simulated_fleet": True}

    def _run_live(self, rig: Rig, stop: threading.Event) -> None:
        due = self._due = dict.fromkeys((f.name for f in self.feeds), 0.0)
        while not stop.is_set():
            for feed in self.feeds:
                if time.monotonic() < due[feed.name]:
                    continue
                due[feed.name] = time.monotonic() + feed.interval_s
                events, state = feed.poll(datetime.now(UTC))  # network: outside the lock
                with self._lock:
                    if rig is not self.rig:
                        return
                    self._advance(rig)
                    for raw in events:
                        try:
                            self.ingest(raw)
                        except Exception:  # one bad item must not stop the feeds
                            log.exception("live event %s failed", raw.get("event_id"))
                    self.live["feeds"][feed.name] = {**state, "interval_s": feed.interval_s}
                    if "slow" in state:
                        self.orchestrator.set_live_slow(state["slow"])
                    self._broadcast("feeds", self.live)
            with self._lock:
                if rig is not self.rig:
                    return
                self._advance(rig)
                rig.sentinel.tick()
                self._broadcast("routera", self.orchestrator.snapshot())
            for _ in range(int(LIVE_TICK_S)):
                if stop.wait(1.0) or self._wake.is_set():
                    break
            self._wake.clear()

    def _ops_event(self, place: str, status: str, reason: str) -> dict[str, Any]:
        """The Sentinel event for a closure a person reports, port or road by what the place is."""
        network = self.rig.sentinel.network
        on_road = network.corridors_through(place) or place in {c.corridor_id for c in network.corridors}
        now = self.rig.clock.now().isoformat()
        payload = {"status": status, "reason": reason, "effective_from": now, "expected_reopen": None}
        common = {"event_id": f"OPS-{place}-{status}-{now}", "received_at": now}
        if on_road:
            return {**common, "source": "ops_road", "kind_hint": "ROAD_STATUS",
                    "payload": {"segment": place, **payload}}
        return {**common, "source": "ops_port", "kind_hint": "PORT_STATUS",
                "payload": {"port_code": place, **payload}}

    def open_decision(self, place: str, person: str, reason: str) -> dict[str, Any]:
        """A named person treats a real report as a real closure. News alone still never confirms anything;
        a person can, on the record. This only opens a plan: it is approved or rejected like any other."""
        with self._lock:
            signal, _ = self.ingest(self._ops_event(place, "closed", f"{person}: {reason}"))
            disruption_id = (signal or {}).get("disruption_id")
            self.orchestrator.chain.log(person, "decision.opened",
                                        {"place": place, "reason": reason, "disruption_id": disruption_id},
                                        f"{person} treated {place} as closed and opened a decision, on: {reason}")
            plan_id = self.orchestrator.plan_for_root.get(disruption_id or "")
            plan = self.orchestrator.plans.get(plan_id or "")
            return {"place": place, "disruption_id": disruption_id, "plan_id": plan_id,
                    "containers": len(plan["rows"]) if plan else 0}

    def close_decision(self, place: str, person: str) -> dict[str, Any]:
        """The undo: the place is open again, and a plan nobody approved closes by itself."""
        with self._lock:
            signal, _ = self.ingest(self._ops_event(place, "open", f"{person}: open again"))
            self.orchestrator.chain.log(person, "decision.closed", {"place": place},
                                        f"{person} marked {place} open again")
            return {"place": place, "disruption_id": (signal or {}).get("disruption_id")}

    def refresh_feed(self, name: str) -> dict[str, Any]:
        """Poll one feed now instead of waiting for its turn."""
        if name not in {f.name for f in self.feeds}:
            raise KeyError(name)
        if not self.live.get("on"):
            raise ValueError("live mode is off")
        wait = REFRESH_MIN_S - (time.monotonic() - self._refreshed.get(name, -REFRESH_MIN_S))
        if wait > 0:
            raise OverflowError(f"{name} was refreshed moments ago; try again in {wait:.0f} s")
        self._refreshed[name] = time.monotonic()
        self._due[name] = 0.0
        self._wake.set()
        return {"feed": name, "refreshing": True}

    @staticmethod
    def _advance(rig: Rig) -> None:
        rig.clock.set(max(rig.clock.now(), datetime.now(UTC)))

    def run(self, fn: Any, *args: Any) -> Any:
        with self._lock:
            return fn(*args)


def create_routera_app(service: RouteraService | None = None, access: Access | None = None) -> FastAPI:
    service = service or RouteraService()
    access = access or Access()
    app = create_app(service)
    app.title = "ROUTERA (SIMULATED FEEDS)"
    app.router.routes = [r for r in app.router.routes if getattr(r, "path", None) != "/"]

    def session_of(request: Request) -> dict[str, str] | None:
        return access.read(request.cookies.get(COOKIE))

    def person(request: Request, body: dict[str, Any]) -> str:
        """The signed-in person, as the audit log names them. Without a session (no code set), the body's."""
        session = session_of(request)
        return Access.person(session) if session else str(body.get("approver") or "Sunil P.")[:60]

    @app.middleware("http")
    async def sign_in_first(request: Request, call_next: Any) -> Any:
        if not access.required or request.url.path in SIGN_IN_PATHS or session_of(request):
            return await call_next(request)
        if request.method == "GET" and "text/html" in request.headers.get("accept", ""):
            return RedirectResponse("/", status_code=303)
        return JSONResponse({"detail": "Sign in first, with your name and the team access code"}, status_code=401)

    @app.get("/", include_in_schema=False)
    def intro() -> FileResponse:
        return FileResponse(STATIC / "intro.html", headers=NO_STORE)

    @app.get("/room", include_in_schema=False, response_model=None)
    def room(request: Request) -> Response:
        if not session_of(request):
            return RedirectResponse("/", status_code=303)
        return FileResponse(STATIC / "index.html", headers=NO_STORE)

    @app.get("/api/session")
    def who(request: Request) -> dict[str, Any]:
        session = session_of(request)
        return {"signed_in": bool(session), **(session or {}), "code_required": access.required, "roles": ROLES}

    @app.post("/api/session")
    def sign_in(request: Request, response: Response, body: dict[str, Any] = OPTIONAL_BODY) -> dict[str, Any]:
        name = Access.clean_name(body.get("name"))
        if not name:
            raise HTTPException(422, "Please add your name. It goes on the audit log.")
        role = str(body.get("role") or "")
        if role not in ROLES:
            raise HTTPException(422, "Pick your role.")
        if access.required and not session_of(request):  # someone already signed in may change name or role
            client = request.client.host if request.client else "?"
            if access.blocked(client):
                raise HTTPException(429, "Too many wrong codes. Wait five minutes, then try again.")
            if not access.code_ok(body.get("code")):
                access.missed(client)
                raise HTTPException(401, "That access code isn't right. Ask your team lead for it.")
        secure = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
        response.set_cookie(COOKIE, access.issue(name, role), max_age=SESSION_S, httponly=True, samesite="lax",
                            secure=secure)
        return {"signed_in": True, "name": name, "role": role, "role_label": ROLES[role]}

    @app.delete("/api/session")
    def sign_out(response: Response) -> dict[str, bool]:
        response.delete_cookie(COOKIE)
        return {"signed_in": False}

    @app.get("/sentinel", include_in_schema=False)
    def sentinel_console() -> FileResponse:
        return FileResponse(SENTINEL_STATIC / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        return service.snapshot()

    @app.post("/api/plans/{plan_id}/approve")
    def approve(plan_id: str, request: Request, body: dict[str, Any] = OPTIONAL_BODY) -> dict[str, Any]:
        approver = person(request, body)
        try:
            plan = service.run(service.orchestrator.approve, plan_id, approver, body.get("overrides") or {})
        except KeyError as exc:
            raise HTTPException(404, f"unknown plan or container {exc}") from exc
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"plan_id": plan_id, "status": plan["status"], "saved_inr": plan["saved_inr"]}

    @app.post("/api/plans/{plan_id}/reject")
    def reject(plan_id: str, request: Request, body: dict[str, Any] = OPTIONAL_BODY) -> dict[str, Any]:
        approver = person(request, body)
        try:
            plan = service.run(service.orchestrator.reject, plan_id, approver, str(body.get("reason") or ""))
        except KeyError as exc:
            raise HTTPException(404, f"unknown plan {exc}") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"plan_id": plan_id, "status": plan["status"]}

    @app.get("/api/plans/{plan_id}/why")
    def why(plan_id: str, container: str = Query(...)) -> dict[str, str]:
        try:
            return service.run(service.orchestrator.why, plan_id, container)
        except (KeyError, StopIteration) as exc:
            raise HTTPException(404, f"no {container} in {plan_id}") from exc

    @app.get("/api/plans/{plan_id}/paperwork")
    def paperwork(plan_id: str, container: str = Query(...), route: str | None = None) -> dict[str, Any]:
        try:
            return service.run(service.orchestrator.paperwork, plan_id, container, route)
        except (KeyError, StopIteration) as exc:
            raise HTTPException(404, f"no {container} in {plan_id}") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/plans/{plan_id}/paperwork/{doc_type}.pdf")
    def paperwork_pdf(plan_id: str, doc_type: str, container: str = Query(...), route: str | None = None) -> Response:
        if not pdf_available():
            raise HTTPException(503, "PDFs need reportlab: pip install reportlab")
        pack = paperwork(plan_id, container, route)
        doc = next((d for d in pack["documents"] if d["doc_type"] == doc_type), None)
        if doc is None:
            raise HTTPException(404, f"{doc_type} is not redrafted for {container}")
        name = f"{plan_id}-{container}-{doc_type}.pdf"
        return Response(render_pdf(doc, pack), media_type="application/pdf",
                        headers={"Content-Disposition": f'inline; filename="{name}"'})

    @app.get("/api/plans/{plan_id}/filing-pack.zip")
    def filing_pack(plan_id: str, container: str = Query(...), route: str | None = None) -> Response:
        name = f"{plan_id}-{container}-filing-pack.zip"
        return Response(build_zip(paperwork(plan_id, container, route)), media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @app.get("/api/traffic/tile/{z}/{x}/{y}.png")
    def traffic_tile(z: int, x: int, y: int, dark: bool = False) -> Response:
        feed = next((f for f in service.feeds if isinstance(f, TrafficFeed)), None)
        if feed is None:
            raise HTTPException(404, "no traffic feed")
        try:
            data = feed.tile(z, x, y, dark)
        except LookupError as exc:
            raise HTTPException(404, "no TomTom key") from exc
        except ValueError as exc:
            raise HTTPException(422, "no such tile") from exc
        except Exception as exc:  # the message could carry the key
            raise HTTPException(502, "traffic tile unavailable") from exc
        return Response(data, media_type="image/png", headers={"Cache-Control": "public, max-age=300"})

    def known_place(place: str) -> str:
        known = {*load_geo()["places"], *(c.corridor_id for c in service.orchestrator.sentinel.network.corridors)}
        if place not in known:
            raise HTTPException(404, f"unknown place {place}")
        return place

    @app.post("/api/decisions/open", status_code=201)
    def open_decision(request: Request, body: dict[str, Any] = OPTIONAL_BODY) -> dict[str, Any]:
        reason = str(body.get("reason") or "reported closure")[:200]
        return service.open_decision(known_place(str(body.get("place"))), person(request, body), reason)

    @app.post("/api/decisions/close")
    def close_decision(request: Request, body: dict[str, Any] = OPTIONAL_BODY) -> dict[str, Any]:
        return service.close_decision(known_place(str(body.get("place"))), person(request, body))

    @app.post("/api/feeds/{name}/refresh", status_code=202)
    def refresh_feed(name: str) -> dict[str, Any]:
        try:
            return service.refresh_feed(name)
        except KeyError as exc:
            raise HTTPException(404, f"no feed {name}") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        except OverflowError as exc:
            raise HTTPException(429, str(exc)) from exc

    @app.get("/api/whatif")
    def what_if(place: str = Query(...)) -> dict[str, Any]:
        return service.run(service.orchestrator.what_if, known_place(place))

    @app.get("/api/map")
    def map_data() -> dict[str, Any]:
        return load_geo()

    @app.get("/api/containers/{container_id}/options")
    def container_options(container_id: str) -> dict[str, Any]:
        try:
            return service.run(service.orchestrator.options, container_id)
        except KeyError as exc:
            raise HTTPException(404, f"no container {container_id}") from exc

    @app.get("/api/rules")
    def rules() -> dict[str, Any]:
        book = service.orchestrator.planner.rules.book
        today = datetime.now(UTC).date()
        cited = [r.cite(today) for r in book.all()]
        return {"rules": cited, "warnings": book.warnings, "to_verify": sum(bool(r["stale"]) for r in cited),
                "pdfs": pdf_available()}

    @app.get("/api/rules/to-verify", response_class=PlainTextResponse)
    def rules_to_verify() -> str:
        return service.orchestrator.planner.rules.book.to_verify_markdown(datetime.now(UTC).date())

    @app.post("/api/live", status_code=202)
    def live() -> dict[str, Any]:
        return service.start_live()

    @app.get("/api/audit")
    def audit() -> dict[str, Any]:
        return service.run(lambda: {**service.chain.verify(), "entries": service.chain.entries})

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m routera", description="ROUTERA control room")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--demo", default=SCENARIO_NAMES[0], choices=[*SCENARIO_NAMES, "none"],
                        help="scenario to play when the page first connects (none: idle)")
    parser.add_argument("--speed", type=float, default=60)
    parser.add_argument("--live", action="store_true",
                        help="start in Live mode: real weather and news instead of a scenario")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--llm", choices=LLM_MODES, help="override SENTINEL_LLM (default off)")
    args = parser.parse_args(argv)

    import uvicorn

    llm = build_llm_client(args.llm)
    service = RouteraService(llm=llm.client, llm_label=llm.label)
    access = Access(os.environ.get("ROUTERA_ACCESS_CODE"), os.environ.get("ROUTERA_SESSION_SECRET"))
    if args.live:
        service.autostart = (LIVE, 1)
    elif args.demo != "none":
        service.autostart = (args.demo, args.speed)
    url = f"http://localhost:{args.port}"
    print(f"ROUTERA · SIMULATED FEEDS · {url}   (LLM: {llm.label})")
    print("Sign-in: the team access code is required." if access.required
          else "Sign-in: name and role only. Set ROUTERA_ACCESS_CODE to require a team access code.")
    if service.bot:
        service.bot.start()
        print("Phone approvals are on: the control room shows how to connect (Sunil's phone panel).")
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    uvicorn.run(create_routera_app(service, access), host=args.host, port=args.port, log_level="warning")
    return 0
