"""ROUTERA control room: Sentinel + Shelf-Life + Planner + Approval & Audit, live in one app.

    python -m routera            # http://localhost:8000, plays the India-inside scenario

Everything from agents/sentinel/api.py (REST + the SSE stream) plus:
    GET  /                            the ROUTERA control room
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
import threading
import time
import webbrowser
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse, Response

from agents.sentinel.api import STATIC_DIR as SENTINEL_STATIC
from agents.sentinel.api import SentinelService, create_app
from agents.sentinel.llm_client import LLM_MODES, build_llm_client
from agents.sentinel.replay import SCENARIO_NAMES, Rig, Scenario, make_rig
from routera.audit import HashChainAudit
from routera.filing import build_zip, pdf_available, render_pdf
from routera.live_feeds import default_feeds, live_books
from routera.orchestrator import Orchestrator

STATIC = Path(__file__).parent / "static"
DEFAULT_PORT = 8000
OPTIONAL_BODY = Body(default={})
LIVE = "live"
LIVE_TICK_S = 60.0
log = logging.getLogger("routera.live")


def live_scenario(now: datetime) -> Scenario:
    return Scenario(
        name=LIVE, title="LIVE: real weather, news and ships for Nashik grapes and Konkan mangoes",
        simulated=True, layer="india", start=now, container_ids=[], steps=[],
        disclaimer="Feeds are REAL (Open-Meteo, Google News; AIS with a free key). The fleet is the "
                   "SIMULATED book moved to today; bookings stay simulated.",
    )


class RouteraService(SentinelService):
    """The Sentinel service, with the orchestrator and a hash-chained audit on every rig."""

    def __init__(self, *args: Any, feeds: list[Any] | None = None, **kwargs: Any) -> None:
        self.feeds = feeds if feeds is not None else default_feeds()
        super().__init__(*args, **kwargs)

    def _install(self, scenario: Scenario | None, rig: Rig) -> None:
        super()._install(scenario, rig)
        self.live: dict[str, Any] = {"on": False, "feeds": {}}

        def emit(kind: str, data: dict[str, Any]) -> None:
            if rig is self.rig:
                self._broadcast(kind, data)

        chain = HashChainAudit(clock=rig.clock, on_append=lambda entry: emit("audit", entry))
        rig.audit.subscribe(lambda e: chain.log(e.actor, e.action, e.details, e.rationale))
        orchestrator = Orchestrator(rig.sentinel, chain, emit, llm=self.llm)
        rig.bus.subscribe(orchestrator.on_bus)
        self.chain, self.orchestrator = chain, orchestrator

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {**super().snapshot(), "routera": self.orchestrator.snapshot(), "live": self.live}

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
        due = dict.fromkeys((f.name for f in self.feeds), 0.0)
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
                    self.live["feeds"][feed.name] = state
                    self._broadcast("feeds", self.live)
            with self._lock:
                if rig is not self.rig:
                    return
                self._advance(rig)
                rig.sentinel.tick()
                self._broadcast("routera", self.orchestrator.snapshot())
            stop.wait(LIVE_TICK_S)

    @staticmethod
    def _advance(rig: Rig) -> None:
        rig.clock.set(max(rig.clock.now(), datetime.now(UTC)))

    def run(self, fn: Any, *args: Any) -> Any:
        with self._lock:
            return fn(*args)


def create_routera_app(service: RouteraService | None = None) -> FastAPI:
    service = service or RouteraService()
    app = create_app(service)
    app.title = "ROUTERA (SIMULATED FEEDS)"
    app.router.routes = [r for r in app.router.routes if getattr(r, "path", None) != "/"]

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/sentinel", include_in_schema=False)
    def sentinel_console() -> FileResponse:
        return FileResponse(SENTINEL_STATIC / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        return service.snapshot()

    @app.post("/api/plans/{plan_id}/approve")
    def approve(plan_id: str, body: dict[str, Any] = OPTIONAL_BODY) -> dict[str, Any]:
        approver = str(body.get("approver") or "Sunil P.")
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
    def reject(plan_id: str, body: dict[str, Any] = OPTIONAL_BODY) -> dict[str, Any]:
        approver = str(body.get("approver") or "Sunil P.")
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
    if args.live:
        service.autostart = (LIVE, 1)
    elif args.demo != "none":
        service.autostart = (args.demo, args.speed)
    url = f"http://localhost:{args.port}"
    print(f"ROUTERA · SIMULATED FEEDS · {url}   (LLM: {llm.label})")
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    uvicorn.run(create_routera_app(service), host=args.host, port=args.port, log_level="warning")
    return 0
