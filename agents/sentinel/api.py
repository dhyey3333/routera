"""FastAPI for the Disruption Sentinel prototype: REST + SSE + the static UI (spec sections 5, 7).

    python -m agents.sentinel.api --demo                 # what `make sentinel-demo` runs

serves http://localhost:8001 and plays the India-inside Nashik corridor scenario (truck strike and
rain on the road to JNPT) as soon as the page connects.
Everything runs offline on SIMULATED feeds.

    POST /sentinel/events          ingest one RawEvent (received_at defaults to the sim clock)
    GET  /sentinel/disruptions     Disruption contract objects (?scope=active|all)
    GET  /sentinel/stream          SSE: a "snapshot" first, then signal / decision / publish /
                                   disruption / step / replay / reset events
    POST /sentinel/replay/{name}   restart on a scenario (?speed=1|10|60 sim min per real min)
    GET  /sentinel/state           the same snapshot as the stream's first event
    GET  /sentinel/scenarios       scenarios for the UI dropdown
    POST /sentinel/tick            run the clock-driven lifecycle now
    GET  /                         static/index.html
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
import webbrowser
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import ValidationError

from agents.sentinel.fakes import ALL_CONTAINER_PATHS
from agents.sentinel.llm_client import LLM_MODES, build_llm_client
from agents.sentinel.models import Disruption, LLMClient, RawEvent
from agents.sentinel.replay import (
    SCENARIO_NAMES,
    Rig,
    Scenario,
    load_scenario,
    make_rig,
    replay,
    rig_for,
)
from agents.sentinel.settings import SentinelConfig, load_config

STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_PORT = 8001
HEARTBEAT_SECONDS = 15.0

Item = tuple[int, str, dict[str, Any]]


def _sse(seq: int, kind: str, data: dict[str, Any]) -> str:
    return f"id: {seq}\nevent: {kind}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


class SentinelService:
    """Holds the current Sentinel and fans its events out to every SSE client.

    One re-entrant lock guards the Sentinel: request handlers and the replay thread take it
    around each ingest/tick, so events reach clients in the order they happened.
    """

    def __init__(
        self,
        *,
        config: SentinelConfig | None = None,
        llm: LLMClient | None = None,
        llm_label: str = "off",
    ) -> None:
        self.config = config or load_config()
        self.llm = llm
        self.llm_label = llm_label
        self.autostart: tuple[str, float] | None = None  # (scenario, speed) on first viewer
        self._lock = threading.RLock()
        self._queues: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue[Item]]] = []
        self._seq = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        idle = load_scenario(SCENARIO_NAMES[0])  # before any replay: full book, same start time
        # Before any replay: both container books, so POSTed events of either layer match.
        idle_rig = make_rig(
            None, idle.start, books=ALL_CONTAINER_PATHS, llm=llm, config=self.config
        )
        self._install(None, idle_rig)

    # --- state ----------------------------------------------------------------------------------

    def _install(self, scenario: Scenario | None, rig: Rig) -> None:
        self.scenario = scenario
        self.rig = rig
        self.signals: list[dict[str, Any]] = []
        self.audit: list[dict[str, Any]] = []
        self.published = 0
        self.topics: dict[str, str] = {}
        self.progress: dict[str, Any] = {
            "running": False,
            "speed": None,
            "step": 0,
            "steps": len(scenario.steps) if scenario else 0,
        }
        rig.sentinel.subscribe(lambda kind, data: self._on_sentinel(rig, kind, data))

    def _on_sentinel(self, rig: Rig, kind: str, data: dict[str, Any]) -> None:
        with self._lock:
            if rig is not self.rig:
                return  # a superseded replay finishing its last step
            if kind == "signal":
                self.signals.append(data)
            elif kind == "decision":
                self.audit.append(data)
            elif kind == "publish":
                self.published += 1
                self.topics[data["disruption"]["disruption_id"]] = data["topic"]
            elif kind == "disruption":
                did = data["disruption"]["disruption_id"]
                data = {**data, "last_topic": self.topics.get(did)}
            self._broadcast(kind, data)

    def _broadcast(self, kind: str, data: dict[str, Any]) -> None:
        with self._lock:
            self._seq += 1
            item = (self._seq, kind, data)
            for loop, queue in list(self._queues):
                try:
                    loop.call_soon_threadsafe(queue.put_nowait, item)
                except RuntimeError:  # that client's event loop is gone
                    self._queues.remove((loop, queue))

    def _describe(self, disruption_id: str) -> dict[str, Any]:
        return {
            **self.rig.sentinel.describe(disruption_id),
            "last_topic": self.topics.get(disruption_id),
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            scenario = self.scenario
            return {
                "simulated": True,
                "scenario": None
                if scenario is None
                else {
                    "name": scenario.name,
                    "title": scenario.title,
                    "disclaimer": scenario.disclaimer,
                    "start": scenario.start.isoformat(),
                },
                "scenarios": scenario_list(),
                "replay": dict(self.progress),
                "clock": self.rig.clock.now().isoformat(),
                "signals": list(self.signals),
                "disruptions": [
                    self._describe(d.disruption_id) for d in self.rig.sentinel.disruptions()
                ],
                "audit": list(self.audit),
                "published": self.published,
                "llm": self.llm_label,
            }

    # --- commands -------------------------------------------------------------------------------

    def ingest(self, raw: dict[str, Any]) -> tuple[dict[str, Any] | None, list[Disruption]]:
        """Ingest one event; returns the Sentinel's signal record and the disruptions touched."""
        with self._lock:
            raw = dict(raw)
            raw.setdefault("received_at", self.rig.clock.now().isoformat())
            try:
                received = RawEvent.model_validate(raw).received_at
            except ValidationError:
                received = None  # the Sentinel will reject and audit it
            if received is not None and received > self.rig.clock.now():
                self.rig.clock.set(received)
            before = len(self.signals)
            views = self.rig.sentinel.ingest(raw)
            signal = self.signals[before] if len(self.signals) > before else None
            return signal, views

    def tick(self) -> list[Disruption]:
        with self._lock:
            return self.rig.sentinel.tick()

    def disruptions(self, scope: Literal["active", "all"]) -> list[Disruption]:
        with self._lock:
            sentinel = self.rig.sentinel
            return sentinel.active() if scope == "active" else sentinel.disruptions()

    def start_replay(self, name: str, speed: float) -> dict[str, Any]:
        scenario = load_scenario(name)
        self.stop_replay()
        stop = threading.Event()
        with self._lock:
            rig = rig_for(scenario, llm=self.llm, config=self.config)
            self._install(scenario, rig)
            self.progress.update(running=True, speed=speed)
            self._stop = stop
            self._broadcast("reset", self.snapshot())
        thread = threading.Thread(
            target=self._run, args=(scenario, rig, speed, stop), name=f"replay-{name}", daemon=True
        )
        self._thread = thread
        thread.start()
        return {"scenario": name, "speed": speed, "steps": len(scenario.steps), "simulated": True}

    def _run(self, scenario: Scenario, rig: Rig, speed: float, stop: threading.Event) -> None:
        def on_step(kind: str, data: dict[str, Any]) -> None:
            if kind == "step" and rig is self.rig:
                self.progress["step"] = data["index"]
                self._broadcast("step", data)

        try:
            replay(scenario, rig, speed=speed, stop=stop, lock=self._lock, on_event=on_step)
        finally:
            with self._lock:
                if rig is self.rig:
                    self.progress["running"] = False
                    self._broadcast("replay", dict(self.progress))

    def stop_replay(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=5)

    def wait_for_replay(self, timeout: float = 10.0) -> None:
        """Block until the current replay finishes (tests use speed 0)."""
        if self._thread is not None:
            self._thread.join(timeout)

    # --- SSE clients ----------------------------------------------------------------------------

    def subscribe(self, loop: asyncio.AbstractEventLoop) -> tuple[asyncio.Queue[Item], Item]:
        queue: asyncio.Queue[Item] = asyncio.Queue()
        with self._lock:
            self._queues.append((loop, queue))
            self._seq += 1
            first = (self._seq, "snapshot", self.snapshot())
            autostart, self.autostart = self.autostart, None
        if autostart is not None:
            self.start_replay(*autostart)
        return queue, first

    def unsubscribe(self, queue: asyncio.Queue[Item]) -> None:
        with self._lock:
            self._queues = [(lp, q) for lp, q in self._queues if q is not queue]


def scenario_list() -> list[dict[str, str]]:
    return [{"name": s.name, "title": s.title} for s in map(load_scenario, SCENARIO_NAMES)]


def create_app(service: SentinelService | None = None) -> FastAPI:
    service = service or SentinelService()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        service.stop_replay()

    app = FastAPI(
        title="ROUTERA Disruption Sentinel (SIMULATED FEEDS)",
        version="0.4.0",
        lifespan=lifespan,
    )
    app.state.service = service

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})

    @app.post("/sentinel/events")
    def post_event(event: dict[str, Any] = Body(...)) -> JSONResponse:  # noqa: B008
        signal, views = service.ingest(event)
        rejected = signal is not None and signal["status"] == "rejected"
        body = {
            "simulated": True,
            "signal": signal,
            "disruptions": [v.model_dump(mode="json") for v in views],
        }
        return JSONResponse(body, status_code=422 if rejected else 200)

    @app.get("/sentinel/disruptions", response_model=list[Disruption])
    def get_disruptions(scope: Literal["active", "all"] = "active") -> list[Disruption]:
        return service.disruptions(scope)

    @app.get("/sentinel/state")
    def get_state() -> dict[str, Any]:
        return service.snapshot()

    @app.get("/sentinel/scenarios")
    def get_scenarios() -> list[dict[str, str]]:
        return scenario_list()

    @app.post("/sentinel/replay/{name}", status_code=202)
    def post_replay(name: str, speed: float = Query(60, ge=0, le=6000)) -> dict[str, Any]:
        if name not in SCENARIO_NAMES:
            raise HTTPException(404, f"unknown scenario {name!r}; known: {list(SCENARIO_NAMES)}")
        return service.start_replay(name, speed)

    @app.post("/sentinel/tick", response_model=list[Disruption])
    def post_tick() -> list[Disruption]:
        return service.tick()

    @app.get("/sentinel/stream")
    async def stream(
        request: Request, limit: int | None = Query(None, ge=1, description="debug: stop after N")
    ) -> StreamingResponse:
        queue, first = service.subscribe(asyncio.get_running_loop())

        async def events() -> AsyncIterator[str]:
            sent = 0
            try:
                item = first
                while True:
                    yield _sse(*item)
                    sent += 1
                    if limit is not None and sent >= limit:
                        return
                    while True:
                        if await request.is_disconnected():
                            return
                        try:
                            item = await asyncio.wait_for(queue.get(), HEARTBEAT_SECONDS)
                            break
                        except TimeoutError:
                            yield ": keep-alive\n\n"
            finally:
                service.unsubscribe(queue)

        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        return StreamingResponse(events(), media_type="text/event-stream", headers=headers)

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agents.sentinel.api",
        description="Serve the Disruption Sentinel prototype UI and API (SIMULATED FEEDS).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--demo",
        nargs="?",
        const=SCENARIO_NAMES[0],
        choices=SCENARIO_NAMES,
        help=f"replay this scenario when the page first connects (default {SCENARIO_NAMES[0]})",
    )
    parser.add_argument("--speed", type=float, default=60, help="demo replay speed (default 60)")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    parser.add_argument("--llm", choices=LLM_MODES, help="override SENTINEL_LLM (default off)")
    args = parser.parse_args(argv)

    import uvicorn  # imported here so the module stays importable without a server

    llm = build_llm_client(args.llm)
    service = SentinelService(llm=llm.client, llm_label=llm.label)
    if args.demo:
        service.autostart = (args.demo, args.speed)
    url = f"http://localhost:{args.port}"
    print(f"Disruption Sentinel · SIMULATED FEEDS · {url}")
    print(f"LLM: {llm.label}" + "".join(f"\n  {note}" for note in llm.notes))
    if args.demo:
        print(f"The {args.demo} replay ({args.speed:g}x) starts when the page connects.")
        if not args.no_browser:
            threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    uvicorn.run(create_app(service), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
