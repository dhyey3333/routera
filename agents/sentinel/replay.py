"""Scenario replayer with an accelerated clock, and the expected-output checker.

    python -m agents.sentinel.replay gulf_closure --speed 60
    python -m agents.sentinel.replay all --speed 0 --check

--speed 60 plays one simulated minute per real second (1 = real time). Fast-forwards, gaps of
an hour or more of simulated time like the 24 h rumour expiry, wait at most --max-gap seconds so
a demo never stalls. --check compares the run with data/expected/<name>.json and exits 1 on any
mismatch. Every feed is SIMULATED.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agents.sentinel.fakes import (
    DATA_DIR,
    DEFAULT_CONTAINERS_PATH,
    InMemoryAudit,
    InMemoryBus,
    JsonContainerRepo,
    SimClock,
)
from agents.sentinel.llm_client import LLM_MODES, build_llm_client
from agents.sentinel.models import BusMessage, Disruption, LLMClient, NetworkRepository, UtcDatetime
from agents.sentinel.sentinel import DisruptionSentinel
from agents.sentinel.settings import SentinelConfig
from agents.sentinel.summary import count

SCENARIO_DIR = DATA_DIR / "scenarios"
EXPECTED_DIR = DATA_DIR / "expected"
# India-inside first (v2); the global shocks run on the same engine.
SCENARIO_NAMES = (
    "nashik_corridor_strike",
    "gulf_ripple",
    "gulf_closure",
    "us_tariff_spike",
    "red_sea_reroute",
)

EventListener = Callable[[str, dict[str, Any]], None]


class ScenarioStep(BaseModel):
    """One feed message at t_min, or a clock tick (fast-forward) when tick is true.

    event stays a raw dict: a scenario may carry a malformed event on purpose to show rejection.
    """

    model_config = ConfigDict(extra="forbid")

    t_min: float = Field(ge=0)
    note: str = ""
    event: dict[str, Any] | None = None
    tick: bool = False

    @model_validator(mode="after")
    def _event_or_tick(self) -> ScenarioStep:
        if (self.event is not None) == self.tick:
            raise ValueError("a step has either an event or tick: true, not both")
        return self


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    title: str
    simulated: Literal[True]
    disclaimer: str
    layer: Literal["india", "global"] = "global"
    start: UtcDatetime
    containers_file: str = DEFAULT_CONTAINERS_PATH.name  # a file in data/
    container_ids: list[str]
    steps: list[ScenarioStep]

    @property
    def book_path(self) -> Path:
        return DATA_DIR / self.containers_file

    @model_validator(mode="after")
    def _steps_in_order(self) -> Scenario:
        times = [step.t_min for step in self.steps]
        if times != sorted(times):
            raise ValueError("steps must be sorted by t_min")
        return self

    def at(self, step: ScenarioStep) -> datetime:
        return self.start + timedelta(minutes=step.t_min)

    def raw_event(self, step: ScenarioStep) -> dict[str, Any]:
        """The step's event as a feed would send it, stamped with simulated received_at."""
        if step.event is None:
            raise ValueError("tick steps carry no event")
        return {**step.event, "received_at": self.at(step).isoformat()}


def _path(name_or_path: str | Path, folder: Path) -> Path:
    path = Path(name_or_path)
    return path if path.suffix == ".json" else folder / f"{name_or_path}.json"


def load_scenario(name_or_path: str | Path) -> Scenario:
    text = _path(name_or_path, SCENARIO_DIR).read_text(encoding="utf-8")
    return Scenario.model_validate_json(text)


def load_expected(name_or_path: str | Path) -> dict[str, Any]:
    return json.loads(_path(name_or_path, EXPECTED_DIR).read_text(encoding="utf-8"))


# --- running ----------------------------------------------------------------------------------


@dataclass
class Rig:
    """A Sentinel wired to the in-memory fakes."""

    sentinel: DisruptionSentinel
    bus: InMemoryBus
    audit: InMemoryAudit
    clock: SimClock


def make_rig(
    container_ids: list[str] | None,
    start: datetime,
    *,
    books: Path | str | Sequence[Path | str] = DEFAULT_CONTAINERS_PATH,
    llm: LLMClient | None = None,
    config: SentinelConfig | None = None,
    network: NetworkRepository | None = None,
) -> Rig:
    bus, audit, clock = InMemoryBus(), InMemoryAudit(), SimClock(start)
    repo = JsonContainerRepo(books, container_ids=container_ids)
    sentinel = DisruptionSentinel(repo, bus, audit, clock, llm=llm, config=config, network=network)
    return Rig(sentinel, bus, audit, clock)


def rig_for(
    scenario: Scenario, *, llm: LLMClient | None = None, config: SentinelConfig | None = None
) -> Rig:
    """A fresh Sentinel over the scenario's own container book, clock at its start."""
    return make_rig(
        scenario.container_ids, scenario.start, books=scenario.book_path, llm=llm, config=config
    )


@dataclass
class StepOutcome:
    step: ScenarioStep
    at: datetime
    signal: dict[str, Any] | None = None
    decisions: list[dict[str, Any]] = field(default_factory=list)
    published: list[dict[str, Any]] = field(default_factory=list)
    views: dict[str, Disruption] = field(default_factory=dict)  # every disruption after the step
    seconds: float = 0.0  # wall time of the ingest or tick


@dataclass
class ReplayResult:
    scenario: Scenario
    rig: Rig
    steps: list[StepOutcome]
    stopped: bool = False


def pause_seconds(gap_min: float, speed: float, max_gap: float, fast_forward_after: float) -> float:
    """Real seconds to wait for a simulated gap. Fast-forwards (long idle gaps) are capped."""
    if speed <= 0:
        return 0.0
    seconds = gap_min * 60 / speed
    return min(seconds, max_gap) if gap_min >= fast_forward_after else seconds


def replay(
    scenario: Scenario,
    rig: Rig | None = None,
    *,
    speed: float = 0,
    max_gap: float = 3.0,
    fast_forward_after: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
    stop: threading.Event | None = None,
    lock: AbstractContextManager[Any] | None = None,
    on_event: EventListener | None = None,
) -> ReplayResult:
    """Feed every step through a Sentinel on the scenario's simulated clock.

    speed is simulated minutes per real minute (0 = no waiting). Gaps of fast_forward_after
    simulated minutes or more wait at most max_gap seconds. stop ends the replay early; lock is
    held around each step (the API shares the Sentinel with request handlers). on_event receives
    ("step", ...) before each step, then the Sentinel's own events.
    """
    rig = rig or rig_for(scenario)
    outcomes: list[StepOutcome] = []
    guard = lock if lock is not None else contextlib.nullcontext()

    def wait(seconds: float) -> bool:
        """Sleep; True when asked to stop."""
        if stop is None:
            sleep(seconds)
            return False
        return stop.wait(seconds)

    def listen(kind: str, data: dict[str, Any]) -> None:
        current = outcomes[-1]
        if kind == "signal":
            current.signal = data
        elif kind == "decision":
            current.decisions.append(data)
        elif kind == "publish":
            current.published.append(data)
        if on_event is not None:
            on_event(kind, data)

    unsubscribe = rig.sentinel.subscribe(listen)
    stopped = False
    try:
        previous: float | None = None
        for index, step in enumerate(scenario.steps):
            if previous is not None and speed > 0:
                gap = step.t_min - previous
                if wait(pause_seconds(gap, speed, max_gap, fast_forward_after)):
                    stopped = True
                    break
            if stop is not None and stop.is_set():
                stopped = True
                break
            previous = step.t_min
            with guard:
                rig.clock.set(scenario.at(step))
                outcomes.append(StepOutcome(step, scenario.at(step)))
                if on_event is not None:
                    on_event(
                        "step",
                        {
                            "t_min": step.t_min,
                            "note": step.note,
                            "tick": step.tick,
                            "at": scenario.at(step).isoformat(),
                            "index": index + 1,
                            "steps": len(scenario.steps),
                        },
                    )
                started = time.perf_counter()
                if step.tick:
                    rig.sentinel.tick()
                else:
                    rig.sentinel.ingest(scenario.raw_event(step))
                outcomes[-1].seconds = time.perf_counter() - started
                outcomes[-1].views = {d.disruption_id: d for d in rig.sentinel.disruptions()}
    finally:
        unsubscribe()
    return ReplayResult(scenario, rig, outcomes, stopped)


# --- checking against data/expected -----------------------------------------------------------

_EXACT_KEYS = (
    "type",
    "location",
    "status",
    "severity",
    "priority",
    "market_unreachable",
    "expected_delay_days",
    "delay_is_estimate",
    "cost_impact_pct",
    "affected_container_ids",
    "affected_weight_kg",
)


def check_disruption(view: Disruption | None, expected: Mapping[str, Any]) -> list[str]:
    """Differences between a Disruption and a labelled row (only the keys the row has)."""
    if view is None:
        return [f"{expected.get('disruption_id')} missing"]
    name = view.disruption_id
    problems = [
        f"{name}.{key} = {getattr(view, key)!r}, expected {expected[key]!r}"
        for key in _EXACT_KEYS
        if key in expected and getattr(view, key) != expected[key]
    ]
    if "severity_range" in expected:
        low, high = expected["severity_range"]
        if not low <= view.severity <= high:
            problems.append(f"{name}.severity = {view.severity}, expected {low}-{high}")
    if (
        "affected_count" in expected
        and len(view.affected_container_ids) != expected["affected_count"]
    ):
        problems.append(
            f"{name} affects {count(len(view.affected_container_ids))}, "
            f"expected {expected['affected_count']}"
        )
    if (
        "confidence_approx" in expected
        and abs(view.confidence - expected["confidence_approx"]) > 0.01
    ):
        problems.append(
            f"{name}.confidence = {view.confidence}, expected ~{expected['confidence_approx']}"
        )
    if "confidence_min" in expected and view.confidence < expected["confidence_min"]:
        problems.append(f"{name}.confidence = {view.confidence} < {expected['confidence_min']}")
    if "source_event_ids" in expected:
        got = [s.event_id for s in view.sources]
        if got != expected["source_event_ids"]:
            problems.append(f"{name}.sources = {got}, expected {expected['source_event_ids']}")
    return problems


def compare_with_expected(result: ReplayResult, expected: Mapping[str, Any]) -> list[str]:
    """Every way a replay differs from its labelled expected file; empty means it matches."""
    problems: list[str] = []
    rows = expected["timeline"]
    if len(rows) != len(result.steps):
        return [f"{len(result.steps)} steps replayed, {len(rows)} labelled"]

    previous: dict[str, Disruption] = {}
    for outcome, row in zip(result.steps, rows, strict=True):
        where = f"t={row['t_min']:g}"
        audit = [d["action"] for d in outcome.decisions]
        if audit != row["audit"]:
            problems.append(f"{where}: audit {audit}, expected {row['audit']}")
        topics = [m["topic"] for m in outcome.published]
        wanted = row.get("published") or []
        if topics != (wanted if isinstance(wanted, list) else [wanted]):
            problems.append(f"{where}: published {topics}, expected {row.get('published')}")
        if row.get("rejected"):
            signal = outcome.signal or {}
            if signal.get("status") != "rejected" or row["reason_contains"] not in signal["reason"]:
                problems.append(
                    f"{where}: {row['event_id']} not rejected for {row['reason_contains']!r}"
                )
            continue
        ids = {d["disruption_id"] for d in outcome.decisions}
        expected_ids = set(row.get("disruption_ids") or [row["disruption_id"]])
        if ids != expected_ids:
            problems.append(
                f"{where}: decisions touch {sorted(map(str, ids))}, expected {sorted(expected_ids)}"
            )
        if "disruption_ids" in row:
            previous = outcome.views
            continue
        view = outcome.views.get(row["disruption_id"])
        problems += [f"{where}: {p}" for p in check_disruption(view, row)]
        before = previous.get(row["disruption_id"])
        if row.get("confidence_increases") and (
            view is None or before is None or view.confidence <= before.confidence
        ):
            problems.append(f"{where}: confidence did not increase")
        if "reason_contains" in row and not any(
            row["reason_contains"] in d["rationale"] for d in outcome.decisions
        ):
            problems.append(f"{where}: no rationale mentions {row['reason_contains']!r}")
        previous = outcome.views

    sentinel = result.rig.sentinel
    final = {d.disruption_id: d for d in sentinel.disruptions()}
    messages = [message for _topic, message in result.rig.bus.messages]
    for item in expected["disruptions"]:
        did = item["disruption_id"]
        problems += check_disruption(final.get(did), item)
        if did not in final:
            continue
        closed_as = sentinel.tracker.get(did).closed_as
        if closed_as != ("resolved" if item.get("status") == "resolved" else None):
            problems.append(f"{did} is {closed_as or 'open'}, expected status {item.get('status')}")
        state = sentinel.describe(did)
        for key in ("layer", "linked"):
            if key in item and state[key] != item[key]:
                problems.append(f"{did}.{key} = {state[key]!r}, expected {item[key]!r}")
        topics = [m["topic"] for m in messages if m["disruption"]["disruption_id"] == did]
        if "published_topics" in item and topics != item["published_topics"]:
            problems.append(f"{did} published {topics}, expected {item['published_topics']}")
    for item in expected["ignored"]:
        did = item["disruption_id"]
        incident = sentinel.tracker.get(did) if did in final else None
        if incident is None or incident.closed_as != "ignored":
            problems.append(f"{did} not ignored")
            continue
        if item["reason_contains"] not in incident.closed_reason:
            problems.append(f"{did} ignored for {incident.closed_reason!r}")
        if incident.disruption.location != item["location"]:
            problems.append(f"{did} at {incident.disruption.location}, expected {item['location']}")
        if any(m["disruption"]["disruption_id"] == did for m in messages):
            problems.append(f"{did} was published but should never be")
    for item in expected["rejected"]:
        entries = [
            e
            for e in result.rig.audit.entries
            if e.action == "reject" and e.details.get("event_id") == item["event_id"]
        ]
        if not entries or item["reason_contains"] not in entries[0].rationale:
            problems.append(f"{item['event_id']} has no matching reject audit entry")
    if len(final) != len(expected["disruptions"]) + len(expected["ignored"]):
        problems.append(
            f"{len(final)} disruptions opened, expected "
            f"{len(expected['disruptions']) + len(expected['ignored'])}"
        )
    for message in messages:
        try:
            BusMessage.model_validate(message)
        except ValidationError as exc:
            problems.append(f"published message breaks the contract: {exc}")
    return problems


# --- CLI ----------------------------------------------------------------------------------------


def _clock_label(t_min: float) -> str:
    return f"T+{int(t_min // 60):02d}:{int(t_min % 60):02d}"


class _Printer:
    """Prints the replay as it happens: one block per step, one line per decision."""

    def __call__(self, kind: str, data: dict[str, Any]) -> None:
        if kind == "step":
            label = "tick" if data["tick"] else "event"
            print(f"\n[{_clock_label(data['t_min'])}] {label}: {data['note']}")
        elif kind == "signal":
            if data["status"] == "rejected":
                print(f"  signal   {data.get('event_id', '?')} REJECTED: {data['reason']}")
            else:
                kind_of = f"{data['type']} @ {data['location']}" if data["type"] else data["role"]
                print(
                    f"  signal   {data['event_id']} {data['source']} (r={data['reliability']:.2f}, "
                    f"{data['classified_by']}): {data['title']} -> {kind_of}"
                )
        elif kind == "decision":
            print(f"  {data['action'].upper():<9}{data['rationale']}")
        elif kind == "publish":
            d = data["disruption"]
            print(
                f"  >> {data['topic']} {d['disruption_id']} {d['type']} {d['location']} "
                f"{d['status']} · severity {d['severity']}/5 {d['priority']} · "
                f"{count(len(d['affected_container_ids']))} · ~{d['expected_delay_days']:g} d "
                f"· conf {d['confidence']:.3f}"
            )


def _print_final(result: ReplayResult) -> None:
    print("\nDisruptions at the end of the replay:")
    for d in result.rig.sentinel.disruptions():
        incident = result.rig.sentinel.tracker.get(d.disruption_id)
        state = incident.closed_as or d.status
        print(
            f"  {d.disruption_id}  {d.type:<16} {d.location:<14} {state:<10} severity "
            f"{d.severity}/5 {d.priority:<6} conf {d.confidence:.3f}  "
            f"{count(len(d.affected_container_ids)):>13}  ~{d.expected_delay_days:g} d"
        )
        print(f"            {d.summary}")


def _utf8_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agents.sentinel.replay",
        description="Replay a SIMULATED scenario through the Disruption Sentinel and print "
        "each decision.",
    )
    parser.add_argument(
        "scenario", nargs="+", help=f"scenario name(s), or 'all': {', '.join(SCENARIO_NAMES)}"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=60,
        help="simulated minutes per real minute (60 = 1 min per second, 0 = instant)",
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=3.0,
        help="most real seconds a fast-forward (a gap of 1 h+ simulated) waits (default 3)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare with data/expected/<name>.json; exit 1 on any mismatch",
    )
    parser.add_argument(
        "--llm",
        choices=LLM_MODES,
        help="override SENTINEL_LLM (default off; the expected files assume off)",
    )
    args = parser.parse_args(argv)
    names = list(SCENARIO_NAMES) if args.scenario == ["all"] else args.scenario
    unknown = [n for n in names if not _path(n, SCENARIO_DIR).exists()]
    if unknown:
        parser.error(f"unknown scenario {', '.join(unknown)} (known: {', '.join(SCENARIO_NAMES)})")
    if args.speed < 0 or args.max_gap < 0:
        parser.error("--speed and --max-gap must not be negative")

    _utf8_stdout()
    llm = build_llm_client(args.llm)
    print(f"LLM: {llm.label}" + "".join(f"\n  {note}" for note in llm.notes))
    failed = False
    for name in names:
        scenario = load_scenario(name)
        print(f"=== Disruption Sentinel · SIMULATED FEEDS · {scenario.name}: {scenario.title}")
        print(f"    {scenario.disclaimer}")
        rig = rig_for(scenario, llm=llm.client)
        result = replay(scenario, rig, speed=args.speed, max_gap=args.max_gap, on_event=_Printer())
        _print_final(result)
        if args.check:
            problems = compare_with_expected(result, load_expected(name))
            verdict = "OK" if not problems else f"{len(problems)} MISMATCH(ES)"
            print(f"\nCheck against data/expected/{name}.json: {verdict}")
            for problem in problems:
                print(f"  - {problem}")
            failed = failed or bool(problems)
        print()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
