"""Spec section 8 end to end: replay vs expected, contract, audit, rumour, performance, CLI."""

from __future__ import annotations

import copy
import itertools
import json
import threading
from datetime import timedelta
from typing import Any

import pytest

from agents.sentinel.models import AUDIT_ACTOR, BusMessage, Disruption
from agents.sentinel.replay import (
    SCENARIO_NAMES,
    ReplayResult,
    Rig,
    compare_with_expected,
    load_expected,
    load_scenario,
    main,
    make_rig,
    replay,
)
from agents.sentinel.tests.builders import GULF_GRAPES, T0, WHEN, advisory, ais, at, news, port

_ids = itertools.count(1)


@pytest.fixture(scope="module")
def runs() -> dict[str, ReplayResult]:
    return {name: replay(load_scenario(name)) for name in SCENARIO_NAMES}


@pytest.fixture
def rig() -> Rig:
    """A Sentinel over the full container book, on a simulated clock at T0."""
    return make_rig(None, T0)


def feed(rig: Rig, source: str, payload: dict[str, Any], *, minute: float = 0) -> list[Disruption]:
    rig.clock.set(T0 + timedelta(minutes=minute))
    return rig.sentinel.ingest(
        {
            "event_id": f"R-{next(_ids):03d}",
            "source": source,
            "received_at": rig.clock.now().isoformat(),
            "payload": payload,
        }
    )


def topics(rig: Rig) -> list[str]:
    return [topic for topic, _message in rig.bus.messages]


# --- scenarios against data/expected ----------------------------------------------------------


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_replay_matches_expected(runs: dict[str, ReplayResult], name: str) -> None:
    assert compare_with_expected(runs[name], load_expected(name)) == []


def test_checker_reports_a_mismatch(runs: dict[str, ReplayResult]) -> None:
    wrong = copy.deepcopy(load_expected("gulf_closure"))
    wrong["disruptions"][0]["severity"] = 4
    wrong["timeline"][2]["published"] = None
    problems = compare_with_expected(runs["gulf_closure"], wrong)
    assert any("DSR-0001.severity = 5, expected 4" in p for p in problems)
    assert any(p.startswith("t=5: published") for p in problems)


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_every_published_message_fits_the_contract(
    runs: dict[str, ReplayResult], name: str
) -> None:
    messages = runs[name].rig.bus.messages
    assert messages
    for topic, message in messages:
        parsed = BusMessage.model_validate(message)
        assert parsed.topic == topic
        assert parsed.disruption.simulated is True
        assert parsed.disruption.status != "candidate"
        json.dumps(message)
        if topic == "disruption.detected":
            assert parsed.changed_fields == list(Disruption.model_fields)
        else:
            assert "last_updated" not in parsed.changed_fields


def test_gulf_escalation_message_lists_what_changed(runs: dict[str, ReplayResult]) -> None:
    updates = runs["gulf_closure"].rig.bus.by_topic("disruption.updated")
    assert len(updates) == 1
    changed = set(updates[0]["changed_fields"])
    assert {
        "type",
        "location",
        "status",
        "severity",
        "priority",
        "expected_delay_days",
        "market_unreachable",
    } <= changed
    assert updates[0]["disruption"]["affected_container_ids"] == GULF_GRAPES


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_every_decision_is_audited_with_a_rationale(
    runs: dict[str, ReplayResult], name: str
) -> None:
    run = runs[name]
    decisions = [(d["action"], d["rationale"]) for step in run.steps for d in step.decisions]
    entries = run.rig.audit.entries
    assert [(e.action, e.rationale) for e in entries] == decisions
    for entry in entries:
        assert entry.actor == AUDIT_ACTOR
        assert entry.rationale.strip()
        assert entry.details["simulated"] is True
        json.dumps(entry.details)


def test_audit_covers_each_kind_of_decision(runs: dict[str, ReplayResult]) -> None:
    gulf = set(runs["gulf_closure"].rig.audit.actions())
    assert {"create", "merge", "escalate", "ignore"} <= gulf
    assert "reject" in runs["us_tariff_spike"].rig.audit.actions()


def test_singapore_rumour_expires_and_is_never_published(runs: dict[str, ReplayResult]) -> None:
    run = runs["gulf_closure"]
    published = [m["disruption"] for _topic, m in run.rig.bus.messages]
    assert all(d["location"] != "SGSIN" for d in published)
    state = run.rig.sentinel.describe("DSR-0003")
    assert (state["ignored"], state["published"], state["linked"]) == (True, False, [])
    assert state["closed_reason"].startswith("ignored – uncorroborated")
    last = run.rig.audit.entries[-1]
    assert (last.action, last.details["disruption_id"]) == ("ignore", "DSR-0003")
    assert "ignored – uncorroborated" in last.rationale


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_ingest_and_publish_take_under_200ms_per_event(
    runs: dict[str, ReplayResult], name: str
) -> None:
    slowest = max(step.seconds for step in runs[name].steps)
    assert slowest < 0.2, f"slowest step took {slowest * 1000:.0f} ms"


def test_replay_pacing_follows_speed_and_caps_fast_forwards() -> None:
    gulf = load_scenario("gulf_closure")  # gaps (simulated min): 2, 3, 3, 1, 3, 18, 1440
    slept: list[float] = []
    replay(gulf, speed=60, max_gap=3, sleep=slept.append)
    assert slept == [2, 3, 3, 1, 3, 18, 3]  # only the 24 h fast-forward is capped
    slept.clear()
    replay(gulf, speed=1, max_gap=3, sleep=slept.append)
    assert slept == [120, 180, 180, 60, 180, 1080, 3]
    slept.clear()
    replay(gulf, speed=0, sleep=slept.append)
    assert slept == []


def test_replay_can_be_stopped() -> None:
    stop = threading.Event()
    seen: list[str] = []

    def stop_after_two(kind: str, _data: dict[str, Any]) -> None:
        if kind == "step":
            seen.append(kind)
            if len(seen) == 2:
                stop.set()

    result = replay(load_scenario("gulf_closure"), speed=6000, stop=stop, on_event=stop_after_two)
    assert result.stopped
    assert len(result.steps) == 2


# --- the Sentinel's own interface ---------------------------------------------------------------


def test_ingest_returns_what_it_created_or_updated(rig: Rig) -> None:
    created = feed(rig, "news", news("Jebel Ali port closed"))
    assert [(d.disruption_id, d.status) for d in created] == [("DSR-0001", "candidate")]
    updated = feed(rig, "port_authority", port("AEJEA", "closed"), minute=1)
    assert [(d.disruption_id, d.status) for d in updated] == [("DSR-0001", "active")]
    assert feed(rig, "tariff_feed", {"importing_country": "US"}, minute=2) == []  # malformed
    assert rig.audit.entries[-1].action == "reject"


def test_active_lists_only_confirmed_disruptions(rig: Rig) -> None:
    feed(rig, "news", news("Singapore port shut"))
    feed(rig, "port_authority", port("AEJEA", "closed"), minute=1)
    assert [d.location for d in rig.sentinel.active()] == ["AEJEA"]
    assert [d.location for d in rig.sentinel.disruptions()] == ["SGSIN", "AEJEA"]


def test_events_arrive_in_order_for_the_stream(rig: Rig) -> None:
    kinds: list[str] = []
    rig.sentinel.subscribe(lambda kind, _data: kinds.append(kind))
    feed(rig, "port_authority", port("AEJEA", "closed"))
    assert kinds == ["signal", "decision", "publish", "disruption"]


def test_signal_info_shows_why_a_signal_was_ignored(rig: Rig) -> None:
    signals: list[dict[str, Any]] = []
    rig.sentinel.subscribe(lambda kind, data: signals.append(data) if kind == "signal" else None)
    feed(rig, "news", news("Grape prices firm at Nashik mandi"))
    feed(rig, "tariff_feed", {"importing_country": "US"}, minute=1)
    assert [s["status"] for s in signals] == ["ignored", "rejected"]
    assert "no disruption terms" in signals[0]["status_reason"]
    assert "new_rate_pct" in signals[1]["reason"]


def test_supporting_signals_are_not_republished(rig: Rig) -> None:
    feed(rig, "port_authority", port("AEJEA", "closed"))
    feed(rig, "ais", ais("MV-ARABIAN-01", WHEN, at(days=3)), minute=3)
    feed(rig, "news", news("Jebel Ali port closed"), minute=4)
    assert topics(rig) == ["disruption.detected"]


def test_delay_must_move_a_full_day_to_republish(rig: Rig) -> None:
    feed(rig, "port_authority", port("AEJEA", "closed", reopen=at(days=5)))
    feed(rig, "port_authority", port("AEJEA", "closed", reopen=at(days=5.5)), minute=1)
    assert topics(rig) == ["disruption.detected"]
    feed(rig, "port_authority", port("AEJEA", "closed", reopen=at(days=7)), minute=2)
    assert topics(rig) == ["disruption.detected", "disruption.updated"]
    assert "expected_delay_days" in rig.bus.messages[-1][1]["changed_fields"]


def test_reopen_publishes_updated_then_resolved(rig: Rig) -> None:
    feed(rig, "port_authority", port("AEJEA", "closed"))
    feed(rig, "port_authority", port("AEJEA", "open"), minute=60)
    rig.clock.set(T0 + timedelta(hours=13))
    resolved = rig.sentinel.tick()
    assert [d.status for d in resolved] == ["resolved"]
    assert topics(rig) == ["disruption.detected", "disruption.updated", "disruption.resolved"]
    assert rig.sentinel.active() == []
    assert rig.audit.actions() == ["create", "resolve", "resolve"]


def test_lane_delay_then_closure_publishes_detected_then_updated(rig: Rig) -> None:
    feed(rig, "carrier_advisory", advisory("delay", days=6, vessels=["MV-ARABIAN-01"]))
    feed(rig, "port_authority", port("AEJEA", "closed"), minute=3)
    assert topics(rig) == ["disruption.detected", "disruption.updated"]
    assert rig.bus.messages[-1][1]["disruption"]["type"] == "PORT_CLOSURE"


# --- CLI ----------------------------------------------------------------------------------------


def test_cli_replays_and_checks_the_gulf(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["gulf_closure", "--speed", "0", "--check"]) == 0
    out = capsys.readouterr().out
    assert "SIMULATED FEEDS" in out
    assert ">> disruption.detected DSR-0001 VESSEL_DELAY" in out
    assert ">> disruption.updated DSR-0001 PORT_CLOSURE AEJEA active" in out
    assert "IGNORE   DSR-0003 ignored – uncorroborated" in out
    assert "Check against data/expected/gulf_closure.json: OK" in out


def test_cli_checks_all_scenarios(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["all", "--speed", "0", "--check"]) == 0
    assert capsys.readouterr().out.count(": OK") == len(SCENARIO_NAMES)


@pytest.mark.parametrize("argv", [["atlantis"], ["gulf_closure", "--speed", "-1"]])
def test_cli_rejects_bad_arguments(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(argv)
    assert caught.value.code == 2
