"""Spec section 3 steps 3 and 5: dedupe/merge and the lifecycle, plus the scenario timelines."""

from __future__ import annotations

import pytest

from agents.sentinel.models import Disruption
from agents.sentinel.replay import SCENARIO_NAMES, load_expected
from agents.sentinel.settings import SentinelConfig
from agents.sentinel.tests.builders import (
    GULF_GRAPES,
    WHEN,
    Harness,
    actions,
    advisory,
    ais,
    at,
    check_view,
    news,
    port,
    run_scenario,
    tariff,
    weather,
)

ARABIAN = ["MV-ARABIAN-01", "MV-ARABIAN-02"]
HOUR = 60
DAY = 24 * HOUR


# --- the scenarios ------------------------------------------------------------------------------


def test_gulf_scenario_produces_exactly_one_port_closure_for_aejea(config: SentinelConfig) -> None:
    run = run_scenario(config, "gulf_closure")
    views = [i.disruption for i in run.tracker.incidents]
    closures = [v for v in views if (v.type, v.location) == ("PORT_CLOSURE", "AEJEA")]
    assert len(closures) == 1
    closure = closures[0]
    assert [s.event_id for s in closure.sources] == ["GC-01", "GC-03", "GC-04", "GC-05", "GC-06"]
    assert closure.affected_container_ids == GULF_GRAPES
    assert (closure.status, closure.severity, closure.priority) == ("active", 5, "urgent")
    assert len(views) == 3  # the closure, INNSA congestion, and the ignored Singapore rumour


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_decisions_follow_the_labelled_timeline(config: SentinelConfig, name: str) -> None:
    run = run_scenario(config, name)
    rows = load_expected(name)["timeline"]
    assert len(run.steps) == len(rows)
    previous: dict[str, Disruption] = {}
    for result, row in zip(run.steps, rows, strict=True):
        label = f"{name} t={row['t_min']}"
        if row.get("rejected"):
            assert result.rejected is not None, label
            assert row["reason_contains"] in result.rejected.reason
            continue
        assert actions(result.decisions) == row["audit"], label
        touched = {d.disruption_id for d in result.decisions}
        assert touched == set(row.get("disruption_ids") or [row["disruption_id"]]), label
        if "disruption_ids" in row:  # a tick that moved several disruptions
            previous = result.views
            continue
        view = result.views[row["disruption_id"]]
        check_view(view, row)
        if row.get("confidence_increases"):
            assert view.confidence > previous[row["disruption_id"]].confidence, label
        if "reason_contains" in row:
            assert any(row["reason_contains"] in d.rationale for d in result.decisions), label
        previous = result.views


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_final_state_matches_the_labelled_disruptions(config: SentinelConfig, name: str) -> None:
    run = run_scenario(config, name)
    expected = load_expected(name)
    incidents = {i.disruption_id: i for i in run.tracker.incidents}
    for item in expected["disruptions"]:
        incident = incidents[item["disruption_id"]]
        assert incident.closed_as == ("resolved" if item["status"] == "resolved" else None)
        check_view(incident.disruption, item)
    for item in expected["ignored"]:
        incident = incidents[item["disruption_id"]]
        assert incident.closed_as == "ignored"
        assert incident.disruption.location == item["location"]
        assert item["reason_contains"] in incident.closed_reason
    assert len(incidents) == len(expected["disruptions"]) + len(expected["ignored"])


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_every_decision_has_a_rationale_and_every_view_fits_the_contract(
    config: SentinelConfig, name: str
) -> None:
    run = run_scenario(config, name)
    for result in run.steps:
        for decision in result.decisions:
            assert decision.rationale.strip()
            assert decision.action in {"create", "merge", "escalate", "ignore", "resolve"}
        for view in result.views.values():
            assert Disruption.model_validate_json(view.model_dump_json()) == view
            assert view.simulated is True
            assert view.rationale


# --- dedupe and merge rules -----------------------------------------------------------------------


def test_news_alone_never_confirms(harness: Harness) -> None:
    for minute, outlet in enumerate(["A", "B", "C"]):
        harness.send(
            "news", news("Jebel Ali port closed", outlet=f"Wire {outlet} (SIM)"), minute=minute
        )
    assert len(harness.tracker.incidents) == 1
    view = harness.view()
    assert view.confidence == pytest.approx(1 - 0.64**3, abs=1e-4)  # above 0.6...
    assert view.status == "candidate"  # ...and still not confirmed
    assert (view.severity, view.affected_container_ids) == (1, [])


def test_same_outlet_repeating_merges_without_raising_confidence(harness: Harness) -> None:
    harness.send("news", news("Jebel Ali port closed"), minute=0)
    decisions = harness.send("news", news("Jebel Ali port closed"), minute=1)
    assert actions(decisions) == ["merge"]
    assert harness.view().confidence == pytest.approx(0.36)


def test_duplicate_event_id_is_ignored(harness: Harness) -> None:
    harness.send("port_authority", port("AEJEA", "closed"), event_id="E-1")
    decisions = harness.send("port_authority", port("AEJEA", "closed"), event_id="E-1")
    assert actions(decisions) == ["ignore"]
    assert "duplicate" in decisions[0].rationale
    assert len(harness.view().sources) == 1


def test_noise_is_ignored_with_the_reason(harness: Harness) -> None:
    decisions = harness.send("tariff_feed", tariff("US", "0306", 10, 12))
    assert actions(decisions) == ["ignore"]
    assert "threshold" in decisions[0].rationale
    assert harness.tracker.incidents == []


def test_lane_delay_escalates_into_the_closure_at_its_destination(harness: Harness) -> None:
    harness.send("carrier_advisory", advisory("delay", days=6, vessels=ARABIAN), minute=0)
    assert (harness.view().type, harness.view().severity) == ("VESSEL_DELAY", 3)
    decisions = harness.send("port_authority", port("AEJEA", "closed"), minute=5)
    assert actions(decisions) == ["merge", "escalate"]
    assert "type VESSEL_DELAY -> PORT_CLOSURE" in decisions[1].rationale
    assert "severity 3 -> 5" in decisions[1].rationale
    assert len(harness.tracker.incidents) == 1
    assert (harness.view().type, harness.view().location) == ("PORT_CLOSURE", "AEJEA")


def test_congestion_escalates_to_closure_at_the_same_port(harness: Harness) -> None:
    harness.send("port_authority", port("AEJEA", "congested"), minute=0)
    decisions = harness.send("port_authority", port("AEJEA", "closed"), minute=5)
    assert actions(decisions) == ["merge", "escalate"]
    assert harness.view().type == "PORT_CLOSURE"


def test_closures_at_different_ports_stay_separate(harness: Harness) -> None:
    harness.send("port_authority", port("AEJEA", "closed"), minute=0)
    harness.send("port_authority", port("QAHMD", "closed"), minute=1)
    assert [i.disruption.location for i in harness.tracker.incidents] == ["AEJEA", "QAHMD"]
    assert harness.view("DSR-0002").affected_container_ids == ["BAN-01"]


def test_ais_on_an_affected_vessel_supports_rather_than_duplicates(harness: Harness) -> None:
    harness.send("port_authority", port("AEJEA", "closed"), minute=0)
    supporting = harness.send("ais", ais("MV-ARABIAN-01", WHEN, at(days=3)), minute=3)
    assert actions(supporting) == ["merge"]
    assert "supporting signal" in supporting[0].rationale
    unrelated = harness.send("ais", ais("MV-PACIFIC-03", WHEN, at(days=3)), minute=4)
    assert actions(unrelated) == ["create"]
    assert harness.view("DSR-0002").affected_container_ids == ["SHR-01", "SHR-02", "SHR-03"]
    assert harness.view().type == "PORT_CLOSURE"


def test_reports_more_than_72h_apart_open_a_new_disruption(harness: Harness) -> None:
    harness.send("port_authority", port("INNSA", "congested"), minute=0)
    assert actions(
        harness.send("port_authority", port("INNSA", "congested"), minute=71 * HOUR)
    ) == ["merge"]
    later = harness.send("port_authority", port("INNSA", "congested"), minute=(71 + 73) * HOUR)
    assert actions(later) == ["create"]
    assert len(harness.tracker.incidents) == 2


# --- lifecycle ------------------------------------------------------------------------------------


def test_candidate_expires_after_24h_as_ignored_uncorroborated(harness: Harness) -> None:
    harness.send("news", news("Singapore port shut, sources claim"), minute=0)
    assert harness.tick(minute=DAY - 1) == []
    decisions = harness.tick(minute=DAY)
    assert actions(decisions) == ["ignore"]
    assert "ignored – uncorroborated" in decisions[0].rationale
    incident = harness.tracker.get("DSR-0001")
    assert incident.closed_as == "ignored"
    assert harness.tick(minute=2 * DAY) == []


def test_a_corroborated_candidate_does_not_expire(harness: Harness) -> None:
    harness.send("news", news("Jebel Ali port closed"), minute=0)
    harness.send("port_authority", port("AEJEA", "closed"), minute=1)
    assert harness.tick(minute=2 * DAY) == []
    assert harness.view().status == "active"


def test_authoritative_source_alone_confirms_and_activates_on_its_start_time(
    harness: Harness,
) -> None:
    decisions = harness.send("tariff_feed", tariff("US", "0306", 10, 50, effective=at(days=2)))
    assert actions(decisions) == ["create"]
    assert harness.view().status == "confirmed"
    assert harness.tick(minute=DAY) == []
    started = harness.tick(minute=2 * DAY)
    assert actions(started) == ["escalate"]
    assert "status confirmed -> active" in started[0].rationale
    assert harness.view().status == "active"


def test_reopen_moves_to_resolving_then_resolved_after_12h(harness: Harness) -> None:
    harness.send("port_authority", port("AEJEA", "closed"), minute=0)
    reopened = harness.send("port_authority", port("AEJEA", "open"), minute=HOUR)
    assert actions(reopened) == ["resolve"]
    assert harness.view().status == "resolving"
    assert harness.tick(minute=HOUR + 11 * HOUR) == []
    done = harness.tick(minute=HOUR + 12 * HOUR)
    assert actions(done) == ["resolve"]
    view = harness.view()
    assert view.status == "resolved"
    assert (
        view.affected_container_ids == GULF_GRAPES
    )  # the orchestrator learns which boxes are free
    assert [s.event_id for s in view.sources] == ["H-001", "H-002"]


def test_new_closure_while_resolving_reopens_the_disruption(harness: Harness) -> None:
    harness.send("port_authority", port("AEJEA", "closed"), minute=0)
    harness.send("port_authority", port("AEJEA", "open"), minute=HOUR)
    decisions = harness.send("port_authority", port("AEJEA", "closed"), minute=2 * HOUR)
    assert actions(decisions) == ["merge", "escalate"]
    assert "status resolving -> active" in decisions[1].rationale
    assert harness.tick(minute=20 * HOUR) == []


def test_rumour_refuted_by_the_port_authority_is_ignored(harness: Harness) -> None:
    harness.send("news", news("Singapore port shut"), minute=0)
    decisions = harness.send("port_authority", port("SGSIN", "open"), minute=10)
    assert actions(decisions) == ["ignore"]
    assert "refuted" in decisions[0].rationale
    assert harness.tracker.get("DSR-0001").closed_as == "ignored"


def test_reopen_with_nothing_open_is_ignored(harness: Harness) -> None:
    decisions = harness.send("port_authority", port("AEJEA", "open"))
    assert actions(decisions) == ["ignore"]
    assert "nothing open to resolve" in decisions[0].rationale


def test_tariff_reversal_resolves_the_tariff(harness: Harness) -> None:
    harness.send("tariff_feed", tariff("US", "0306", 10, 50), minute=0)
    assert actions(harness.send("tariff_feed", tariff("US", "0306", 50, 10), minute=5)) == [
        "resolve"
    ]
    assert harness.view().status == "resolving"


def test_weather_alert_resolves_when_its_window_ends(harness: Harness) -> None:
    harness.send("weather", weather("orange", port_code="INMUN", valid_to=at(hours=6)), minute=0)
    assert harness.view().status == "active"
    assert actions(harness.tick(minute=7 * HOUR)) == ["resolve"]
    assert harness.view().status == "resolving"
    assert actions(harness.tick(minute=19 * HOUR)) == ["resolve"]
    assert harness.view().status == "resolved"
