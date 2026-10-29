"""The shared contract (models.py), the in-memory fakes and the simulated data files."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from agents.sentinel.fakes import (
    DEFAULT_CONTAINERS_PATH,
    InMemoryAudit,
    InMemoryBus,
    JsonContainerRepo,
    ScriptedLLM,
    SimClock,
)
from agents.sentinel.models import BusMessage, ContainerRef, Disruption
from agents.sentinel.replay import SCENARIO_NAMES, load_expected, load_scenario

T0 = datetime(2026, 3, 2, 6, 0, tzinfo=UTC)
GULF_GRAPES = [f"GRP-{i:02d}" for i in range(1, 13)]


def disruption(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "disruption_id": "DSR-0001",
        "type": "PORT_CLOSURE",
        "status": "active",
        "location": "AEJEA",
        "first_seen": T0.isoformat(),
        "last_updated": T0.isoformat(),
        "expected_delay_days": 14,
        "delay_is_estimate": True,
        "cost_impact_pct": 0,
        "market_unreachable": True,
        "severity": 5,
        "priority": "urgent",
        "confidence": 0.997,
        "sources": [
            {"event_id": "GC-04", "source": "port_authority", "received_at": T0.isoformat()}
        ],
        "affected_container_ids": GULF_GRAPES,
        "affected_weight_kg": 216000,
        "summary": "PORT_CLOSURE at AEJEA: 12 containers affected, ~14 days delay, severity 5/5.",
        "rationale": ["port_authority: AEJEA closed (reopen unknown)"],
        "simulated": True,
    }
    return base | overrides


# --- contract ------------------------------------------------------------------------------------


def test_disruption_contract_round_trips_through_json() -> None:
    model = Disruption.model_validate(disruption())
    again = Disruption.model_validate_json(model.model_dump_json())
    assert again == model
    message = BusMessage(topic="disruption.detected", disruption=model, changed_fields=["status"])
    assert BusMessage.model_validate(message.model_dump(mode="json")) == message


def test_disruption_defaults_to_simulated() -> None:
    data = disruption()
    del data["simulated"]
    assert Disruption.model_validate(data).simulated is True


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"severity": 0}, id="severity-below-1"),
        pytest.param({"severity": 6}, id="severity-above-5"),
        pytest.param({"confidence": 1.2}, id="confidence-above-1"),
        pytest.param({"type": "EARTHQUAKE"}, id="unknown-type"),
        pytest.param({"status": "expired"}, id="unknown-status"),
        pytest.param({"priority": "high"}, id="unknown-priority"),
        pytest.param({"simulated": False}, id="not-simulated"),
        pytest.param({"disruption_id": "D-1"}, id="bad-id"),
        pytest.param({"owner": "someone"}, id="extra-field"),
    ],
)
def test_disruption_contract_rejects_bad_values(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Disruption.model_validate(disruption(**overrides))


def test_bus_message_rejects_unknown_topic() -> None:
    with pytest.raises(ValidationError):
        BusMessage.model_validate(
            {"topic": "disruption.maybe", "disruption": disruption(), "changed_fields": []}
        )


def test_container_ref_ignores_extra_teammate_fields() -> None:
    ref = ContainerRef.model_validate(
        {
            "container_id": "X-1",
            "crop": "grapes",
            "hs_code": "0806.10",
            "weight_kg": 1,
            "vessel_id": "mv-a",
            "origin_port": "innsa",
            "destination_port": "aejea",
            "destination_country": "ae",
            "route_legs": ["aejea"],
            "eta": "2026-03-04T08:00:00",
            "status": "AT_SEA",
            "days_left": None,
            "farmer_group": "FPO-7",
        }
    )
    assert (ref.hs_code, ref.vessel_id, ref.destination_port, ref.status) == (
        "080610",
        "MV-A",
        "AEJEA",
        "at_sea",
    )
    assert ref.eta.tzinfo is not None


# --- fakes ---------------------------------------------------------------------------------------


def test_bus_keeps_messages_and_notifies_until_unsubscribed() -> None:
    bus = InMemoryBus()
    seen: list[str] = []
    unsubscribe = bus.subscribe(lambda topic, _message: seen.append(topic))
    bus.publish("disruption.detected", {"n": 1})
    unsubscribe()
    bus.publish("disruption.updated", {"n": 2})
    assert seen == ["disruption.detected"]
    assert bus.by_topic("disruption.updated") == [{"n": 2}]


def test_audit_requires_a_rationale_and_numbers_entries() -> None:
    audit = InMemoryAudit()
    audit.log("disruption_sentinel", "create", {"id": "DSR-0001"}, "news-only candidate")
    with pytest.raises(ValueError, match="rationale"):
        audit.log("disruption_sentinel", "merge", {}, "  ")
    assert [entry.seq for entry in audit.entries] == [1]
    assert audit.actions() == ["create"]


def test_sim_clock_only_moves_forward() -> None:
    clock = SimClock(T0)
    assert clock.advance(timedelta(minutes=8)) == T0 + timedelta(minutes=8)
    with pytest.raises(ValueError, match="back"):
        clock.set(T0)


def test_container_repo_filters_to_a_scenario_book() -> None:
    assert len(JsonContainerRepo().list_active()) == 31
    book = JsonContainerRepo(container_ids=["GRP-01", "POM-01"]).list_active()
    assert [c.container_id for c in book] == ["GRP-01", "POM-01"]
    with pytest.raises(ValueError, match="NOPE-1"):
        JsonContainerRepo(container_ids=["NOPE-1"])


def test_container_repo_skips_delivered(tmp_path: Path) -> None:
    rows = (
        DEFAULT_CONTAINERS_PATH.read_text(encoding="utf-8")
        .replace('"container_id": "GRP-01", ', '"container_id": "GRP-01", "x": 1, ', 1)
        .replace('"at_sea", "days_left": 9.5', '"delivered", "days_left": 9.5', 1)
    )
    path = tmp_path / "containers.json"
    path.write_text(rows, encoding="utf-8")
    ids = [c.container_id for c in JsonContainerRepo(path).list_active()]
    assert "GRP-01" not in ids
    assert len(ids) == 30


def test_scripted_llm_replays_replies_and_raises_queued_errors() -> None:
    llm = ScriptedLLM(news=[{"type": None}, TimeoutError("slow")], summaries=["ok"])
    assert llm.classify_news("h", "b") == {"type": None}
    with pytest.raises(TimeoutError):
        llm.classify_news("h", "b")
    assert llm.summarise({}) == "ok"
    with pytest.raises(RuntimeError, match="no reply"):
        llm.summarise({})


# --- simulated data ------------------------------------------------------------------------------


def test_gulf_seed_is_12_grape_containers_plus_6_elsewhere() -> None:
    scenario = load_scenario("gulf_closure")
    book = JsonContainerRepo(scenario.book_path, container_ids=scenario.container_ids).list_active()
    gulf = [c for c in book if c.destination_port == "AEJEA" or "AEJEA" in c.route_legs]
    others = [c for c in book if c not in gulf]
    assert sorted(c.container_id for c in gulf) == GULF_GRAPES
    assert all(c.crop == "grapes" and c.weight_kg == 18000 for c in gulf)
    assert all((c.origin_port, c.destination_port) == ("INNSA", "AEJEA") for c in gulf)
    assert {c.vessel_id for c in gulf} == {"MV-ARABIAN-01", "MV-ARABIAN-02"}
    assert len(others) == 6


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_scenarios_are_simulated_and_reference_known_containers(name: str) -> None:
    scenario = load_scenario(name)
    assert scenario.simulated is True
    assert "SIMULATED" in scenario.disclaimer
    JsonContainerRepo(scenario.book_path, container_ids=scenario.container_ids)  # unknown ids raise
    for step in scenario.steps:
        if step.event and step.event["source"] == "news":
            assert "(SIM)" in step.event["payload"]["outlet"]


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_expected_files_are_consistent_with_their_scenario(name: str) -> None:
    scenario = load_scenario(name)
    expected = load_expected(name)
    assert expected["scenario"] == name
    assert expected["simulated"] is True
    event_ids = {step.event["event_id"] for step in scenario.steps if step.event}
    tick_times = {step.t_min for step in scenario.steps if step.tick}
    for row in expected["timeline"]:
        assert row.get("event_id") in event_ids or (row.get("tick") and row["t_min"] in tick_times)
    for item in expected["disruptions"]:
        assert set(item["source_event_ids"]) <= event_ids
        assert set(item["affected_container_ids"]) <= set(scenario.container_ids)
        if "affected_weight_kg" in item:
            repo = JsonContainerRepo(scenario.book_path)
            book = {c.container_id: c for c in repo.list_active()}
            weight = sum(book[cid].weight_kg for cid in item["affected_container_ids"])
            assert weight == item["affected_weight_kg"]
