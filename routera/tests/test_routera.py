"""ROUTERA engine: shelf life, saturation, compliance, routes, planning, approval and the chain."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from datetime import timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agents.sentinel.fakes import ALL_CONTAINER_PATHS, JsonContainerRepo
from agents.sentinel.models import ContainerRef
from agents.sentinel.replay import load_scenario, replay, rig_for
from routera.app import RouteraService, create_routera_app
from routera.audit import HashChainAudit
from routera.compliance import ComplianceRules
from routera.market import load_markets, realised_price
from routera.orchestrator import Orchestrator
from routera.reroute import Route, load_routes, unavailable
from routera.shelf_life import CROPS, ShelfLifeTracker, rate

BOOK = {c.container_id: c for c in JsonContainerRepo(ALL_CONTAINER_PATHS).list_active()}
ROUTES = {r.id: r for r in load_routes()}


def box(cid: str) -> ContainerRef:
    return BOOK[cid]


def run(name: str, steps: int | None = None) -> Orchestrator:
    """Replay a scenario (optionally only its first steps) with an orchestrator attached."""
    scenario = load_scenario(name)
    if steps is not None:
        scenario = scenario.model_copy(update={"steps": scenario.steps[:steps]})
    rig = rig_for(scenario)
    orchestrator = Orchestrator(rig.sentinel, HashChainAudit(clock=rig.clock), lambda k, d: None)
    rig.bus.subscribe(orchestrator.on_bus)
    replay(scenario, rig, speed=0)
    return orchestrator


def pending(o: Orchestrator) -> list[dict[str, Any]]:
    return [p for p in o.plans.values() if p["status"] == "pending"]


def moves(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [r for r in plan["rows"] if r["choice"] and r["choice"]["route_id"] != "STAY"]


# --- the algorithms -------------------------------------------------------------------------------


def test_q10_rate_doubles_and_triples_with_temperature() -> None:
    grapes = CROPS["grapes"]
    assert rate(grapes, grapes.ref_temp_c) == pytest.approx(1.0)
    assert rate(grapes, grapes.ref_temp_c + 10) == pytest.approx(grapes.q10)


def test_shelf_life_estimate_project_and_excursion() -> None:
    c = box("NSK-01")
    tracker = ShelfLifeTracker(start=load_scenario("nashik_corridor_strike").start)
    later = tracker.start + timedelta(days=2)
    assert tracker.estimate(c, later).days_left == pytest.approx(c.days_left - 2)
    warm = tracker.project(c, 10.0, [(1.0, CROPS["grapes"].ref_temp_c + 10)])
    assert warm == pytest.approx(10.0 - CROPS["grapes"].q10)
    lost = tracker.record_excursion(c, above_c=10, minutes=180)
    assert lost > 0
    assert tracker.estimate(c, later).days_left == pytest.approx(c.days_left - 2 - lost, abs=1e-3)


def test_saturation_lowers_price_down_to_the_floor_and_crash_cuts_it() -> None:
    m = load_markets()["SGSIN"]
    first, sat_first = realised_price(m, added_t=0, load_t=18)
    later, sat_later = realised_price(m, added_t=100, load_t=18)
    flooded, sat_flooded = realised_price(m, added_t=10_000, load_t=18)
    assert m.price_inr_per_kg > first > later > flooded
    assert 1 > sat_first > sat_later > sat_flooded == m.floor_ratio
    crashed, _ = realised_price(m, added_t=0, load_t=18, crash_pct=40)
    assert crashed == pytest.approx(first * 0.6)


def test_only_verified_law_blocks_unverified_rules_go_to_a_human_and_paperwork_is_drafted() -> None:
    rules = ComplianceRules()
    eu = rules.check(box("GRP-09"), "NL")  # EU MRL, verified: a hard block
    assert eu.status == "block" and eu.rules == ("EU-MRL-001",)
    sg = rules.check(box("GRP-03"), "SG")  # Singapore MRL is still TODO_VERIFY: a human decides
    assert sg.status == "review" and "SG-MRL-001" in sg.note
    permit = rules.check(box("GRP-01"), "MY")  # unverified permit rule: apply for it, don't block
    assert permit.status == "docs" and permit.extra_days >= 10 and "MY-PERMIT-001" in permit.rules
    docs = rules.check(box("NSK-01"), "SG")
    assert docs.status == "docs" and docs.draft and "PHYTOSANITARY" in docs.draft
    assert rules.check(box("NSK-01"), "IN").status == "ok"


def test_route_unavailable_for_blocked_place_struck_carrier_or_full_capacity() -> None:
    road: Route = ROUTES["N-JNPT-LK"]
    assert unavailable(road, {}, {}, {}) is None
    assert "ghat" in (unavailable(road, {"KASARA_GHAT": "ghat closed"}, {}, {}) or "")
    assert unavailable(road, {}, {road.carrier: "strike"}, {}) == "strike"
    assert "capacity" in (unavailable(road, {}, {}, {road.id: road.capacity}) or "")


# --- plans from the scenarios ---------------------------------------------------------------------


def test_nashik_mid_strike_plan_avoids_struck_trucks_the_ghat_and_the_full_cold_store() -> None:
    o = run("nashik_corridor_strike", steps=9)  # strike live, ghat shut, cold store full
    ctx = o.context()
    assert "KASARA_GHAT" in ctx.blocked and "CS-NSK" in ctx.blocked and ctx.struck
    plans = pending(o)
    assert plans and sum(p["saved_inr"] for p in plans) > 0
    rows = [row for plan in plans for row in moves(plan)]
    assert rows
    for row in rows:
        route = ROUTES[row["choice"]["route_id"]]
        assert unavailable(route, ctx.blocked, ctx.struck, {}) is None, row["container_id"]
        assert row["value_saved_inr"] >= 0
    chosen = {r["choice"]["route_id"] for r in rows}
    assert not chosen & {"N-JNPT-LK", "N-JNPT-SG", "N-HOLD-LK", "N-VASHI"}


def test_gulf_closure_plan_skips_jebel_ali_blocked_markets_and_failed_lots() -> None:
    o = run("gulf_closure")
    plans = pending(o)
    assert plans
    for plan in plans:
        for row in plan["rows"]:
            assert row["value_saved_inr"] >= 0
            choice = row["choice"]
            if choice is None or choice["route_id"] == "STAY":
                continue  # keeping a booking (e.g. frozen cargo waiting out the closure)
            assert choice["destination"] != "AEJEA"
            assert choice["compliance"]["status"] != "block"
            if row["container_id"] == "GRP-03":
                assert choice["destination"] != "SGSIN"


def test_each_disruption_gets_its_own_plan_with_only_its_own_containers() -> None:
    o = run("gulf_ripple")
    plans = pending(o)
    assert len(plans) > 1
    hits = {i.disruption_id: set(i.disruption.affected_container_ids) for i in o._incidents()}
    for plan in plans:
        [disruption_id] = plan["disruption_ids"]
        assert set(plan["container_ids"]) <= hits[disruption_id]
        assert "linked" not in plan["title"] and "_" not in plan["title"]


def test_one_box_reads_as_one_box_everywhere() -> None:
    o = run("nashik_corridor_strike")  # includes a reefer failure, which hits a single truck
    reefer = next(p for p in pending(o) if p["title"].startswith("Reefer failure"))
    assert reefer["headline"].startswith("1 container at risk.")
    o.approve(reefer["plan_id"], "Asha K.")
    said = json.dumps({"state": o.snapshot(), "disruptions": [i.disruption.summary for i in o._incidents()]},
                      default=str, ensure_ascii=False)
    assert not re.search(r"(?<![\d,.])1 (containers|moves|documents)\b", said)


def test_a_box_hit_by_two_disruptions_moves_once_and_counts_once() -> None:
    o = run("gulf_ripple")
    seen: dict[str, list[dict[str, Any]]] = {}
    for plan in pending(o):
        for row in plan["rows"]:
            seen.setdefault(row["container_id"], []).append(plan)
    shared = {cid: plans for cid, plans in seen.items() if len(plans) > 1}
    assert shared  # this scenario has boxes hit twice
    naive = sum(p["saved_inr"] for p in pending(o))
    assert o.snapshot()["kpis"]["on_table_inr"] < naive  # not double-counted
    plan = next(p for p in pending(o) if moves(p) and {r["container_id"] for r in moves(p)} & set(shared))
    moved = {r["container_id"] for r in moves(plan)}
    o.approve(plan["plan_id"], "Asha K.")
    for other in pending(o):
        assert not moved & set(other["container_ids"])  # gone from every other plan
    assert len(o.bookings) == len(moved)
    assert o.context().used  # and later plans see the route slots it took


def test_a_plan_whose_boxes_were_all_moved_elsewhere_closes_with_the_true_reason() -> None:
    o = run("gulf_ripple")
    for plan in sorted(pending(o), key=lambda p: -len(p["rows"])):
        if plan["status"] == "pending" and moves(plan):
            o.approve(plan["plan_id"], "Asha K.")
    reasons = {p["closed_reason"] for p in o.plans.values() if p["status"] == "closed"}
    assert "the disruption cleared before anyone had to act" not in reasons


# --- the human decision and the chain -------------------------------------------------------------


@pytest.fixture(scope="module")
def gulf() -> Orchestrator:
    return run("gulf_closure")


def test_compliance_block_cannot_be_overridden(gulf: Orchestrator) -> None:
    plan = max(pending(gulf), key=lambda p: p["saved_inr"])
    row, option = next(
        (r, o) for r in plan["rows"] for o in r["options"] if o["compliance"]["status"] == "block"
    )
    with pytest.raises(PermissionError, match="cannot be overridden"):
        gulf.approve(plan["plan_id"], "Asha K.", {row["container_id"]: option["route_id"]})
    assert plan["status"] == "pending"


def test_approve_books_moves_drafts_docs_and_logs_a_valid_chain(gulf: Orchestrator) -> None:
    plan = max(pending(gulf), key=lambda p: p["saved_inr"])
    n = len(moves(plan))
    gulf.approve(plan["plan_id"], "Asha K.")
    assert plan["status"] == "approved"
    assert len([b for b in gulf.bookings if b["plan_id"] == plan["plan_id"]]) == n
    actions = [e["action"] for e in gulf.chain.entries]
    assert {"plan.approved", "booking.created", "payout.logged"} <= set(actions)
    assert gulf.chain.verify()["ok"]
    with pytest.raises(ValueError, match="not pending"):
        gulf.approve(plan["plan_id"], "Asha K.")
    [drafted] = [e for e in gulf.chain.entries if e["action"] == "documents.drafted"]
    cid = moves(plan)[0]["container_id"]
    pack = gulf.paperwork(plan["plan_id"], cid)
    assert drafted["details"]["packs"][cid] == pack["sha256"] == pack["on_chain"]
    assert len(drafted["details"]["packs"]) == n and pack["documents"]


def test_tampering_with_any_entry_breaks_the_chain() -> None:
    chain = HashChainAudit()
    for i in range(3):
        chain.log("approval_audit", "test", {"i": i}, f"entry {i}")
    assert chain.verify() == {**chain.verify(), "ok": True, "length": 3}
    chain.entries[1]["details"]["i"] = 99
    assert chain.verify()["ok"] is False and chain.verify()["broken_at"] == 2
    with pytest.raises(ValueError, match="no rationale"):
        chain.log("x", "y", {}, "  ")


def test_why_explains_choice_baseline_and_alternatives(gulf: Orchestrator) -> None:
    plan = next(iter(gulf.plans.values()))
    cid = plan["rows"][0]["container_id"]
    text = gulf.why(plan["plan_id"], cid)["text"]
    assert cid in text and "Without ROUTERA" in text


# --- the app ----------------------------------------------------------------------------------------


@pytest.fixture
def client() -> Iterator[TestClient]:
    service = RouteraService()
    with TestClient(create_routera_app(service)) as c:
        c.service = service  # type: ignore[attr-defined]
        yield c


def test_api_end_to_end(client: TestClient) -> None:
    assert "ROUTERA" in client.get("/").text
    assert client.get("/sentinel").status_code == 200
    client.service.start_replay("gulf_closure", 0)  # type: ignore[attr-defined]
    client.service.wait_for_replay(30)  # type: ignore[attr-defined]
    state = client.get("/api/state").json()
    assert state["simulated"] is True and state["routera"]["kpis"]["pending_plans"] >= 1
    plan = max(
        (p for p in state["routera"]["plans"] if p["status"] == "pending"),
        key=lambda p: p["saved_inr"],
    )
    pid, cid = plan["plan_id"], plan["rows"][0]["container_id"]
    assert client.get(f"/api/plans/{pid}/why", params={"container": cid}).status_code == 200
    assert client.get(f"/api/plans/{pid}/why", params={"container": "NOPE"}).status_code == 404
    mover = next(r for r in plan["rows"] if r["choice"] and r["choice"]["route_id"] != "STAY")
    pack = client.get(f"/api/plans/{pid}/paperwork", params={"container": mover["container_id"]})
    assert pack.status_code == 200 and pack.json()["documents"]
    assert client.get(f"/api/plans/{pid}/paperwork", params={"container": "NOPE"}).status_code == 404
    zipped = client.get(f"/api/plans/{pid}/filing-pack.zip", params={"container": mover["container_id"]})
    assert zipped.status_code == 200 and zipped.content.startswith(b"PK")
    pdf = client.get(f"/api/plans/{pid}/paperwork/commercial_invoice.pdf", params={"container": mover["container_id"]})
    assert pdf.status_code in (200, 503) and (pdf.status_code == 503 or pdf.content.startswith(b"%PDF"))
    rules = client.get("/api/rules").json()
    assert rules["to_verify"] > 0 and rules["warnings"]
    assert "MY-PERMIT-001" in client.get("/api/rules/to-verify").text
    ok = client.post(f"/api/plans/{pid}/approve", json={"approver": "Asha K."})
    assert ok.status_code == 200 and ok.json()["status"] == "approved"
    assert client.post(f"/api/plans/{pid}/approve", json={}).status_code == 409
    assert client.post("/api/plans/PLN-9999/reject", json={}).status_code == 404
    audit = client.get("/api/audit").json()
    assert audit["ok"] and any(e["action"] == "payout.logged" for e in audit["entries"])
    assert client.get("/api/state").json()["routera"]["kpis"]["protected_inr"] > 0
