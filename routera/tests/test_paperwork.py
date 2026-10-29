"""Paperwork model (ported from the team's governance module): change sets, drafts and checks."""

from __future__ import annotations

from datetime import UTC, datetime

from agents.sentinel.fakes import JsonContainerRepo
from routera.live_feeds import LIVE_BOOKS
from routera.market import load_markets
from routera.paperwork import (
    DECLARATION,
    LAB,
    NOTICE,
    PHYTO,
    PROCESSOR,
    REINSPECT,
    change_set,
    draft_pack,
    validate,
)
from routera.reroute import load_routes

BOOK = {c.container_id: c for c in JsonContainerRepo(LIVE_BOOKS).list_active()}
MARKETS = load_markets()
ROUTES = {r.id: r for r in load_routes()}
AT = datetime(2026, 3, 3, 6, tzinfo=UTC)
LOT = {"phyto": ["AE"], "treated": []}
EXPORTER = {"iec": "SIM-IEC-0301", "packhouse_code": "PH (SIM)", "orchard_code": "FARM (SIM)"}


def test_a_mid_voyage_diversion_redoes_destination_documents_and_keeps_the_lab_report() -> None:
    change = change_set(BOOK["GRP-01"], {"action": "switch_destination"}, MARKETS["LKCMB"], ["AE"])
    assert PHYTO in change["regenerate"] and LAB in change["keep"]
    assert set(change["new"]) == {NOTICE, REINSPECT}  # at sea: the new port inspects again
    assert change["obsolete"] == [] and change["hours"] == 24.0  # the phyto is the slowest


def test_a_phyto_already_issued_for_the_new_country_is_kept() -> None:
    change = change_set(BOOK["MNG-01"], {"action": "reroute_vessel"}, MARKETS["NLRTM"], ["NL"])
    assert PHYTO in change["keep"] and PHYTO not in change["regenerate"]


def test_selling_an_export_box_to_processors_makes_export_documents_obsolete() -> None:
    change = change_set(BOOK["NSK-01"], {"action": "processor_salvage"}, MARKETS["PROC-NSK"], ["LK"])
    assert {DECLARATION, PHYTO, LAB} <= set(change["obsolete"])
    assert change["new"] == [PROCESSOR] and change["hours"] == 2.0


def test_a_draft_pack_never_invents_the_new_buyer_and_its_documents_agree() -> None:
    option = {"action": "switch_destination", "price_inr_per_kg": 115.3}
    pack = draft_pack(BOOK["GRP-01"], option, MARKETS["LKCMB"], ROUTES["G-LK"], LOT, EXPORTER, "CODEX", None,
                      "PLN-0001", "Port closure · Jebel Ali", AT)
    assert pack["to_fill"] == ["consignee_name"] and pack["mismatches"] == []
    invoice = next(d for d in pack["documents"] if d["doc_type"] == "commercial_invoice")
    assert invoice["fields"]["consignee_name"] == "MISSING: consignee_name" and invoice["status"] == "draft"
    assert invoice["fields"]["total_value_inr"] == round(BOOK["GRP-01"].weight_kg * 115.3)
    phyto = next(d for d in pack["documents"] if d["doc_type"] == PHYTO)
    assert phyto["status"] == "validated" and "ISPM 12" in phyto["fields"]["legal_basis"]
    again = draft_pack(BOOK["GRP-01"], option, MARKETS["LKCMB"], ROUTES["G-LK"], LOT, EXPORTER, "CODEX", None,
                       "PLN-0001", "Port closure · Jebel Ali", AT)
    assert again["sha256"] == pack["sha256"]  # deterministic, so the chain can vouch for it
    invoice["fields"]["net_weight_kg"] = 17000.0
    assert any("Packing List" in m for m in validate(pack["documents"]))
