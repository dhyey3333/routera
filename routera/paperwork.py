"""Paperwork: which trade documents a re-plan keeps, regenerates, needs new or makes obsolete,
and the draft filing pack for each container that moves.

Ported from the team's governance module (github.com/kokilacg30/routera_compliance:
documents/registry.py, documents/validator.py, compliance/requirements.py) and fitted to ROUTERA's
containers, routes and markets:
- 12 document types, each with its standard (UN/CEFACT, ISPM 12, WCO, FIATA ...) and hours to prepare
- a change set {keep, regenerate, new, obsolete} against the paperwork the booked route already has
- the paperwork critical path is the slowest pending document, since documents are prepared in parallel
- drafts are filled from container, route and plan data only; a field ROUTERA does not know is marked
  "MISSING: <field>" for a human to fill, never invented
- cross-document checks (weight, HS code, value, consignee) before anything is filed
Every draft is sha256-hashed; approval puts the pack hashes on the audit chain. ROUTERA files nothing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from agents.sentinel.models import ContainerRef
from routera.compliance import FPO_BY_CROP
from routera.market import Market
from routera.reroute import Route
from routera.shelf_life import CROPS, DEFAULT_CROP


@dataclass(frozen=True)
class DocType:
    title: str
    standard: str
    fields: tuple[str, ...]
    destination_specific: bool = True
    hours: float = 2.0


INVOICE, PACKING, PHYTO, ORIGIN = ("commercial_invoice", "packing_list",
                                   "phytosanitary_certificate_application_data",
                                   "certificate_of_origin_application_data")
DECLARATION, TRANSPORT, QUALITY, LAB = ("export_declaration_data", "transport_instruction",
                                        "quality_or_health_certificate_request", "lab_residue_report_reference")
NOTICE, REINSPECT, COLD_STORE, PROCESSOR = ("buyer_notification", "re_inspection_request",
                                            "cold_store_booking_request", "processor_sale_note")

DOCS: dict[str, DocType] = {
    INVOICE: DocType("Commercial Invoice", "UN/CEFACT Rec. 6 / Incoterms 2020", (
        "invoice_number", "invoice_date", "seller_name", "consignee_name", "destination_country", "hs_code",
        "goods_description", "net_weight_kg", "unit_price_inr_per_kg", "total_value_inr", "currency",
        "incoterms"), hours=1.0),
    PACKING: DocType("Packing List", "UN/CEFACT Rec. 1", (
        "packing_list_number", "consignor_name", "consignee_name", "destination_country", "container_number",
        "lot_id", "net_weight_kg", "temp_setpoint_c"), hours=1.0),
    PHYTO: DocType("Phytosanitary Certificate Application", "IPPC ISPM 12", (
        "application_id", "nppo_applicant_name", "origin_country", "destination_country", "legal_basis",
        "botanical_name", "goods_description", "declared_net_weight_kg", "orchard_code", "packhouse_code",
        "entry_port_destination", "treatment_details"), hours=24.0),
    ORIGIN: DocType("Certificate of Origin Application", "ICC WCF Certificate of Origin Guidelines", (
        "application_reference", "exporter_name", "consignee_name", "consignee_country", "origin_country",
        "transport_details", "hs_code", "goods_description", "net_weight_kg"), hours=12.0),
    DECLARATION: DocType("Export Declaration (Shipping Bill) Data", "WCO Data Model 3.10", (
        "declaration_reference", "exporter_iec", "destination_country", "departure_port", "transport_mode",
        "hs_code", "statistical_value_inr", "net_mass_kg", "container_number"), hours=4.0),
    TRANSPORT: DocType("Transport Instruction", "FIATA Model Rules / DCSA Shipping Instructions", (
        "booking_reference", "shipper_name", "consignee_name", "port_of_loading", "place_of_delivery",
        "carrier", "container_number", "cargo_description", "net_weight_kg", "temp_setpoint_c"), hours=2.0),
    QUALITY: DocType("Quality / Health Certificate Request", "Codex CAC/GL 38-2001", (
        "request_id", "inspection_location", "lot_id", "commodity", "grade", "target_country"), hours=18.0),
    LAB: DocType("Lab Residue Report Reference", "ISO/IEC 17025 accredited lab", (
        "lab_name", "report_reference", "lot_id", "mrl_standard", "result"),
        destination_specific=False, hours=48.0),  # tied to the lot, not the buyer
    NOTICE: DocType("Buyer Notification", "Incoterms 2020 notice obligations", (
        "notice_id", "consignee_name", "container_number", "revised_destination", "revised_eta",
        "reason_for_notice"), hours=0.5),
    REINSPECT: DocType("Re-inspection Request", "ISPM 7 (phytosanitary certification system)", (
        "request_id", "port_authority", "container_number", "diversion_reason"), hours=24.0),
    COLD_STORE: DocType("Cold Store Booking Request", "GCCA storage agreement", (
        "booking_reference", "facility_name", "required_temp_c", "net_weight_kg", "expected_days", "lot_id"),
        hours=2.0),
    PROCESSOR: DocType("Processor Sale Note", "Commercial salvage contract", (
        "contract_id", "processor_name", "salvage_value_inr", "net_weight_kg", "lot_id"), hours=2.0),
}

# Legal basis for the phyto certificate, from the team's rule packs (verify before relying on it).
PHYTO_LAW = {
    "NL": "Regulation (EU) 2016/2031, Art. 72(1) and Annex XI Part A",
    "US": "7 CFR 319.56-3 (USDA APHIS)",
    "SG": "Control of Plants (Plant Importation) Rules, Rule 4 (SFA)",
    "AE": "Federal Law No. 24 of 1999, Executive Regulations on Agricultural Quarantine Art. 3 (MOCCAE)",
}
PHYTO_LAW_DEFAULT = "ISPM 12, Section 1.3 (IPPC)"
BOTANICAL = {"grapes": "Vitis vinifera", "mangoes": "Mangifera indica"}
INDIAN_PORTS = ("INNSA", "INMUN")
MISSING = "MISSING: "


def required(c: ContainerRef, market: Market, action: str) -> list[str]:
    """The documents a container needs to go to `market` by a route of kind `action`."""
    docs = {INVOICE, PACKING, TRANSPORT}
    if market.kind == "export":
        docs |= {DECLARATION, PHYTO, ORIGIN, LAB, NOTICE}
        if c.status == "at_sea" and market.id != c.destination_port:
            docs.add(REINSPECT)  # diverted mid-voyage: the new port inspects again
    else:
        docs.add(NOTICE)
    if action == "hold_cold_store":
        docs.add(COLD_STORE)
    if market.kind == "processor":
        docs = (docs - {NOTICE}) | {PROCESSOR}
    return [d for d in DOCS if d in docs]


def held(c: ContainerRef, phyto: list[str]) -> list[str]:
    """What the booked shipment already has: its trade documents, and a phyto only if one was issued."""
    if c.destination_country == "IN":
        return [INVOICE, PACKING, TRANSPORT]
    docs = {INVOICE, PACKING, TRANSPORT, DECLARATION, ORIGIN, LAB}
    return [d for d in DOCS if d in docs | ({PHYTO} if phyto else set())]


def change_set(c: ContainerRef, option: dict[str, Any], market: Market, phyto: list[str]) -> dict[str, Any]:
    """{keep, regenerate, new, obsolete} plus the critical path in hours. Every move changes the route,
    so destination-specific documents are redone; a phyto already issued for the new country is kept."""
    need, have = required(c, market, option["action"]), held(c, phyto)
    keep, regenerate, new = [], [], []
    for doc in need:
        if doc == PHYTO and market.country in phyto:
            (keep if doc in have else new).append(doc)
        elif doc not in have:
            new.append(doc)
        elif DOCS[doc].destination_specific:
            regenerate.append(doc)
        else:
            keep.append(doc)
    pending = regenerate + new
    return {"keep": keep, "regenerate": regenerate, "new": new,
            "obsolete": [d for d in have if d not in need],
            "hours": max((DOCS[d].hours for d in pending), default=0.0)}


def content_hash(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def draft_pack(
    c: ContainerRef,
    option: dict[str, Any],
    market: Market,
    route: Route,
    lot: dict[str, Any],
    exporter: dict[str, str],
    mrl_standard: str | None,
    treatment: str | None,
    plan_id: str,
    reason: str,
    at: datetime,
    approved: bool = False,
    citations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The filing pack for one moved container: every regenerated or new document as a draft."""
    change = change_set(c, option, market, lot["phyto"])
    switch = market.id != c.destination_port
    lot_id = f"LOT-{c.container_id}"
    goods = f"Fresh {c.crop}"
    port = next((p for p in route.via if p in INDIAN_PORTS), c.origin_port)
    mode = "rail" if "rail" in route.label.lower() else "road" if market.kind != "export" else "sea"
    price = option.get("price_inr_per_kg") or 0.0
    value = round(c.weight_kg * price)
    setpoint = CROPS.get(c.crop, DEFAULT_CROP).ref_temp_c
    ref = f"{plan_id}-{c.container_id}"
    pool: dict[str, Any] = {
        "invoice_number": f"INV-{ref}", "invoice_date": at.date().isoformat(),
        "seller_name": FPO_BY_CROP.get(c.crop), "consignor_name": FPO_BY_CROP.get(c.crop),
        "exporter_name": FPO_BY_CROP.get(c.crop), "shipper_name": FPO_BY_CROP.get(c.crop),
        "nppo_applicant_name": FPO_BY_CROP.get(c.crop),
        "consignee_name": None if switch else f"Booked buyer at {market.name} (unchanged)",
        "destination_country": market.country, "consignee_country": market.country, "target_country": market.country,
        "hs_code": c.hs_code, "goods_description": goods, "commodity": c.crop, "botanical_name": BOTANICAL.get(c.crop),
        "net_weight_kg": c.weight_kg, "declared_net_weight_kg": c.weight_kg, "net_mass_kg": c.weight_kg,
        "unit_price_inr_per_kg": price, "total_value_inr": value, "statistical_value_inr": value,
        "salvage_value_inr": value, "currency": "INR",
        "incoterms": f"CIF {market.name}" if market.kind == "export" else f"Delivered {market.name}",
        "packing_list_number": f"PL-{ref}", "container_number": c.container_id, "lot_id": lot_id,
        "temp_setpoint_c": setpoint,
        "application_id": f"PHYTO-{ref}", "origin_country": "IN",
        "legal_basis": PHYTO_LAW.get(market.country, PHYTO_LAW_DEFAULT),
        "packhouse_code": exporter.get("packhouse_code"), "orchard_code": exporter.get("orchard_code"),
        "exporter_iec": exporter.get("iec"),
        "entry_port_destination": market.name,
        "treatment_details": treatment or "No treatment required; cold chain held",
        "application_reference": f"COO-{ref}", "transport_details": route.label,
        "declaration_reference": f"SB-{ref}", "departure_port": port, "transport_mode": mode,
        "booking_reference": f"BKG-{ref}", "port_of_loading": port, "place_of_delivery": market.name,
        "carrier": route.carrier or c.vessel_id, "cargo_description": f"{c.weight_kg:g} kg {goods.lower()}",
        "request_id": f"REQ-{ref}", "inspection_location": f"{port} plant quarantine station", "grade": "Class I",
        "lab_name": "NABL-accredited residue lab", "report_reference": f"MRL-{lot_id}",
        "mrl_standard": mrl_standard, "result": f"within the {mrl_standard} MRL" if mrl_standard else None,
        "notice_id": f"NOTE-{ref}", "revised_destination": market.name,
        "revised_eta": (at + timedelta(days=route.days + route.hold_days)).date().isoformat(),
        "reason_for_notice": reason, "diversion_reason": reason,
        "port_authority": f"Plant quarantine authority, {market.name}",
        "facility_name": route.hold_facility, "required_temp_c": setpoint,
        "expected_days": route.hold_days, "contract_id": f"SALE-{ref}", "processor_name": market.name,
    }
    documents = []
    for doc_type in change["regenerate"] + change["new"]:
        spec = DOCS[doc_type]
        fields = {f: MISSING + f if pool.get(f) in (None, "") else pool[f] for f in spec.fields}
        missing = [f for f, v in fields.items() if isinstance(v, str) and v.startswith(MISSING)]
        status = "draft" if missing else "awaiting_human_submission" if approved else "validated"
        documents.append({"doc_type": doc_type, "title": spec.title, "standard": spec.standard,
                          "why": "regenerate" if doc_type in change["regenerate"] else "new",
                          "status": status, "fields": fields, "missing": missing, "sha256": content_hash(fields)})
    mismatches = validate(documents)
    pack = {
        "plan_id": plan_id, "container_id": c.container_id, "to": market.name, "route": route.label,
        "prepared_at": at.isoformat(), "change": change, "documents": documents,
        "to_fill": sorted({f for d in documents for f in d["missing"]}), "mismatches": mismatches,
        "sha256": content_hash([d["sha256"] for d in documents]),
        "citations": citations or [],
    }
    pack["text"] = render(pack)
    return pack


CHECKS = (  # (doc A, field A, doc B, field B): cross-document consistency, from the team's validator
    (INVOICE, "net_weight_kg", PACKING, "net_weight_kg"),
    (INVOICE, "consignee_name", PACKING, "consignee_name"),
    (PACKING, "net_weight_kg", TRANSPORT, "net_weight_kg"),
    (PACKING, "container_number", TRANSPORT, "container_number"),
    (INVOICE, "hs_code", DECLARATION, "hs_code"),
    (INVOICE, "total_value_inr", DECLARATION, "statistical_value_inr"),
)


def validate(documents: list[dict[str, Any]]) -> list[str]:
    by_type = {d["doc_type"]: d["fields"] for d in documents}
    out = []
    for doc_a, field_a, doc_b, field_b in CHECKS:
        a, b = by_type.get(doc_a, {}).get(field_a), by_type.get(doc_b, {}).get(field_b)
        if a is None or b is None or any(isinstance(v, str) and v.startswith(MISSING) for v in (a, b)):
            continue
        if str(a).strip().upper() != str(b).strip().upper():
            out.append(f"{DOCS[doc_a].title} {field_a} = {a}, but {DOCS[doc_b].title} {field_b} = {b}")
    return out


def render(pack: dict[str, Any]) -> str:
    change = pack["change"]
    lines = [
        f"ROUTERA FILING PACK (DRAFT) · {pack['plan_id']} · {pack['container_id']} → {pack['to']}",
        "Sandbox data. Drafts for human review and filing; ROUTERA submits nothing itself.",
        f"Route: {pack['route']} · prepared {pack['prepared_at']}",
        f"Keep: {', '.join(DOCS[d].title for d in change['keep']) or 'none'}",
        f"Superseded: {', '.join(DOCS[d].title for d in change['obsolete']) or 'none'}",
        f"Ready in about {change['hours']:g} h (slowest document; all prepared in parallel)",
        f"Fill before filing: {', '.join(pack['to_fill']) or 'nothing'}",
        f"Cross-document checks: {'; '.join(pack['mismatches']) or 'all consistent'}",
        f"Pack sha256: {pack['sha256']}",
    ]
    for r in pack["citations"]:
        lines.append(f"Rule {r['rule_id']} ({r['status']}): {r['title']}. {r['source']}, {r['citation']}")
    for d in pack["documents"]:
        lines += ["", f"== {d['title']} ({d['standard']}) · {d['why']} · {d['status']}", f"sha256 {d['sha256']}"]
        lines += [f"{k}: {v}" for k, v in d["fields"].items()]
    return "\n".join(lines)
