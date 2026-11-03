"""Compliance: a deterministic rules engine over the lot's facts and the team's cited rule packs.

Each check names the rule that governs it (routera/rulebook.py):
- block: a verified rule fails (e.g. the EU residue limit). Nobody can override it.
- review: a failing rule that is not verified (e.g. the Singapore MRL, still TODO_VERIFY). An unverified
  rule may never block, but the planner won't pick the option either; only a human may, after checking.
- docs: paperwork, a permit or a treatment to arrange, drafted for sign-off, with its days and cost.
- ok: nothing to do.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from agents.sentinel.models import ContainerRef
from routera.rulebook import RuleBook

DATA_DIR = Path(__file__).parent / "data"
FPO_BY_CROP = {"grapes": "Nashik Grape FPO", "mangoes": "Konkan Mango FPO"}
PHYTO_DOC = "phytosanitary_certificate_application_data"
KEY_LABEL = {"fsvp_agent_us": "US FSVP agent"}


@dataclass(frozen=True)
class Check:
    status: Literal["ok", "docs", "review", "block"]
    note: str
    extra_days: float = 0.0
    extra_cost_inr: float = 0.0
    draft: str | None = None
    rules: tuple[str, ...] = ()  # ids of the rules that decided this


class ComplianceRules:
    def __init__(self, path: Path = DATA_DIR / "compliance.json", book: RuleBook | None = None) -> None:
        raw = json.loads(path.read_text(encoding="utf-8"))
        self.markets: dict[str, dict[str, Any]] = raw["markets"]
        self.default_lot: dict[str, Any] = raw["default_lot"]
        self.lots: dict[str, dict[str, Any]] = raw["lots"]
        self.treatments: dict[str, dict[str, dict[str, Any]]] = raw.get("treatments", {})
        self.exporters: dict[str, dict[str, str]] = raw.get("exporters", {})
        self.book = book or RuleBook()

    def lot(self, c: ContainerRef) -> dict[str, Any]:
        override = self.lots.get(c.container_id, {})
        return {
            "mrl": {**self.default_lot["mrl"], **override.get("mrl", {})},
            "permits": override.get("permits", self.default_lot["permits"]),
            "phyto": override.get("phyto", [c.destination_country]),
            "treated": override.get("treated", []),
            "note": override.get("note", ""),
        }

    def treatment_note(self, c: ContainerRef, country: str) -> str | None:
        """The plant-health treatment `country` requires for this crop, and whether the lot has it."""
        treatment = (self.treatments.get(c.crop) or {}).get(country)
        if not treatment:
            return None
        if treatment["name"] in self.lot(c)["treated"]:
            return f"{treatment['name']} done"
        return f"{treatment['name']} at {treatment['facility']} (booking drafted)"

    def check(self, c: ContainerRef, country: str) -> Check:
        rules = self.markets.get(country)
        if rules is None or not rules["requires"]:
            return Check("ok", "domestic sale: no export paperwork")
        lot = self.lot(c)
        standard = rules.get("mrl_standard")
        mrl = self.book.find(country, c.crop, "max_residue", standard=standard) if standard else None
        if standard and lot["mrl"].get(standard) == "fail":
            note = f"residue test fails the {standard} MRL" + (f" ({lot['note']})" if lot["note"] else "")
            if mrl and mrl.effect == "block":
                return Check("block", note, rules=(mrl.rule_id,))
            why = f"{mrl.rule_id} is {mrl.status}" if mrl else "no rule pack covers it"
            return Check("review", f"{note}; {why}, so only a human may send it", rules=(mrl.rule_id,) if mrl else ())
        todo: list[tuple[str, float, float, str, str | None]] = []  # (note, days, cost, draft, rule id)
        permit = self.book.find(country, c.crop, "required_certificate", certificate="import_permit")
        if (permit or "import_permit" in rules["requires"]) and country not in lot["permits"]:
            days = rules.get("permit_days", 10)
            if permit and permit.effect == "block":
                return Check("block", f"no {country} import permit (takes about {days} d)", rules=(permit.rule_id,))
            basis = f"{permit.rule_id} ({permit.status}): {permit.source['title']}" if permit else "no rule pack"
            todo.append((
                f"{country} import permit applied for (+{days:g} d; rule unverified, so not a block)",
                days, rules.get("permit_cost_inr", 0),
                f"IMPORT PERMIT APPLICATION (DRAFT, needs sign-off)\nImporting country: {country}\nBasis: {basis}",
                permit.rule_id if permit else None,
            ))
        treatment = (self.treatments.get(c.crop) or {}).get(country)
        if treatment and treatment["name"] not in lot["treated"]:
            rule = self.book.find(country, c.crop, "required_treatment")
            todo.append((
                f"{treatment['name']} for {country} booked (+{treatment['days']:g} d)",
                treatment["days"], treatment["cost_inr"],
                f"TREATMENT BOOKING REQUEST (DRAFT, needs sign-off)\nTreatment: {treatment['name']} at "
                f"{treatment['facility']}\nRequired by: {rule.source['title'] if rule else country + ' rules'}",
                rule.rule_id if rule else None,
            ))
        phyto = self.book.find(country, c.crop, "required_document", document=PHYTO_DOC)
        if country not in lot["phyto"]:
            todo.append((
                f"phyto certificate for {country} drafted (+{rules['doc_days']:g} d)",
                rules["doc_days"], rules["doc_cost_inr"],
                f"PHYTOSANITARY CERTIFICATE RE-ISSUE REQUEST (DRAFT, needs sign-off)\nNew importing country: "
                f"{country}\nOriginal certificate: {', '.join(lot['phyto']) or 'none'}\n"
                f"Legal basis: {phyto.source['citation'] if phyto else 'ISPM 12'}\n"
                f"Residue results: within the {standard} MRL",
                phyto.rule_id if phyto else None,
            ))
        exporter = self.exporters.get(c.crop, {})
        for rule in self.book.applicable(country, c.crop):
            gaps = [k for k in rule.parameters.get("required_keys", []) if not exporter.get(k)]
            if rule.type == "traceability_requirement" and gaps:
                names = ", ".join(KEY_LABEL.get(g, g.replace("_", " ")) for g in gaps)
                todo.append((
                    f"{names} to name for traceability", 0.0, 0.0,
                    f"TRACEABILITY DETAILS NEEDED (DRAFT)\n{rule.title}: {', '.join(gaps)}\n"
                    f"Rule: {rule.source['citation']}", rule.rule_id,
                ))
        applied = tuple(r.rule_id for r in (mrl, phyto) if r)
        if not todo:
            return Check("ok", f"phyto for {country} held; residue within the {standard} MRL", rules=applied)
        header = (f"Exporter: {FPO_BY_CROP.get(c.crop, 'Grower FPO')}\nContainer: {c.container_id} · "
                  f"{c.crop} · {c.weight_kg:g} kg\nReason: market switch after a disruption (ROUTERA plan)")
        return Check(
            "docs",
            "; ".join(t[0] for t in todo),
            sum(t[1] for t in todo),
            sum(t[2] for t in todo),
            "\n\n".join([header, *(t[3] for t in todo)]),
            tuple(dict.fromkeys([*applied, *(t[4] for t in todo if t[4])])),
        )
