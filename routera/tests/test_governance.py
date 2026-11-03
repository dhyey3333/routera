"""Governance ported from the team's module: cited rule packs, human review of unverified rules,
PDF and ZIP filing packs, and escalation that never turns silence into consent."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pytest

from routera.filing import build_zip, pdf_available
from routera.rulebook import RuleBook
from routera.tests.test_routera import moves, pending, run

TODAY = date(2026, 10, 30)


def test_every_rule_is_cited_and_only_verified_rules_can_block() -> None:
    book = RuleBook()
    assert book.all() and all(r.source["title"] and r.source["citation"] for r in book.all())
    assert all(r.status == "verified" for r in book.all() if r.effect == "block")
    assert any("MY-PERMIT-001" in w for w in book.warnings)  # declared block, unverified: downgraded
    assert {r.rule_id for r in book.applicable("IN", "grapes")} <= {r.rule_id for r in book.packs[0].rules}
    assert book.find("NL", "mangoes", "required_treatment").rule_id == "EU-MANGO-HWT-001"
    assert book.find("NL", "grapes", "required_treatment") is None  # mango rules skip grapes
    report = book.to_verify_markdown(TODAY)
    assert "SG-MRL-001" in report and "EU-MRL-001" not in report


def test_an_uncited_rule_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "bad.yaml").write_text(
        "pack_id: destination_xx\nscope: {destination: [XX]}\nrules:\n"
        "  - {rule_id: XX-1, title: t, type: other, verification_status: verified}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no source citation"):
        RuleBook(tmp_path)


def test_a_review_option_is_never_auto_picked_but_a_human_may_take_it_on_the_record() -> None:
    o = run("gulf_ripple")  # GLF-23 failed the Singapore MRL, whose rule is still TODO_VERIFY
    plan, row, option = next(
        (p, r, opt) for p in pending(o) for r in p["rows"] for opt in r["options"]
        if opt["compliance"]["status"] == "review" and opt["feasible"]
    )
    assert row["choice"] is None or row["choice"]["compliance"]["status"] != "review"
    assert option["human_only"] and "Human review" in " ".join(plan["guardrails"])
    assert f"{option['destination_name']} for {row['container_id']}" in " ".join(plan["guardrails"])
    o.approve(plan["plan_id"], "Asha K.", {row["container_id"]: option["route_id"]})
    [approved] = [e for e in o.chain.entries if e["action"] == "plan.approved"]
    assert approved["details"]["reviewed_rules"] == {row["container_id"]: option["compliance"]["rules"]}


def test_the_filing_pack_zip_has_a_cover_sheet_pdfs_and_a_manifest_and_is_reproducible() -> None:
    o = run("gulf_closure")
    plan = max(pending(o), key=lambda p: p["saved_inr"])
    o.approve(plan["plan_id"], "Asha K.")
    cid = moves(plan)[0]["container_id"]
    pack = o.paperwork(plan["plan_id"], cid)
    data = build_zip(pack)
    assert data == build_zip(pack)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = z.namelist()
        manifest = json.loads(z.read("manifest.json"))
        cover = z.read("COVER_SHEET.md").decode()
        assert names[0] == "COVER_SHEET.md" and "manifest.json" in names
        assert manifest["pack_sha256"] == manifest["on_chain"] == pack["sha256"]
        assert {f["path"] for f in manifest["files"]} == set(names) - {"manifest.json"}
        if pdf_available():
            pdf = next(n for n in names if n.endswith(".pdf"))
            assert z.read(pdf).startswith(b"%PDF")
    assert "Human checklist" in cover and "ISPM12-REQ-001" in cover and "matches the audit chain" in cover


def test_a_silent_plan_escalates_up_the_chain_and_never_approves_itself() -> None:
    o = run("gulf_closure")
    plan = max(pending(o), key=lambda p: p["saved_inr"])
    created = o.clock.now()  # treat it as a plan that has just arrived
    plan.update(created_at=created.isoformat(), escalation=None)
    o.clock.set(created + timedelta(hours=1))
    o.snapshot()
    assert plan["escalation"]["level"] == 0 and plan["escalation"]["next"] == "Export manager"
    o.clock.set(created + timedelta(hours=2.5))
    o.snapshot()
    assert plan["escalation"]["role"] == "Export manager" and not plan["escalation"]["breached"]
    o.clock.set(created + timedelta(hours=9))
    o.snapshot()
    o.snapshot()  # no new level: logged once
    steps = [e for e in o.chain.entries if e["action"] == "plan.escalated" and e["details"]["plan_id"] ==
             plan["plan_id"]]
    assert [e["details"]["role"] for e in steps][-2:] == ["Export manager", "FPO director"]
    assert steps[-1]["details"]["breached"] and "never consent" in steps[-1]["rationale"]
    assert plan["status"] == "pending" and not [b for b in o.bookings if b["plan_id"] == plan["plan_id"]]
    assert o.chain.verify()["ok"]
