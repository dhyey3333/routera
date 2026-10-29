"""Filing packs: PDF drafts and the ZIP a human files from.

Ported from the team's governance module (documents/generator.py render_pdf, documents/filing_pack.py):
- each draft renders as a PDF stamped DRAFT, with fields still to fill in red
- the ZIP holds a cover sheet (summary, human checklist, legal citations), every document as JSON and
  PDF, and manifest.json with a sha256 for each file
PDFs are rendered with reportlab in invariant mode and the ZIP uses fixed timestamps, so the same pack
always gives the same bytes. Without reportlab the ZIP still carries the cover sheet and the JSON.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import datetime
from typing import Any
from xml.sax.saxutils import escape

MISSING = "MISSING: "
CHECKLIST = (
    "Check the invoice amount and the new buyer against the sales contract",
    "Fill every field marked MISSING, then sign",
    "Lodge the phyto application with NPPO India (PQMS) and the shipping bill on ICEGATE",
    "Send the transport instruction to the carrier and confirm the reefer setpoint",
    "Send the buyer notification and the amended invoice to the consignee",
)


UNITS = (("_inr_per_kg", " (INR per kg)"), ("_inr", " (INR)"), ("_kg", " (kg)"), ("_c", " (°C)"))
ACRONYMS = {"hs": "HS", "nppo": "NPPO", "iec": "IEC", "id": "ID", "eta": "ETA"}


def label(key: str) -> str:
    """'total_value_inr' -> 'Total value (INR)', 'hs_code' -> 'HS code'."""
    unit = ""
    for suffix, text in UNITS:
        if key.endswith(suffix):
            key, unit = key[: -len(suffix)], text
            break
    words = " ".join(ACRONYMS.get(w, w) for w in key.split("_"))
    return words[:1].upper() + words[1:] + unit


def pdf_available() -> bool:
    try:
        import reportlab  # noqa: F401
    except ImportError:
        return False
    return True


def _plain(value: Any) -> str:
    """Text the PDF's built-in Helvetica can draw (no arrows or rupee sign)."""
    return escape(str(value).replace("→", "->").replace("₹", "Rs ").replace("✍️", ""))


def render_pdf(doc: dict[str, Any], pack: dict[str, Any]) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    title = ParagraphStyle("T", parent=styles["Heading1"], fontSize=15, leading=19, alignment=1,
                           textColor=colors.HexColor("#1A365D"), spaceAfter=4)
    stamp = ParagraphStyle("S", parent=styles["Normal"], fontSize=9, leading=12, alignment=1,
                           textColor=colors.HexColor("#C53030"), spaceAfter=10)
    head = ParagraphStyle("H", parent=styles["Heading2"], fontSize=10.5, leading=13,
                          textColor=colors.HexColor("#2B6CB0"), spaceBefore=6, spaceAfter=4)
    cell = ParagraphStyle("C", parent=styles["Normal"], fontSize=8, leading=10)
    grid = [("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#A0AEC0")),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]

    def p(text: str) -> Paragraph:
        return Paragraph(text, cell)

    meta = [
        [p("<b>Standard</b>"), p(_plain(doc["standard"])),
         p("<b>Status</b>"), p(_plain(doc["status"].replace("_", " ")))],
        [p("<b>Plan / container</b>"), p(_plain(f"{pack['plan_id']} / {pack['container_id']}")),
         p("<b>Prepared</b>"), p(_plain(pack["prepared_at"][:16].replace("T", " ") + " UTC"))],
        [p("<b>Destination</b>"), p(_plain(pack["to"])), p("<b>Reason</b>"), p(_plain(doc["why"]))],
    ]
    rows = []
    for key, value in doc["fields"].items():
        missing = isinstance(value, str) and value.startswith(MISSING)
        shown = "<font color='#C53030'><b>MISSING: to be filled by a human</b></font>" if missing else _plain(value)
        rows.append([p(f"<b>{_plain(label(key))}</b>"), p(shown)])
    story = [
        Paragraph(_plain(f"ROUTERA: {doc['title'].upper()}"), title),
        Paragraph("DRAFT FOR HUMAN REVIEW. To be checked, signed and filed by an authorised person on the "
                  "official channel. Simulated data.", stamp),
        Table(meta, colWidths=[85, 175, 70, 175], style=TableStyle(
            [*grid, ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7FAFC"))])),
        Spacer(1, 8),
        Paragraph("PARTICULARS", head),
        Table(rows, colWidths=[150, 355], style=TableStyle(
            [*grid, ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")])])),
        Spacer(1, 10),
        p(f"<b>Content sha256:</b> {doc['sha256']}"),
        Spacer(1, 6),
        p("<b>Notice:</b> ROUTERA prepared this draft from plan data. It is not a government certificate, "
          "customs entry or bill of lading until a human validates it and files it with the competent authority."),
    ]
    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36,
                      title=f"{doc['title']} {pack['container_id']}", author="ROUTERA (draft)",
                      invariant=1).build(story)
    return buf.getvalue()


def cover_sheet(pack: dict[str, Any], files: list[str]) -> str:
    change = pack["change"]
    lines = [
        "# ROUTERA filing pack: cover sheet",
        f"**Plan** `{pack['plan_id']}` · **Container** `{pack['container_id']}` · **To** {pack['to']} · "
        f"**Prepared** {pack['prepared_at']} · **Status** DRAFT, ready for human filing",
        "",
        "> ROUTERA never files with a government system. An authorised person reviews, signs and lodges",
        "> these documents on the official channels. All data is simulated.",
        "",
        f"- Route: {pack['route']}",
        f"- Ready in about {change['hours']:g} h (slowest document; all prepared in parallel)",
        f"- Kept from the booked shipment: {', '.join(change['keep']) or 'none'}",
        f"- Superseded: {', '.join(change['obsolete']) or 'none'}",
        f"- Fields to fill: {', '.join(pack['to_fill']) or 'none'}",
        f"- Cross-document checks: {'; '.join(pack['mismatches']) or 'all consistent'}",
        f"- Pack sha256: `{pack['sha256']}`" + (" (matches the audit chain)" if pack.get("on_chain") == pack["sha256"]
                                                else ""),
        "",
        "## Documents",
        "| Document | Why | Status | sha256 |",
        "| :--- | :--- | :--- | :--- |",
        *[f"| {d['title']} | {d['why']} | {d['status']} | `{d['sha256'][:16]}…` |" for d in pack["documents"]],
        "",
        "## Human checklist",
        *[f"{i}. [ ] {step}" for i, step in enumerate(CHECKLIST, 1)],
        "",
        "## Legal basis (from the team's rule packs)",
        *[f"- **{r['rule_id']}** ({r['status']}): {r['title']}. {r['source']}, {r['citation']}"
          + (f" <{r['url']}>" if r["url"] else "") for r in pack["citations"]],
        "",
        "## Files",
        *[f"- {f}" for f in files],
        "",
    ]
    return "\n".join(lines)


def build_zip(pack: dict[str, Any]) -> bytes:
    stamp = datetime.fromisoformat(pack["prepared_at"]).timetuple()[:6]
    files: dict[str, bytes] = {}
    described: list[dict[str, str]] = []
    with_pdf = pdf_available()
    for i, doc in enumerate(pack["documents"], 1):
        stem = f"{i:02d}_{doc['doc_type']}"
        files[f"{stem}.json"] = json.dumps({k: doc[k] for k in ("doc_type", "title", "standard", "why", "status",
                                                              "sha256", "missing", "fields")},
                                           indent=2, ensure_ascii=False).encode()
        described.append({"path": f"{stem}.json", "doc_type": doc["doc_type"], "description": "structured data"})
        if with_pdf:
            files[f"{stem}.pdf"] = render_pdf(doc, pack)
            described.append({"path": f"{stem}.pdf", "doc_type": doc["doc_type"], "description": "draft PDF"})
    files = {"COVER_SHEET.md": cover_sheet(pack, list(files)).encode(), **files}
    described.insert(0, {"path": "COVER_SHEET.md", "doc_type": "cover_sheet", "description": "summary, checklist"})
    for entry in described:
        entry["sha256"] = hashlib.sha256(files[entry["path"]]).hexdigest()
    manifest = {
        "plan_id": pack["plan_id"], "container_id": pack["container_id"], "prepared_at": pack["prepared_at"],
        "pack_sha256": pack["sha256"], "on_chain": pack.get("on_chain"), "pdfs": with_pdf,
        "files": described, "rule_citations": pack["citations"], "human_checklist": list(CHECKLIST),
    }
    files["manifest.json"] = json.dumps(manifest, indent=2, ensure_ascii=False).encode()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(zipfile.ZipInfo(name, date_time=stamp), data, compress_type=zipfile.ZIP_DEFLATED)
    return buf.getvalue()
