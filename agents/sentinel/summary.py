"""Pipeline step 9: the one-line summary.

With an LLM, the model writes a sentence of at most 25 words from a fixed subset of the
Disruption's fields, and the sentence is validated: one line, word and length limits, and no
number that is not in those facts. Anything else (no LLM, error, timeout, refusal, a failed
check) uses the spec's template.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

from agents.sentinel.models import Disruption, LLMClient


def count(n: int, noun: str = "container") -> str:
    """'1 container', '12 containers'."""
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"

log = logging.getLogger(__name__)

MAX_SUMMARY_WORDS = 25
MAX_SUMMARY_CHARS = 220
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def template_summary(d: Disruption) -> str:
    """Spec template: "{type} at {location}: {n} containers affected, ~{delay} days delay, ..."."""
    return (
        f"{d.type} at {d.location}: {count(len(d.affected_container_ids))} affected, "
        f"~{d.expected_delay_days:g} days delay, severity {d.severity}/5."
    )


def summary_facts(d: Disruption) -> dict[str, Any]:
    """The only data the LLM sees: no rationale, sources or free text from the feeds."""
    return {
        "type": d.type,
        "location": d.location,
        "status": d.status,
        "severity_out_of_5": d.severity,
        "priority": d.priority,
        "expected_delay_days": d.expected_delay_days,
        "delay_is_estimate": d.delay_is_estimate,
        "market_unreachable": d.market_unreachable,
        "cost_impact_pct": d.cost_impact_pct,
        "affected_containers": len(d.affected_container_ids),
        "affected_tonnes": round(d.affected_weight_kg / 1000, 1),
        "confidence_pct": round(d.confidence * 100, 1),
    }


def _number_forms(text: str) -> set[str]:
    forms: set[str] = set()
    for token in _NUMBER.findall(text):
        value = float(token)
        forms |= {token, f"{value:g}"}
        if value.is_integer():
            forms.add(str(int(value)))
    return forms


def validate_summary(reply: Any, facts: dict[str, Any]) -> str:
    """The reply as a clean one-liner, or ValueError saying why it cannot be used."""
    if not isinstance(reply, str):
        raise ValueError(f"summary is {type(reply).__name__}, not text")
    text = reply.strip().strip('"').strip()
    if not text:
        raise ValueError("summary is empty")
    if "\n" in text:
        raise ValueError("summary spans several lines")
    if len(text.split()) > MAX_SUMMARY_WORDS:
        raise ValueError(f"summary has more than {MAX_SUMMARY_WORDS} words")
    if len(text) > MAX_SUMMARY_CHARS:
        raise ValueError(f"summary is longer than {MAX_SUMMARY_CHARS} characters")
    allowed = _number_forms(json.dumps(facts)) | {"5"}  # "severity 3/5"
    invented = sorted(_number_forms(text) - allowed)
    if invented:
        raise ValueError(f"summary uses numbers not in the facts: {', '.join(invented)}")
    return text


@dataclass(frozen=True)
class SummaryResult:
    text: str
    by: Literal["llm", "template"]
    note: str = ""  # why the template was used, when an LLM was configured


class Summariser:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm

    def summarise(self, d: Disruption) -> SummaryResult:
        template = template_summary(d)
        if self.llm is None:
            return SummaryResult(template, "template")
        facts = summary_facts(d)
        try:
            return SummaryResult(validate_summary(self.llm.summarise(facts), facts), "llm")
        except Exception as exc:  # any LLM failure falls back to the template
            note = f"LLM summary rejected ({type(exc).__name__}: {str(exc)[:160]})"
            log.warning("%s for %s; using the template", note, d.disruption_id)
            return SummaryResult(template, "template", note)
