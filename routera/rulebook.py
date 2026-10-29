"""Rule packs: the team's YAML regulatory rules, each with its legal citation.

Ported from the team's governance module (github.com/kokilacg30/routera_compliance: rulepacks/,
rulepacks/loader.py, rulepacks/freshness.py). The loader keeps the team's non-negotiables:
- every rule cites its source (title and citation); an uncited rule must be an explained assumption
- only a verified rule may block. An unverified rule that says "block" is downgraded to needs_docs,
  with a warning
- a rule never verified, or last verified more than 180 days ago, is flagged stale
Packs resolve generic < international (ISPM 12) < destination, so a destination rule wins. The
compliance engine asks the book which rule governs each check, so every block, draft or human review
names its rule and how far that rule can be trusted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

RULEPACK_DIR = Path(__file__).parent / "rulepacks"
MAX_AGE_DAYS = 180
COMMODITY_GROUP = {"onion": "fresh_vegetables", "shrimp": "seafood"}  # everything else is fresh fruit
DOMESTIC = "IN"


@dataclass(frozen=True)
class Rule:
    rule_id: str
    pack_id: str
    title: str
    type: str
    effect: str  # block | needs_docs | warn
    status: str  # verified | assumption | TODO_VERIFY
    source: dict[str, str]
    parameters: dict[str, Any] = field(default_factory=dict)
    effective_date: str | None = None
    last_verified: str | None = None
    notes: str = ""

    def stale(self, today: date) -> str | None:
        """Why this rule needs re-verifying, or None when it is fresh."""
        if self.status == "TODO_VERIFY":
            return "marked TODO_VERIFY: needs official validation"
        if not self.last_verified:
            return "never verified"
        age = (today - date.fromisoformat(self.last_verified)).days
        return f"last verified {age} days ago (limit {MAX_AGE_DAYS})" if age > MAX_AGE_DAYS else None

    def cite(self, today: date | None = None) -> dict[str, Any]:
        out = {"rule_id": self.rule_id, "pack_id": self.pack_id, "title": self.title, "type": self.type,
               "effect": self.effect, "status": self.status, "source": self.source["title"],
               "url": self.source["url"], "citation": self.source["citation"], "notes": self.notes}
        if today is not None:
            out["stale"] = self.stale(today)
        return out


@dataclass(frozen=True)
class Pack:
    pack_id: str
    precedence: int
    destinations: tuple[str, ...] | None  # None means every destination
    groups: tuple[str, ...] | None
    rules: tuple[Rule, ...]


def _codes(value: Any) -> tuple[str, ...] | None:
    if value in (None, "*"):
        return None
    items = [value] if isinstance(value, str) else list(value)
    return None if "*" in items else tuple(str(v).upper() for v in items)


def _precedence(pack_id: str) -> int:
    if "generic" in pack_id:
        return 10
    if "ispm" in pack_id or "international" in pack_id:
        return 20
    return 30 if "destination" in pack_id else 15


class RuleBook:
    def __init__(self, folder: Path = RULEPACK_DIR) -> None:
        self.warnings: list[str] = []
        self.packs = sorted((self._load(f) for f in sorted(folder.glob("*.yaml"))),
                            key=lambda p: (p.precedence, p.pack_id))

    def _load(self, path: Path) -> Pack:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError(f"{path.name}: a rule pack must be a YAML mapping")
        pack_id = raw["pack_id"]
        rules = []
        for r in raw.get("rules", []):
            rule_id, status = r["rule_id"], r.get("verification_status", "TODO_VERIFY")
            source = r.get("source") or {}
            if not source and not (status == "assumption" and r.get("notes", "").strip()):
                raise ValueError(f"{path.name}: rule {rule_id} has no source citation")
            if source and not (source.get("title") and source.get("citation")):
                raise ValueError(f"{path.name}: rule {rule_id} needs a source title and citation")
            effect = r.get("effect_on_failure", "warn")
            if effect == "block" and status != "verified":
                self.warnings.append(f"{rule_id}: 'block' downgraded to 'needs_docs' because it is {status}")
                effect = "needs_docs"
            rules.append(Rule(
                rule_id=rule_id, pack_id=pack_id, title=r["title"], type=r["type"], effect=effect, status=status,
                source={k: str(source.get(k, "")) for k in ("title", "url", "citation")},
                parameters=r.get("parameters") or {}, effective_date=r.get("effective_date"),
                last_verified=r.get("last_verified"), notes=(r.get("notes") or "").strip(),
            ))
        scope = raw.get("scope", {})
        return Pack(pack_id, _precedence(pack_id), _codes(scope.get("destination")),
                    _codes(scope.get("commodity_groups")), tuple(rules))

    def applicable(self, country: str, crop: str) -> list[Rule]:
        """Rules for exporting `crop` to `country`, lowest precedence first; a domestic sale only
        meets the generic pack. A later rule with the same id overrides an earlier one."""
        group = COMMODITY_GROUP.get(crop, "fresh_fruit").upper()
        merged: dict[str, Rule] = {}
        for pack in self.packs:
            if country.upper() == DOMESTIC and pack.precedence != 10:
                continue
            if pack.destinations is not None and country.upper() not in pack.destinations:
                continue
            if pack.groups is not None and group not in pack.groups:
                continue
            for rule in pack.rules:
                if rule.parameters.get("crop") not in (None, crop):
                    continue  # a crop-specific rule, e.g. mango treatments
                merged.pop(rule.rule_id, None)
                merged[rule.rule_id] = rule
        return list(merged.values())

    def find(self, country: str, crop: str, kind: str, **params: Any) -> Rule | None:
        """The highest-precedence rule of type `kind` whose parameters match `params`."""
        for rule in reversed(self.applicable(country, crop)):
            if rule.type == kind and all(rule.parameters.get(k) == v for k, v in params.items()):
                return rule
        return None

    def all(self) -> list[Rule]:
        return [r for p in self.packs for r in p.rules]

    def to_verify_markdown(self, today: date) -> str:
        """RULES_TO_VERIFY.md: every rule that is unverified, an assumption, or stale."""
        lines = [
            "# ROUTERA: rules to verify",
            "",
            "No unverified rule may block a shipment; these only draft paperwork or ask a human.",
            "Check each against the primary source before marking it verified.",
            "",
            "| Rule | Pack | Title | Status | Why | Source |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        flagged = [r for r in self.all() if r.status != "verified" or r.stale(today)]
        for r in flagged:
            src = f"[{r.source['title']}]({r.source['url']}) ({r.source['citation']})" if r.source["url"] \
                else r.source["title"]
            lines.append(f"| `{r.rule_id}` | `{r.pack_id}` | {r.title} | `{r.status}` | "
                         f"{r.stale(today) or r.notes} | {src} |")
        lines += ["", f"**{len(flagged)} of {len(self.all())} rules to verify.**", ""]
        lines += [f"- Loader: {w}" for w in self.warnings]
        return "\n".join(lines) + "\n"
