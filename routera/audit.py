"""Approval & Audit: an append-only, hash-chained log. Each entry's hash covers the previous one,
so editing any past entry breaks verification from that point on. Implements the AuditSink."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

from agents.sentinel.models import Clock

GENESIS = "0" * 64


def _digest(entry: dict[str, Any]) -> str:
    body = {k: entry[k] for k in ("seq", "at", "actor", "action", "details", "rationale", "prev")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


class HashChainAudit:
    def __init__(
        self,
        clock: Clock | None = None,
        on_append: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.clock = clock
        self.on_append = on_append
        self.entries: list[dict[str, Any]] = []

    def log(self, actor: str, action: str, details: dict[str, Any], rationale: str) -> None:
        if not rationale.strip():
            raise ValueError(f"audit entry {action!r} has no rationale")
        entry = {
            "seq": len(self.entries) + 1,
            "at": self.clock.now().isoformat() if self.clock else None,
            "actor": actor,
            "action": action,
            "details": json.loads(json.dumps(details, default=str)),
            "rationale": rationale,
            "prev": self.entries[-1]["hash"] if self.entries else GENESIS,
        }
        entry["hash"] = _digest(entry)
        self.entries.append(entry)
        if self.on_append is not None:
            self.on_append(entry)

    def verify(self) -> dict[str, Any]:
        prev = GENESIS
        for entry in self.entries:
            if entry["prev"] != prev or _digest(entry) != entry["hash"]:
                return {"ok": False, "length": len(self.entries), "broken_at": entry["seq"]}
            prev = entry["hash"]
        return {"ok": True, "length": len(self.entries), "broken_at": None, "head": prev}

    def tail(self, n: int = 120) -> list[dict[str, Any]]:
        return self.entries[-n:]
