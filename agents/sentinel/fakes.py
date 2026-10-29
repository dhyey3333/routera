"""In-memory stand-ins for the interfaces the Sentinel depends on.

Swap them for the team's real implementations: ContainerRepository -> CAP OData / SQLite repo,
AuditSink -> hash-chained audit log, EventBus -> team bus (spec section 9),
NetworkRepository -> the team's master data for corridors, carriers, facilities and markets.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from agents.sentinel.models import ContainerRef, Network, as_utc

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_CONTAINERS_PATH = DATA_DIR / "containers.json"  # the global book (v1 scenarios)
INDIA_CONTAINERS_PATH = DATA_DIR / "containers_india.json"  # the India-inside book (v2)
ALL_CONTAINER_PATHS = (DEFAULT_CONTAINERS_PATH, INDIA_CONTAINERS_PATH)
DEFAULT_NETWORK_PATH = DATA_DIR / "network.json"

BusListener = Callable[[str, dict[str, Any]], None]


class InMemoryBus:
    """Keeps every published message and forwards it to subscribers (the SSE stream, tests)."""

    def __init__(self) -> None:
        self.messages: list[tuple[str, dict[str, Any]]] = []
        self._listeners: list[BusListener] = []

    def publish(self, topic: str, message: dict[str, Any]) -> None:
        self.messages.append((topic, message))
        for listener in list(self._listeners):
            listener(topic, message)

    def subscribe(self, listener: BusListener) -> Callable[[], None]:
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener)

    def by_topic(self, topic: str) -> list[dict[str, Any]]:
        return [message for t, message in self.messages if t == topic]


@dataclass(frozen=True)
class AuditEntry:
    seq: int
    actor: str
    action: str
    details: dict[str, Any]
    rationale: str


AuditListener = Callable[[AuditEntry], None]


class InMemoryAudit:
    """Append-only audit trail. Refuses entries without a rationale (spec: every decision)."""

    def __init__(self) -> None:
        self.entries: list[AuditEntry] = []
        self._listeners: list[AuditListener] = []

    def log(self, actor: str, action: str, details: dict[str, Any], rationale: str) -> None:
        if not rationale.strip():
            raise ValueError(f"audit entry {action!r} has no rationale")
        entry = AuditEntry(len(self.entries) + 1, actor, action, dict(details), rationale)
        self.entries.append(entry)
        for listener in list(self._listeners):
            listener(entry)

    def subscribe(self, listener: AuditListener) -> Callable[[], None]:
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener)

    def actions(self) -> list[str]:
        return [entry.action for entry in self.entries]


class JsonContainerRepo:
    """Containers from JSON lists of ContainerRef, optionally limited to one scenario's book."""

    INACTIVE_STATUSES = frozenset({"delivered", "cancelled"})

    def __init__(
        self,
        path: Path | str | Sequence[Path | str] = DEFAULT_CONTAINERS_PATH,
        container_ids: Iterable[str] | None = None,
    ) -> None:
        paths = [path] if isinstance(path, str | Path) else list(path)
        raw = [item for p in paths for item in json.loads(Path(p).read_text(encoding="utf-8"))]
        containers = [ContainerRef.model_validate(item) for item in raw]
        by_id = {c.container_id: c for c in containers}
        if len(by_id) != len(containers):
            raise ValueError(f"duplicate container_id in {path}")
        if container_ids is None:
            self._containers = containers
        else:
            wanted = list(container_ids)
            missing = [cid for cid in wanted if cid not in by_id]
            if missing:
                raise ValueError(f"unknown container ids: {missing}")
            self._containers = [by_id[cid] for cid in wanted]

    def list_active(self) -> list[ContainerRef]:
        return [c for c in self._containers if c.status not in self.INACTIVE_STATUSES]


class JsonNetworkRepo:
    """The illustrative reference network (corridors, carriers, facilities, markets)."""

    def __init__(self, path: Path | str = DEFAULT_NETWORK_PATH) -> None:
        self._network = Network.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def network(self) -> Network:
        return self._network


class SimClock:
    """Simulated clock for replays. Only moves forward."""

    def __init__(self, start: datetime) -> None:
        self._now = as_utc(start)

    def now(self) -> datetime:
        return self._now

    def set(self, when: datetime) -> None:
        when = as_utc(when)
        if when < self._now:
            raise ValueError(f"clock cannot go back from {self._now} to {when}")
        self._now = when

    def advance(self, delta: timedelta) -> datetime:
        self.set(self._now + delta)
        return self._now


class ScriptedLLM:
    """Fake LLMClient: returns queued replies in order; a queued Exception instance is raised."""

    def __init__(self, news: Iterable[Any] = (), summaries: Iterable[Any] = ()) -> None:
        self._news = list(news)
        self._summaries = list(summaries)
        self.calls: list[str] = []

    def classify_news(self, headline: str, body: str) -> dict[str, Any]:
        self.calls.append("classify_news")
        return self._next(self._news)

    def summarise(self, disruption: dict[str, Any]) -> str:
        self.calls.append("summarise")
        return self._next(self._summaries)

    @staticmethod
    def _next(queue: list[Any]) -> Any:
        if not queue:
            raise RuntimeError("ScriptedLLM has no reply queued")
        reply = queue.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply
