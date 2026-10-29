"""DisruptionSentinel: normalise -> classify -> merge/lifecycle -> publish + audit.

Publishes disruption.detected / .updated / .resolved only on material change (status, type,
location, severity, affected set, or delay moved by >= 1 day). Candidates are never published.
Every decision, and every rejected event, goes to the audit sink with its rationale.

Summaries: with an LLM configured, a disruption gets an LLM summary when it is published (never
while it is an unconfirmed candidate); it is cached per material state and shown on every view the
Sentinel hands out. Without one, or when the LLM reply fails validation, the template is used.

describe() adds what the UI and the Scenario Planner need beyond the contract: the layer
("india" for the domestic leg, "global" for sea / foreign markets), the option families the
planner can rank for this type (config `options_by_type`), and the other disruptions that share
containers with it (how a global shock ripples into India).

subscribe() gives a live feed for the replay CLI and the UI stream: ("signal", info),
("decision", info), ("publish", bus message) and ("disruption", state) events, in that order.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from agents.sentinel import scoring
from agents.sentinel.classify import Classifier
from agents.sentinel.fakes import JsonNetworkRepo
from agents.sentinel.merge import Decision, DisruptionTracker
from agents.sentinel.models import (
    AUDIT_ACTOR,
    DOMESTIC_TYPES,
    AuditSink,
    BusMessage,
    Clock,
    ContainerRepository,
    Disruption,
    EventBus,
    LLMClient,
    NetworkRepository,
    NewsPayload,
    RawEvent,
    Signal,
)
from agents.sentinel.normalise import LOCODE_RE, Gazetteer, NormalisationError, Normaliser
from agents.sentinel.settings import SentinelConfig, load_config
from agents.sentinel.summary import Summariser, SummaryResult

SentinelListener = Callable[[str, dict[str, Any]], None]
LISTED_STATUSES = frozenset({"confirmed", "active", "resolving"})
CONTRACT_FIELDS: tuple[str, ...] = tuple(Disruption.model_fields)


def material_changes(before: Disruption, after: Disruption, config: SentinelConfig) -> list[str]:
    """Why `after` is worth publishing again; empty when the change is not material."""
    reasons = [
        f
        for f in ("status", "type", "location", "severity")
        if getattr(before, f) != getattr(after, f)
    ]
    if set(before.affected_container_ids) != set(after.affected_container_ids):
        reasons.append("affected_container_ids")
    delay_moved = abs(after.expected_delay_days - before.expected_delay_days)
    if delay_moved >= config.publish.delay_change_days:
        reasons.append("expected_delay_days")
    return reasons


def changed_fields(before: Disruption, after: Disruption) -> list[str]:
    return [
        f
        for f in CONTRACT_FIELDS
        if f != "last_updated" and getattr(before, f) != getattr(after, f)
    ]


def _summary_key(d: Disruption) -> tuple[Any, ...]:
    """A new summary is written only when what it would say changes."""
    return (
        d.disruption_id,
        d.type,
        d.location,
        d.status,
        d.severity,
        len(d.affected_container_ids),
        round(d.expected_delay_days, 1),
        d.cost_impact_pct,
    )


def _title(signal: Signal) -> str:
    if isinstance(signal.payload, NewsPayload):
        return f"{signal.payload.headline} ({signal.payload.outlet})"
    return signal.reason.split(": ", 1)[-1]


def _loose_ids(event: RawEvent | Mapping[str, Any]) -> dict[str, Any]:
    """Best-effort identifiers of an event that failed validation."""
    if isinstance(event, RawEvent):
        return {
            "event_id": event.event_id,
            "source": event.source,
            "received_at": event.received_at.isoformat(),
        }
    if isinstance(event, Mapping):
        return {
            key: str(event[key])
            for key in ("event_id", "source", "received_at")
            if event.get(key) is not None
        }
    return {}


class DisruptionSentinel:
    def __init__(
        self,
        containers: ContainerRepository,
        bus: EventBus,
        audit: AuditSink,
        clock: Clock,
        llm: LLMClient | None = None,
        config: SentinelConfig | None = None,
        network: NetworkRepository | None = None,
    ) -> None:
        self.config = config or load_config()
        self.network = (network or JsonNetworkRepo()).network()
        codes = self.network.codes()
        gazetteer = Gazetteer(self.config.locations, codes)
        self.normaliser = Normaliser(self.config, gazetteer)
        self.classifier = Classifier(self.config, gazetteer, llm)
        self.tracker = DisruptionTracker(self.config, gazetteer, containers, clock, self.network)
        indian_regions = {r for r, members in gazetteer.regions.items() if set(members) <= codes}
        self._indian_places = codes | indian_regions
        self.bus = bus
        self.audit = audit
        self.clock = clock
        self.llm = llm
        self.summariser = Summariser(llm)
        self._summaries: dict[tuple[Any, ...], SummaryResult] = {}
        self._published: dict[str, Disruption] = {}
        self._listeners: list[SentinelListener] = []

    # --- spec interface -------------------------------------------------------------------------

    def ingest(self, event: RawEvent | Mapping[str, Any]) -> list[Disruption]:
        """Process one feed message. Returns the disruptions it created or updated."""
        try:
            signal = self.classifier.classify(self.normaliser.normalise(event))
        except NormalisationError as exc:
            self._reject(event, exc)
            return []
        decisions = self.tracker.add(signal)
        self._emit("signal", self._signal_info(signal, decisions))
        touched = [d.disruption_id for d in decisions if d.disruption_id]
        return self._settle(decisions, touched)

    def tick(self) -> list[Disruption]:
        """Clock-driven lifecycle (expiry, start times, resolution). Returns what changed."""
        decisions = self.tracker.tick()
        open_ids = [i.disruption_id for i in self.tracker.incidents if i.is_open]
        touched = [d.disruption_id for d in decisions if d.disruption_id]
        return self._settle(decisions, [*touched, *open_ids])

    def active(self) -> list[Disruption]:
        """Confirmed, active and resolving disruptions (what the orchestrator acts on)."""
        return [
            self._present(i.disruption)
            for i in self.tracker.incidents
            if i.is_open and i.disruption.status in LISTED_STATUSES
        ]

    # --- extras for the API, UI and tests -------------------------------------------------------

    def disruptions(self) -> list[Disruption]:
        """Every disruption ever opened, candidates and closed ones included."""
        return [self._present(i.disruption) for i in self.tracker.incidents]

    def describe(self, disruption_id: str) -> dict[str, Any]:
        incident = self.tracker.get(disruption_id)
        summary = self._summaries.get(_summary_key(incident.disruption))
        return {
            "disruption": self._present(incident.disruption).model_dump(mode="json"),
            "ignored": incident.closed_as == "ignored",
            "closed_reason": incident.closed_reason,
            "published": disruption_id in self._published,
            "summary_by": summary.by if summary else "template",
            "layer": self.layer_of(incident.disruption),
            "options": self.config.options_for(incident.disruption.type),
            "linked": self._linked(disruption_id),
        }

    def layer_of(self, d: Disruption) -> str:
        """ "india" for the domestic leg (pack house, road, Indian port, mandi), else "global"."""
        place = d.location
        indian_port = LOCODE_RE.match(place) is not None and place.startswith("IN")
        domestic = d.type in DOMESTIC_TYPES or place in self._indian_places or indian_port
        return "india" if domestic else "global"

    def _linked(self, disruption_id: str) -> list[str]:
        """Other disruptions (not ignored) that affect at least one of the same containers."""
        mine = set(self.tracker.get(disruption_id).disruption.affected_container_ids)
        return [
            i.disruption_id
            for i in self.tracker.incidents
            if i.disruption_id != disruption_id
            and i.closed_as != "ignored"
            and mine & set(i.disruption.affected_container_ids)
        ]

    def subscribe(self, listener: SentinelListener) -> Callable[[], None]:
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener)

    # --- internals ------------------------------------------------------------------------------

    def _settle(self, decisions: list[Decision], candidates: Iterable[str]) -> list[Disruption]:
        for decision in decisions:
            self._record(decision)
        published = self._publish_changes(dict.fromkeys(candidates))
        touched = [d.disruption_id for d in decisions if d.disruption_id]
        ids = list(dict.fromkeys([*touched, *published]))
        states = {i: self.describe(i) for i in ids}
        # Disruptions linked to a changed one get a fresh "linked" list too.
        for linked_id in [lid for state in list(states.values()) for lid in state["linked"]]:
            states.setdefault(linked_id, self.describe(linked_id))
        for state in states.values():
            self._emit("disruption", state)
        return [self._present(self.tracker.get(i).disruption) for i in ids]

    def _publish_changes(self, disruption_ids: Iterable[str]) -> list[str]:
        published = []
        for disruption_id in disruption_ids:
            view = self.tracker.get(disruption_id).disruption
            if view.status == "candidate":
                continue  # unconfirmed (or ignored): never on the bus
            before = self._published.get(disruption_id)
            if before is None:
                topic = "disruption.detected"
                changed = list(CONTRACT_FIELDS)
            else:
                if not material_changes(before, view, self.config):
                    continue
                topic = "disruption.updated"
                changed = changed_fields(before, view)
            if view.status == "resolved":
                topic = "disruption.resolved"
            view = self._present(view, write=True)
            message = BusMessage(topic=topic, disruption=view, changed_fields=changed)
            payload = message.model_dump(mode="json")
            self.bus.publish(topic, payload)
            self._published[disruption_id] = view
            self._emit("publish", payload)
            published.append(disruption_id)
        return published

    def _present(self, view: Disruption, *, write: bool = False) -> Disruption:
        """The view with its LLM summary, if one exists; write=True asks the LLM when missing."""
        key = _summary_key(view)
        summary = self._summaries.get(key)
        if summary is None and write:
            summary = self._summaries[key] = self.summariser.summarise(view)
        return view if summary is None else view.model_copy(update={"summary": summary.text})

    def _record(self, decision: Decision) -> None:
        details = {**decision.details, "simulated": True}
        self.audit.log(AUDIT_ACTOR, decision.action, details, decision.rationale)
        self._emit(
            "decision",
            {
                "action": decision.action,
                "rationale": decision.rationale,
                "disruption_id": decision.disruption_id,
                "event_id": decision.event_id,
                "details": details,
            },
        )

    def _reject(self, event: RawEvent | Mapping[str, Any], exc: NormalisationError) -> None:
        ids = _loose_ids(event)
        event_id = exc.event_id or ids.get("event_id") or "event"
        rationale = f"{event_id} rejected – {exc.reason}"
        details = {**ids, "at": self.clock.now().isoformat(), "simulated": True}
        self._emit("signal", {**ids, "status": "rejected", "reason": exc.reason, "simulated": True})
        self.audit.log(AUDIT_ACTOR, "reject", details, rationale)
        self._emit(
            "decision",
            {
                "action": "reject",
                "rationale": rationale,
                "disruption_id": None,
                "event_id": ids.get("event_id"),
                "details": details,
            },
        )

    def _signal_info(self, signal: Signal, decisions: list[Decision]) -> dict[str, Any]:
        joined = next((d.disruption_id for d in decisions if d.disruption_id), None)
        ignored = joined is None and all(d.action == "ignore" for d in decisions)
        return {
            "event_id": signal.event_id,
            "source": signal.source,
            "received_at": signal.received_at.isoformat(),
            "kind_hint": signal.kind_hint,
            "title": _title(signal),
            "role": signal.role,
            "type": signal.type,
            "location": signal.location,
            "classified_by": signal.classified_by,
            "reliability": round(scoring.reliability(signal, self.config), 2),
            "reason": signal.reason,
            "status": "ignored" if ignored else "accepted",
            "status_reason": decisions[0].rationale if ignored and decisions else "",
            "disruption_id": joined,
            "simulated": True,
        }

    def _emit(self, kind: str, data: dict[str, Any]) -> None:
        for listener in list(self._listeners):
            listener(kind, data)
