"""Pipeline steps 3 and 5 (spec section 3): dedupe/merge signals into incidents, and the lifecycle.

An Incident collects every signal that describes one real-world event. After each change, and on
each clock tick, it is re-assessed into a Disruption with scoring.py and matching.py.

A signal joins an open incident whose last signal is under 72 h old when (strongest rule first):
  0. they share a fingerprint (type, location, lane / country+hs);
  1. a structured port closure / congestion / weather signal hits the incident's port, or the
     destination of its delayed or suspended lane (how a lane delay escalates to a closure);
  2. a vessel delay names a vessel the incident already covers (a supporting signal);
  3. a news report describes the same kind of event at an overlapping place;
  4. a structured signal lands inside the place of a news-only candidate (corroboration).
It joins one incident only: the strongest rule wins, then the oldest incident.

Lifecycle: candidate -> confirmed (conf >= 0.6 or an authoritative source, and never news alone)
-> active (an official start time has passed) -> resolving (reopen / rate reversal / on-time
signal, or a weather window ending) -> resolved (no adverse signal for 12 h). Candidates that
stay unconfirmed for 24 h are dropped as "ignored – uncorroborated".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal

from agents.sentinel import matching, scoring
from agents.sentinel.models import (
    AuditAction,
    CarrierAdvisoryPayload,
    Clock,
    ComplianceFeedPayload,
    ContainerRepository,
    Disruption,
    DisruptionStatus,
    FacilityStatusPayload,
    MandiPricePayload,
    Network,
    PortAuthorityPayload,
    ReeferTelemetryPayload,
    RoadAuthorityPayload,
    Signal,
    TariffPayload,
    TransportUnionPayload,
    WeatherPayload,
)
from agents.sentinel.normalise import Gazetteer
from agents.sentinel.settings import SentinelConfig
from agents.sentinel.summary import template_summary

STATUS_LEVEL: dict[str, int] = {
    "resolved": -1,
    "candidate": 0,
    "resolving": 1,
    "confirmed": 2,
    "active": 3,
}
PORT_FAMILY = frozenset({"PORT_CLOSURE", "PORT_CONGESTION", "WEATHER_ALERT"})
LANE_ACTIONS = frozenset({"suspend", "delay"})
# Which disruption types a resolution signal from each source can resolve.
RESOLVES: dict[str, frozenset[str]] = {
    "port_authority": frozenset({"PORT_CLOSURE", "PORT_CONGESTION"}),
    "tariff_feed": frozenset({"TARIFF_CHANGE"}),
    "ais": frozenset({"VESSEL_DELAY"}),
    "road_authority": frozenset({"ROAD_CLOSURE"}),
    "tomtom": frozenset({"ROAD_CLOSURE"}),
    "transport_union": frozenset({"TRUCK_STRIKE"}),
    "reefer_telemetry": frozenset({"REEFER_FAILURE"}),
    "facility_status": frozenset({"FACILITY_OUTAGE"}),
    "compliance_feed": frozenset({"COMPLIANCE_HOLD"}),
    "mandi_prices": frozenset({"MARKET_PRICE_CRASH"}),
}
# Feeds that report a state as it is now: in effect from the moment they arrive.
_IN_EFFECT_ON_ARRIVAL = (
    ReeferTelemetryPayload | FacilityStatusPayload | ComplianceFeedPayload | MandiPricePayload
)


def _stamp(when: datetime) -> str:
    return f"{when:%Y-%m-%d %H:%M}Z"


def _lane_destination(signal: Signal) -> str | None:
    p = signal.payload
    if isinstance(p, CarrierAdvisoryPayload) and p.action in LANE_ACTIONS:
        return p.affected_lane.split("-", 1)[1]
    return None


def _effective_from(signal: Signal) -> datetime | None:
    """The official start time a structured source states, if any."""
    p = signal.payload
    if isinstance(p, PortAuthorityPayload | TariffPayload | RoadAuthorityPayload):
        return p.effective_from
    if isinstance(p, TransportUnionPayload):
        return p.effective_from
    if isinstance(p, WeatherPayload):
        return p.valid_from
    if isinstance(p, _IN_EFFECT_ON_ARRIVAL):
        return signal.received_at
    return None


@dataclass(frozen=True)
class Decision:
    """One Sentinel decision; the Sentinel writes each to the audit sink."""

    action: AuditAction
    rationale: str
    disruption_id: str | None = None
    event_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class Incident:
    disruption_id: str
    first_seen: datetime
    last_signal_at: datetime
    signals: list[Signal] = field(default_factory=list)  # adverse and supporting signals
    resolutions: list[Signal] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # why each signal joined
    resolving_since: datetime | None = None
    closed_as: Literal["resolved", "ignored"] | None = None
    closed_reason: str = ""
    status_reason: str = ""
    affected_vessels: frozenset[str] = frozenset()
    view: Disruption | None = None

    @property
    def is_open(self) -> bool:
        return self.closed_as is None

    @property
    def news_only(self) -> bool:
        return all(s.source == "news" for s in self.signals)

    @property
    def disruption(self) -> Disruption:
        if self.view is None:
            raise RuntimeError(f"{self.disruption_id} has not been assessed")
        return self.view

    def footprint(self) -> set[str]:
        return {place for s in self.signals for place in s.footprint}

    def vessels(self) -> set[str]:
        return {v for s in self.signals for v in s.vessel_ids} | self.affected_vessels

    def lane_destinations(self) -> set[str]:
        return {d for s in self.signals if (d := _lane_destination(s)) is not None}


class DisruptionTracker:
    """Owns the incidents. add() and tick() return the decisions to audit."""

    def __init__(
        self,
        config: SentinelConfig,
        gazetteer: Gazetteer,
        containers: ContainerRepository,
        clock: Clock,
        network: Network | None = None,
    ) -> None:
        self.config = config
        self.gazetteer = gazetteer
        self.containers = containers
        self.clock = clock
        self.network = network
        self._incidents: dict[str, Incident] = {}
        self._seen_events: set[str] = set()
        self._rank = {dtype: i for i, dtype in enumerate(config.type_rank)}

    @property
    def incidents(self) -> list[Incident]:
        return list(self._incidents.values())

    def get(self, disruption_id: str) -> Incident:
        return self._incidents[disruption_id]

    def _open(self) -> list[Incident]:
        return [i for i in self._incidents.values() if i.is_open]

    # --- ingest ---------------------------------------------------------------------------------

    def add(self, signal: Signal) -> list[Decision]:
        if signal.event_id in self._seen_events:
            return [
                self._decision(
                    "ignore", f"{signal.event_id} ignored – duplicate event_id", signal=signal
                )
            ]
        self._seen_events.add(signal.event_id)
        if signal.role == "noise":
            return [
                self._decision(
                    "ignore", f"{signal.event_id} ignored – {signal.reason}", signal=signal
                )
            ]
        if signal.role == "resolution":
            return self._apply_resolution(signal)
        related = self._find_related(signal)
        if related is None:
            return self._create(signal)
        incident, why = related
        return self._merge(incident, signal, why)

    def tick(self) -> list[Decision]:
        """Clock-driven lifecycle: expiry, resolution, and start times passing."""
        now = self.clock.now()
        life = self.config.lifecycle
        decisions: list[Decision] = []
        for incident in self._open():
            before = incident.disruption
            if before.status == "candidate":
                if now - incident.first_seen >= timedelta(hours=life.candidate_ttl_hours):
                    reason = (
                        f"ignored – uncorroborated: no structured confirmation within "
                        f"{life.candidate_ttl_hours:g} h of the first report"
                    )
                    decisions.append(self._close(incident, "ignored", reason, "ignore"))
                continue
            if incident.resolving_since is not None:
                if now - incident.resolving_since >= timedelta(hours=life.resolve_after_hours):
                    reason = f"resolved: no adverse signal for {life.resolve_after_hours:g} h"
                    decisions.append(self._close(incident, "resolved", reason, "resolve"))
                continue
            if self._alert_window_over(incident, now):
                incident.resolving_since = now
                self._assess(incident)
                decisions.append(
                    self._decision(
                        "resolve",
                        f"{incident.disruption_id} -> resolving: weather alert window ended",
                        incident,
                    )
                )
                continue
            after = self._assess(incident)
            escalation = self._escalation(before, after, incident)
            if escalation:
                decisions.append(
                    self._decision("escalate", f"{incident.disruption_id} {escalation}", incident)
                )
        return decisions

    # --- create / merge -------------------------------------------------------------------------

    def _create(self, signal: Signal) -> list[Decision]:
        disruption_id = f"DSR-{len(self._incidents) + 1:04d}"
        incident = Incident(
            disruption_id,
            first_seen=signal.received_at,
            last_signal_at=signal.received_at,
            signals=[signal],
            notes=[f"{signal.event_id} ({signal.source}) opened it: {signal.reason}"],
        )
        self._incidents[disruption_id] = incident
        view = self._assess(incident)
        opened = f"{disruption_id} {view.type} at {view.location} from {signal.event_id}"
        if view.status == "candidate":
            rationale = (
                f"new candidate {opened}: {incident.status_reason}; "
                "not published until corroborated"
            )
        else:
            rationale = (
                f"new {view.status} {opened}: {incident.status_reason}; "
                f"severity {view.severity}/5, "
                f"{len(view.affected_container_ids)} containers"
            )
        return [self._decision("create", rationale, incident, signal)]

    def _merge(self, incident: Incident, signal: Signal, why: str) -> list[Decision]:
        before = incident.disruption
        was_news_only = incident.news_only
        assert signal.type is not None
        incident.signals.append(signal)
        incident.last_signal_at = max(incident.last_signal_at, signal.received_at)
        note = f"{signal.event_id} ({signal.source}) merged: {why}"
        incident.notes.append(note)
        if (
            incident.resolving_since is not None
            and self._rank[signal.type] <= self._rank[before.type]
        ):
            incident.resolving_since = None  # the disruption is back: stop resolving
        after = self._assess(incident)

        changes = [f"confidence {before.confidence:.3f} -> {after.confidence:.3f}"]
        if after.type != before.type:
            basis = "structured source outranks the news guess" if was_news_only else "higher rank"
            changes.append(f"type {before.type} -> {after.type} ({basis})")
        decisions = [
            self._decision(
                "merge", f"{incident.disruption_id}: {note}; {'; '.join(changes)}", incident, signal
            )
        ]
        escalation = self._escalation(before, after, incident)
        if escalation:
            decisions.append(
                self._decision(
                    "escalate", f"{incident.disruption_id} {escalation}", incident, signal
                )
            )
        return decisions

    def _find_related(self, signal: Signal) -> tuple[Incident, str] | None:
        window = timedelta(hours=self.config.lifecycle.merge_window_hours)
        best: tuple[int, int, Incident, str] | None = None
        for order, incident in enumerate(self._open()):
            if signal.received_at - incident.last_signal_at > window:
                continue
            relation = self._relation(incident, signal)
            if relation is None:
                continue
            strength, why = relation
            if best is None or (strength, order) < best[:2]:
                best = (strength, order, incident, why)
        return None if best is None else (best[2], best[3])

    def _relation(self, incident: Incident, signal: Signal) -> tuple[int, str] | None:
        if signal.fingerprint in {s.fingerprint for s in incident.signals}:
            return 0, f"same fingerprint ({signal.type}, {signal.location})"
        view = incident.disruption
        structured = signal.source != "news"

        if structured and signal.type in PORT_FAMILY:
            port = signal.location
            if view.type in PORT_FAMILY and view.location == port:
                return 1, f"{signal.type} at {port}, where it is already {view.type}"
            if port in incident.lane_destinations():
                return 1, f"{signal.type} at {port}, the destination of the delayed lane"
        lane_end = _lane_destination(signal)
        if structured and lane_end and view.type in PORT_FAMILY and view.location == lane_end:
            return 1, f"lane {signal.location} ends at {lane_end}, already {view.type}"

        if signal.type == "VESSEL_DELAY":
            shared_vessels = sorted(set(signal.vessel_ids) & incident.vessels())
            if shared_vessels:
                return 2, f"supporting signal: {', '.join(shared_vessels)} is already covered"

        shared = sorted(set(signal.footprint) & incident.footprint())
        if not shared:
            return None
        if signal.location in shared:  # name the place the signal is about, not an alphabetic first
            shared.insert(0, signal.location)
        if not self._same_kind(signal, incident):
            return None  # e.g. a rain alert never corroborates a strike rumour
        if not structured:
            return 3, f"another report of {signal.type} at {shared[0]}"
        if incident.news_only:
            return (
                4,
                f"{signal.source} corroborates the news-only candidate (shared place {shared[0]})",
            )
        return None

    def _same_kind(self, signal: Signal, incident: Incident) -> bool:
        """Same type, same family (config `families`), or both port-level (closure, congestion,
        weather at a port)."""
        assert signal.type is not None
        types = {s.type for s in incident.signals if s.type is not None}
        family = self.config.family_of(signal.type)
        return (
            signal.type in types
            or any(self.config.family_of(t) == family for t in types)
            or (signal.type in PORT_FAMILY and bool(types & PORT_FAMILY))
        )

    # --- resolution -----------------------------------------------------------------------------

    def _apply_resolution(self, signal: Signal) -> list[Decision]:
        resolvable = RESOLVES.get(signal.source, frozenset())
        targets = [
            i
            for i in self._open()
            if i.disruption.type in resolvable and i.disruption.location == signal.location
        ]
        if not targets:
            return [
                self._decision(
                    "ignore",
                    f"{signal.event_id} ignored – {signal.reason}; nothing open to resolve",
                    signal=signal,
                )
            ]
        decisions = []
        for incident in targets:
            incident.resolutions.append(signal)
            incident.notes.append(f"{signal.event_id} ({signal.source}): {signal.reason}")
            if incident.disruption.status == "candidate":
                reason = f"ignored – refuted: {signal.reason}"
                decisions.append(self._close(incident, "ignored", reason, "ignore", signal))
                continue
            if incident.resolving_since is None:
                incident.resolving_since = self.clock.now()
            self._assess(incident)
            hours = self.config.lifecycle.resolve_after_hours
            decisions.append(
                self._decision(
                    "resolve",
                    f"{incident.disruption_id} {incident.disruption.type} at "
                    f"{incident.disruption.location} -> resolving: {signal.reason}; resolved after "
                    f"{hours:g} h without new adverse signals",
                    incident,
                    signal,
                )
            )
        return decisions

    def _close(
        self,
        incident: Incident,
        closed_as: Literal["resolved", "ignored"],
        reason: str,
        action: AuditAction,
        signal: Signal | None = None,
    ) -> Decision:
        incident.closed_as = closed_as
        incident.closed_reason = reason
        self._assess(incident)
        return self._decision(action, f"{incident.disruption_id} {reason}", incident, signal)

    def _alert_window_over(self, incident: Incident, now: datetime) -> bool:
        if incident.disruption.type != "WEATHER_ALERT":
            return False
        ends = [
            s.payload.valid_to
            for s in incident.signals
            if isinstance(s.payload, WeatherPayload) and s.type == "WEATHER_ALERT"
        ]
        return bool(ends) and all(end is not None and end <= now for end in ends)

    # --- assessment -----------------------------------------------------------------------------

    def _primary(self, incident: Incident) -> Signal:
        """The type-defining signal: highest-ranked type among structured signals (news if none),
        latest wins on ties."""
        structured = [s for s in incident.signals if s.source != "news"]
        pool = structured or incident.signals
        best = min(self._rank[s.type] for s in pool if s.type is not None)
        return [s for s in pool if s.type is not None and self._rank[s.type] == best][-1]

    def _status(
        self, incident: Incident, confidence: float, now: datetime
    ) -> tuple[DisruptionStatus, str]:
        if incident.closed_as == "resolved":
            return "resolved", incident.closed_reason
        life = self.config.lifecycle
        structured = [s for s in incident.signals if s.source != "news"]
        if not structured:
            return "candidate", "news only, and news never confirms a disruption on its own"
        authoritative = sorted({s.source for s in structured} & set(life.authoritative_sources))
        if not authoritative and confidence < life.confirm_threshold:
            return "candidate", f"confidence {confidence:.2f} < {life.confirm_threshold:g}"
        if authoritative:
            basis = f"confirmed by authoritative {', '.join(authoritative)}"
        else:
            basis = (
                f"confidence {confidence:.2f} >= {life.confirm_threshold:g} "
                "with a structured source"
            )
        if incident.resolving_since is not None:
            return "resolving", f"{basis}; resolving since {_stamp(incident.resolving_since)}"
        started = [t for s in structured if (t := _effective_from(s)) is not None and t <= now]
        if started:
            return "active", f"{basis}; in effect since {_stamp(min(started))}"
        return "confirmed", basis

    def _assess(self, incident: Incident) -> Disruption:
        now = self.clock.now()
        cfg = self.config
        primary = self._primary(incident)
        assert primary.type is not None
        conf = scoring.confidence(incident.signals, cfg)
        status, status_reason = self._status(incident, conf.value, now)
        delay = scoring.expected_delay(primary, now, cfg)
        target = matching.target_for(primary, self.gazetteer, self.network)

        if status == "candidate":
            containers = []
            severity, priority = 1, "normal"
            severity_reason = "severity held at 1 while unconfirmed"
            match_reason = "affected containers are listed once confirmed"
        else:
            containers = matching.match_containers(target, self.containers.list_active(), cfg)
            delay = scoring.with_missed_sailings(delay, primary.type, containers, now, cfg)
            score = scoring.severity(delay, containers, cfg)
            severity, priority, severity_reason = score.level, score.priority, score.reason
            match_reason = f"{len(containers)} containers: {matching.describe(target, cfg)}"

        rationale = [
            f"type {primary.type}: {primary.reason} [{primary.event_id}]",
            f"status {status}: {status_reason}",
            conf.reason,
            delay.reason,
            severity_reason,
            match_reason,
            *incident.notes,
        ]
        if incident.closed_reason:
            rationale.append(incident.closed_reason)
        sources = sorted(
            (s.source_ref() for s in [*incident.signals, *incident.resolutions]),
            key=lambda ref: ref.received_at,
        )
        draft = Disruption(
            disruption_id=incident.disruption_id,
            type=primary.type,
            status=status,
            location=primary.location,
            first_seen=incident.first_seen,
            last_updated=now,
            expected_delay_days=round(delay.days, 2),
            delay_is_estimate=delay.is_estimate,
            cost_impact_pct=delay.cost_impact_pct,
            market_unreachable=delay.market_unreachable,
            severity=severity,
            priority=priority,
            confidence=round(conf.value, 4),
            sources=sources,
            affected_container_ids=[c.container_id for c in containers],
            affected_weight_kg=sum(c.weight_kg for c in containers),
            summary="",
            rationale=rationale,
        )
        view = draft.model_copy(update={"summary": template_summary(draft)})
        previous = incident.view
        if previous is not None and _same_state(previous, view):
            view = previous  # nothing changed: keep last_updated
        incident.view = view
        incident.status_reason = status_reason
        incident.affected_vessels = frozenset(c.vessel_id for c in containers if c.vessel_id)
        return view

    def _escalation(self, before: Disruption, after: Disruption, incident: Incident) -> str | None:
        parts = []
        if STATUS_LEVEL[after.status] > STATUS_LEVEL[before.status]:
            parts.append(f"status {before.status} -> {after.status} ({incident.status_reason})")
        if after.severity > before.severity:
            priority = f", priority {after.priority}" if after.priority != before.priority else ""
            parts.append(f"severity {before.severity} -> {after.severity}{priority}")
        if after.type != before.type and self._rank[after.type] < self._rank[before.type]:
            parts.append(f"type {before.type} -> {after.type}")
        return "; ".join(parts) or None

    def _decision(
        self,
        action: AuditAction,
        rationale: str,
        incident: Incident | None = None,
        signal: Signal | None = None,
    ) -> Decision:
        details: dict[str, Any] = {"at": self.clock.now().isoformat()}
        if signal is not None:
            details |= {"event_id": signal.event_id, "source": signal.source}
        if incident is not None and incident.view is not None:
            v = incident.view
            details |= {
                "disruption_id": v.disruption_id,
                "type": v.type,
                "location": v.location,
                "status": v.status,
                "severity": v.severity,
                "confidence": v.confidence,
            }
        return Decision(
            action,
            rationale,
            incident.disruption_id if incident else None,
            signal.event_id if signal else None,
            details,
        )


def _same_state(a: Disruption, b: Disruption) -> bool:
    return a.model_dump(exclude={"last_updated"}) == b.model_dump(exclude={"last_updated"})
