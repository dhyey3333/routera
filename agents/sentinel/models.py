"""Shared contract for the Disruption Sentinel (spec sections 2 and 4, interfaces on page 6).

This module is the single source of truth the orchestrator and the other agents import
(Disruption, BusMessage, ContainerRef, RawEvent). Do not change a field without telling the team.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal, Protocol, get_args

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)


def as_utc(value: datetime) -> datetime:
    """Treat naive timestamps as UTC and convert aware ones to UTC, so comparisons never fail."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _upper(value: Any) -> Any:
    return value.strip().upper() if isinstance(value, str) else value


def _lower(value: Any) -> Any:
    return value.strip().lower() if isinstance(value, str) else value


def _hs_digits(value: Any) -> Any:
    return value.replace(".", "").replace(" ", "") if isinstance(value, str) else value


UtcDatetime = Annotated[datetime, AfterValidator(as_utc)]
Code = Annotated[str, BeforeValidator(_upper), Field(min_length=1)]
CountryCode = Annotated[str, BeforeValidator(_upper), Field(pattern=r"^[A-Z]{2}$")]
HsCode = Annotated[str, BeforeValidator(_hs_digits), Field(pattern=r"^\d{2,10}$")]

SourceName = Literal[
    "port_authority",
    "carrier_advisory",
    "ais",
    "ais_port",  # port status estimated from live AIS anchorage counts (ROUTERA live mode)
    "tariff_feed",
    "weather",
    "gdacs",  # UN/EU official disaster alerts near the network (ROUTERA live mode)
    "tomtom",  # live road closures on the corridor from TomTom Traffic (ROUTERA live mode)
    "ops_port",  # a named person on the ops desk treats a report as a real port closure
    "ops_road",  # the same for a road
    "news",
    # India-inside feeds (v2)
    "road_authority",
    "transport_union",
    "reefer_telemetry",
    "facility_status",
    "compliance_feed",
    "mandi_prices",
]
DisruptionType = Literal[
    "PORT_CLOSURE",
    "PORT_CONGESTION",
    "LANE_SUSPENSION",
    "ROUTE_REROUTE",
    "VESSEL_DELAY",
    "TARIFF_CHANGE",
    "WEATHER_ALERT",
    # India-inside disruptions (v2): farm -> pack house -> pre-cool -> truck -> port
    "ROAD_CLOSURE",
    "TRUCK_STRIKE",
    "REEFER_FAILURE",
    "FACILITY_OUTAGE",
    "COMPLIANCE_HOLD",
    "MARKET_PRICE_CRASH",
]
DisruptionStatus = Literal["candidate", "confirmed", "active", "resolving", "resolved"]
Priority = Literal["normal", "urgent"]
Topic = Literal["disruption.detected", "disruption.updated", "disruption.resolved"]
AuditAction = Literal["create", "merge", "escalate", "ignore", "resolve", "reject"]

SOURCES: tuple[str, ...] = get_args(SourceName)
DISRUPTION_TYPES: tuple[str, ...] = get_args(DisruptionType)
AUDIT_ACTOR = "disruption_sentinel"

# Disruptions that only happen on the domestic leg. PORT_CONGESTION and WEATHER_ALERT are
# domestic too when they hit an Indian port or region (see DisruptionSentinel.describe).
DOMESTIC_TYPES = frozenset(
    {
        "ROAD_CLOSURE",
        "TRUCK_STRIKE",
        "REEFER_FAILURE",
        "FACILITY_OUTAGE",
        "COMPLIANCE_HOLD",
        "MARKET_PRICE_CRASH",
    }
)
# The domestic leg, in order. ContainerRef.status takes one of these (or "packed", the v1 name for
# a stuffed box waiting to move, or "delivered" / "cancelled").
LEGS = ("harvested", "at_packhouse", "precooling", "in_cold_store", "on_truck", "at_port", "at_sea")


# --- 2.1 / 2.2 Inputs -------------------------------------------------------------------------


class RawEvent(BaseModel):
    """One message from a (simulated) feed."""

    model_config = ConfigDict(extra="ignore")

    event_id: str = Field(min_length=1)
    source: Annotated[SourceName, BeforeValidator(_lower)]
    received_at: UtcDatetime
    kind_hint: str | None = None
    payload: dict[str, Any]


class _Payload(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


class PortAuthorityPayload(_Payload):
    port_code: Code
    status: Annotated[Literal["open", "congested", "closed"], BeforeValidator(_lower)]
    reason: str = ""
    effective_from: UtcDatetime
    expected_reopen: UtcDatetime | None = None


class CarrierAdvisoryPayload(_Payload):
    carrier: str = Field(min_length=1)
    affected_lane: str = Field(min_length=1)
    action: Annotated[Literal["suspend", "reroute", "delay"], BeforeValidator(_lower)]
    via: str | None = None
    added_transit_days: float = Field(default=0, ge=0)
    vessel_ids: list[Code] = Field(default_factory=list)


class AisPayload(_Payload):
    vessel_id: Code
    expected_eta: UtcDatetime
    new_eta: UtcDatetime
    position: dict[str, float] | str | None = None
    speed_knots: float | None = Field(default=None, ge=0)
    deviation_flag: bool = False


class TariffPayload(_Payload):
    importing_country: CountryCode
    hs_code: HsCode
    old_rate_pct: float = Field(ge=0)
    new_rate_pct: float = Field(ge=0)
    effective_from: UtcDatetime


class WeatherPayload(_Payload):
    region: str | None = None
    port_code: Code | None = None
    alert_level: Annotated[Literal["yellow", "orange", "red"], BeforeValidator(_lower)]
    hazard: str = Field(min_length=1)
    valid_from: UtcDatetime
    valid_to: UtcDatetime | None = None

    @model_validator(mode="after")
    def _needs_place_and_sane_window(self) -> WeatherPayload:
        if not (self.region or self.port_code):
            raise ValueError("weather alert needs a region or a port_code")
        if self.valid_to is not None and self.valid_to < self.valid_from:
            raise ValueError("valid_to is before valid_from")
        return self


class NewsPayload(_Payload):
    headline: str = Field(min_length=1)
    body: str = ""
    outlet: str = "unknown outlet"


# --- India-inside feeds (v2) --------------------------------------------------------------------


class RoadAuthorityPayload(_Payload):
    """Highway / traffic police / NHAI status for a road segment or corridor."""

    segment: Code  # e.g. KASARA_GHAT, or a corridor id like NSK-JNPT
    status: Annotated[Literal["open", "restricted", "closed"], BeforeValidator(_lower)]
    reason: str = ""
    effective_from: UtcDatetime
    expected_reopen: UtcDatetime | None = None


class TransportUnionPayload(_Payload):
    """A transporters' association notice (strike, go-slow, call-off)."""

    union: Code
    region: str = Field(min_length=1)
    action: Annotated[Literal["strike", "go_slow", "called_off"], BeforeValidator(_lower)]
    effective_from: UtcDatetime
    expected_end: UtcDatetime | None = None
    carrier_ids: list[Code] = Field(default_factory=list)  # empty: every member of the union


class ReeferTelemetryPayload(_Payload):
    """A reefer truck's temperature log."""

    truck_id: Code
    setpoint_c: float
    temp_c: float
    minutes_above: float = Field(default=0, ge=0)  # how long it has been above the setpoint
    status: Annotated[Literal["ok", "excursion", "breakdown"], BeforeValidator(_lower)]


class FacilityStatusPayload(_Payload):
    """Pack house, pre-cooler or cold store status."""

    facility_id: Code
    status: Annotated[Literal["ok", "power_failure", "full"], BeforeValidator(_lower)]
    free_capacity_t: float | None = Field(default=None, ge=0)
    expected_restore: UtcDatetime | None = None


class ComplianceFeedPayload(_Payload):
    """Customs, phytosanitary inspection or residue-lab status."""

    authority: str = Field(min_length=1)
    facility_id: Code | None = None
    port_code: Code | None = None
    market_country: CountryCode | None = None  # None: any export market
    status: Annotated[Literal["delayed", "held", "failed", "cleared"], BeforeValidator(_lower)]
    reason: str = ""
    expected_clear: UtcDatetime | None = None

    @model_validator(mode="after")
    def _needs_a_place(self) -> ComplianceFeedPayload:
        if not (self.facility_id or self.port_code):
            raise ValueError("compliance status needs a facility_id or a port_code")
        return self


class MandiPricePayload(_Payload):
    """A domestic wholesale market price (Agmarknet-style)."""

    market_id: str = Field(min_length=1)
    commodity: str = Field(min_length=1)
    modal_price_inr_per_kg: float = Field(gt=0)
    baseline_price_inr_per_kg: float = Field(gt=0)  # e.g. the 30-day average
    arrivals_t: float | None = Field(default=None, ge=0)


AnyPayload = (
    PortAuthorityPayload
    | CarrierAdvisoryPayload
    | AisPayload
    | TariffPayload
    | WeatherPayload
    | NewsPayload
    | RoadAuthorityPayload
    | TransportUnionPayload
    | ReeferTelemetryPayload
    | FacilityStatusPayload
    | ComplianceFeedPayload
    | MandiPricePayload
)

PAYLOAD_MODELS: dict[str, type[_Payload]] = {
    "port_authority": PortAuthorityPayload,
    "carrier_advisory": CarrierAdvisoryPayload,
    "ais": AisPayload,
    "ais_port": PortAuthorityPayload,
    "tariff_feed": TariffPayload,
    "weather": WeatherPayload,
    "gdacs": WeatherPayload,
    "news": NewsPayload,
    "road_authority": RoadAuthorityPayload,
    "tomtom": RoadAuthorityPayload,
    "ops_port": PortAuthorityPayload,
    "ops_road": RoadAuthorityPayload,
    "transport_union": TransportUnionPayload,
    "reefer_telemetry": ReeferTelemetryPayload,
    "facility_status": FacilityStatusPayload,
    "compliance_feed": ComplianceFeedPayload,
    "mandi_prices": MandiPricePayload,
}


class ContainerRef(BaseModel):
    """Container data provided by teammates (read-only). days_left is from the Shelf-Life Tracker.

    route_legs are the road segments, corridors, ports and chokepoints still ahead of the
    container; origin_port is not matched. status is the leg it is on (LEGS). A domestic load for
    an Indian market uses the market id as destination_port and "IN" as destination_country.
    """

    model_config = ConfigDict(extra="ignore")

    container_id: str = Field(min_length=1)
    crop: str
    hs_code: HsCode
    weight_kg: float = Field(ge=0)
    vessel_id: Code | None = None
    origin_port: Code
    destination_port: Code
    destination_country: CountryCode
    route_legs: list[Code] = Field(default_factory=list)
    eta: UtcDatetime
    status: Annotated[str, BeforeValidator(_lower)]
    days_left: float | None = None
    # Domestic leg (v2, optional)
    facility_id: Code | None = None  # pack house / pre-cooler / cold store holding it now
    carrier_id: Code | None = None  # road or rail carrier for the leg to the port or market
    truck_id: Code | None = None
    market_id: Code | None = None  # domestic market for a domestic load
    cutoff_at: UtcDatetime | None = None  # port gate-in cut-off for its booked vessel


# --- Reference network (v2): corridors, carriers, facilities, markets ---------------------------
# Illustrative data in data/network.json; swap for the team's master data.


class Corridor(BaseModel):
    corridor_id: Code
    name: str
    mode: Literal["road", "rail"]
    region: Code  # where it starts
    port: Code
    segments: list[Code] = Field(default_factory=list)
    km: float = Field(ge=0)
    transit_hours: float = Field(ge=0)


class Carrier(BaseModel):
    carrier_id: Code
    name: str
    mode: Literal["road", "rail"]
    union: Code | None = None
    region: Code


class Facility(BaseModel):
    facility_id: Code
    name: str
    kind: Literal["packhouse", "precool", "cold_store"]
    region: Code
    capacity_t: float = Field(ge=0)
    cost_inr_per_t_day: float | None = Field(default=None, ge=0)


class Market(BaseModel):
    market_id: Code
    name: str
    city: str


class Network(BaseModel):
    simulated: Literal[True]
    note: str
    corridors: list[Corridor]
    carriers: list[Carrier]
    facilities: list[Facility]
    markets: list[Market]

    def codes(self) -> set[str]:
        """Every code a feed may name: corridors, road segments, facilities, markets, carriers."""
        out = {c.corridor_id for c in self.corridors} | {f.facility_id for f in self.facilities}
        out |= {m.market_id for m in self.markets} | {c.carrier_id for c in self.carriers}
        out |= {s for c in self.corridors for s in c.segments}
        return out

    def facility(self, facility_id: str) -> Facility | None:
        return next((f for f in self.facilities if f.facility_id == facility_id), None)

    def union_members(self, union: str) -> set[str]:
        return {c.carrier_id for c in self.carriers if c.union == union}

    def corridors_through(self, segment: str) -> set[str]:
        return {c.corridor_id for c in self.corridors if segment in c.segments}


# --- Internal: a normalised, classified event ---------------------------------------------------

SignalRole = Literal["disruption", "resolution", "noise"]
ClassifiedBy = Literal["rules", "keywords", "llm", "none"]


class SourceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    source: SourceName
    received_at: UtcDatetime


class Signal(BaseModel):
    """A validated RawEvent with canonical locations plus its classification. Never on the bus.

    location  – canonical primary location (port, chokepoint, region, lane, vessel or "US:0306")
    footprint – every canonical place the signal touches (ports, chokepoints, lane, region members);
                merge.py uses it to relate signals that do not share an exact fingerprint
    role      – "disruption" (has a type), "resolution" (reopen / rate reversal / back on schedule)
                or "noise" (below threshold or not a disruption)
    """

    event_id: str
    source: SourceName
    received_at: UtcDatetime
    kind_hint: str | None = None
    payload: AnyPayload
    location: str = ""
    footprint: list[str] = Field(default_factory=list)
    vessel_ids: list[str] = Field(default_factory=list)
    role: SignalRole = "noise"
    type: DisruptionType | None = None
    scope: str = ""
    type_confidence: float = Field(default=1.0, ge=0, le=1)
    classified_by: ClassifiedBy = "none"
    reason: str = ""

    @property
    def fingerprint(self) -> tuple[str, str, str] | None:
        """(type, location, lane / country+hs), spec section 3 step 3; None for non-disruptions."""
        return None if self.type is None else (self.type, self.location, self.scope)

    def source_ref(self) -> SourceRef:
        return SourceRef(event_id=self.event_id, source=self.source, received_at=self.received_at)


# --- 4. Output contract (shared with the team) --------------------------------------------------


class Disruption(BaseModel):
    """A trustworthy disruption. Published inside BusMessage on disruption.* topics.

    location is a port code, chokepoint, lane ("INNSA-AEJEA"), vessel_id or country:hs ("US:0806");
    for India-inside types also a road segment (KASARA_GHAT), region (NASHIK), truck, facility
    (CS-NSK) or market (MKT-VASHI).
    """

    model_config = ConfigDict(extra="forbid")

    disruption_id: str = Field(pattern=r"^DSR-\d{4,}$")
    type: DisruptionType
    status: DisruptionStatus
    location: str = Field(min_length=1)
    first_seen: UtcDatetime
    last_updated: UtcDatetime
    expected_delay_days: float = Field(ge=0)
    delay_is_estimate: bool
    cost_impact_pct: float
    market_unreachable: bool
    severity: int = Field(ge=1, le=5)
    priority: Priority
    confidence: float = Field(ge=0, le=1)
    sources: list[SourceRef]
    affected_container_ids: list[str]
    affected_weight_kg: float = Field(ge=0)
    summary: str
    rationale: list[str]
    simulated: Literal[True] = True


class BusMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: Topic
    disruption: Disruption
    changed_fields: list[str]


# --- Interfaces the Sentinel depends on (inject them; fakes.py ships in-memory versions) --------


class ContainerRepository(Protocol):
    def list_active(self) -> list[ContainerRef]: ...


class EventBus(Protocol):
    def publish(self, topic: str, message: dict[str, Any]) -> None: ...


class AuditSink(Protocol):
    def log(self, actor: str, action: str, details: dict[str, Any], rationale: str) -> None: ...


class LLMClient(Protocol):
    """Optional. None means offline mode: keyword classification and template summaries."""

    def classify_news(self, headline: str, body: str) -> dict[str, Any]: ...

    def summarise(self, disruption: dict[str, Any]) -> str: ...


class Clock(Protocol):
    def now(self) -> datetime: ...


class NetworkRepository(Protocol):
    def network(self) -> Network: ...
