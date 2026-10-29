"""Pipeline step 6 (spec section 3.2): which containers a disruption affects.

Matching uses the disruption's primary (type-defining) signal only; supporting signals such as
an AIS slow-down merged into a port closure do not widen the affected set. The India-inside rules
(v2) also use the reference network: a road segment's corridors, a union's carriers, a
facility's region.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime

from agents.sentinel.models import (
    CarrierAdvisoryPayload,
    ContainerRef,
    DisruptionType,
    Network,
    Signal,
    TariffPayload,
    TransportUnionPayload,
)
from agents.sentinel.normalise import Gazetteer
from agents.sentinel.settings import SentinelConfig


@dataclass(frozen=True)
class MatchTarget:
    type: DisruptionType
    location: str
    places: frozenset[str]  # location plus region members / corridors / pack houses, per rule
    lane: tuple[str, str] | None = None
    vessel_ids: frozenset[str] = frozenset()
    country: str | None = None
    hs_prefix: str | None = None
    effective_from: datetime | None = None
    carrier_ids: frozenset[str] = frozenset()  # TRUCK_STRIKE
    scope: str = ""  # FACILITY_OUTAGE: power_failure | full; COMPLIANCE_HOLD: market country


def target_for(
    primary: Signal, gazetteer: Gazetteer, network: Network | None = None
) -> MatchTarget:
    if primary.type is None:
        raise ValueError(f"signal {primary.event_id} has no disruption type")
    p = primary.payload
    places = set(gazetteer.expand([primary.location]))
    lane = None
    if isinstance(p, CarrierAdvisoryPayload):
        origin, destination = p.affected_lane.split("-", 1)
        lane = (origin, destination)
    country = hs_prefix = None
    effective_from = None
    if primary.type == "TARIFF_CHANGE":
        country, _, hs = primary.location.partition(":")
        hs_prefix = hs or None
        if isinstance(p, TariffPayload):
            effective_from = p.effective_from
    carriers: set[str] = set()
    if primary.type == "TRUCK_STRIKE":
        if isinstance(p, TransportUnionPayload) and p.carrier_ids:
            carriers = set(p.carrier_ids)
        elif network is not None and primary.scope:
            carriers = network.union_members(primary.scope)
    if network is not None and primary.type == "ROAD_CLOSURE":
        places |= network.corridors_through(primary.location)
    if network is not None and primary.type == "FACILITY_OUTAGE" and primary.scope == "full":
        full = network.facility(primary.location)
        if full is not None:  # a full cold store strands the boxes waiting at pack houses nearby
            places = {
                f.facility_id
                for f in network.facilities
                if f.region == full.region and f.kind in ("packhouse", "precool")
            }
    return MatchTarget(
        type=primary.type,
        location=primary.location,
        places=frozenset(places),
        lane=lane,
        vessel_ids=frozenset(primary.vessel_ids),
        country=country,
        hs_prefix=hs_prefix,
        effective_from=effective_from,
        carrier_ids=frozenset(carriers),
        scope=primary.scope,
    )


Rule = Callable[[MatchTarget, ContainerRef, SentinelConfig], bool]


def _pre_port(c: ContainerRef, cfg: SentinelConfig) -> bool:
    return c.status in cfg.matching.pre_port_statuses


def _at_place(t: MatchTarget, c: ContainerRef, cfg: SentinelConfig) -> bool:
    in_transit = c.status in cfg.matching.in_transit_statuses
    here = (
        c.destination_port in t.places
        or c.facility_id in t.places
        or not t.places.isdisjoint(c.route_legs)
    )
    return in_transit and here


def _on_lane(t: MatchTarget, c: ContainerRef, cfg: SentinelConfig) -> bool:
    return t.lane == (c.origin_port, c.destination_port)


def _through_chokepoint(t: MatchTarget, c: ContainerRef, cfg: SentinelConfig) -> bool:
    return t.location in c.route_legs


def _on_vessel(t: MatchTarget, c: ContainerRef, cfg: SentinelConfig) -> bool:
    if t.vessel_ids:
        return c.vessel_id in t.vessel_ids
    return t.lane is not None and _on_lane(t, c, cfg)  # advisory delay with no vessels listed


def _tariffed(t: MatchTarget, c: ContainerRef, cfg: SentinelConfig) -> bool:
    return (
        t.country == c.destination_country
        and t.hs_prefix is not None
        and c.hs_code.startswith(t.hs_prefix)
        and (t.effective_from is None or c.eta >= t.effective_from)
    )


def _on_road(t: MatchTarget, c: ContainerRef, cfg: SentinelConfig) -> bool:
    return _pre_port(c, cfg) and not t.places.isdisjoint(c.route_legs)


def _struck(t: MatchTarget, c: ContainerRef, cfg: SentinelConfig) -> bool:
    return _pre_port(c, cfg) and c.carrier_id in t.carrier_ids


def _on_truck(t: MatchTarget, c: ContainerRef, cfg: SentinelConfig) -> bool:
    return c.truck_id == t.location


def _at_facility(t: MatchTarget, c: ContainerRef, cfg: SentinelConfig) -> bool:
    if t.scope == "full":
        return c.status == "at_packhouse" and c.facility_id in t.places
    return c.facility_id == t.location


def _held(t: MatchTarget, c: ContainerRef, cfg: SentinelConfig) -> bool:
    here = c.facility_id == t.location or t.location in c.route_legs
    waiting = _pre_port(c, cfg) or c.status == "at_port"
    market = c.destination_country == t.scope if t.scope else c.destination_country != "IN"
    return here and waiting and market


def _in_market(t: MatchTarget, c: ContainerRef, cfg: SentinelConfig) -> bool:
    return t.location in (c.market_id, c.destination_port)


RULES: dict[str, Rule] = {
    "PORT_CLOSURE": _at_place,
    "PORT_CONGESTION": _at_place,
    "WEATHER_ALERT": _at_place,
    "LANE_SUSPENSION": _on_lane,
    "ROUTE_REROUTE": _through_chokepoint,
    "VESSEL_DELAY": _on_vessel,
    "TARIFF_CHANGE": _tariffed,
    "ROAD_CLOSURE": _on_road,
    "TRUCK_STRIKE": _struck,
    "REEFER_FAILURE": _on_truck,
    "FACILITY_OUTAGE": _at_facility,
    "COMPLIANCE_HOLD": _held,
    "MARKET_PRICE_CRASH": _in_market,
}


def match_containers(
    target: MatchTarget, containers: Iterable[ContainerRef], config: SentinelConfig
) -> list[ContainerRef]:
    rule = RULES[target.type]
    hits = [c for c in containers if rule(target, c, config)]
    return sorted(hits, key=lambda c: c.container_id)


def describe(target: MatchTarget, config: SentinelConfig) -> str:
    """The matching rule in words, for the disruption's rationale."""
    rule = RULES[target.type]
    pre_port = "/".join(config.matching.pre_port_statuses)
    if rule is _at_place:
        where = target.location
        if len(target.places) > 1:
            where += f" ({len(target.places) - 1} member places)"
        statuses = "/".join(config.matching.in_transit_statuses)
        return f"destination or route leg {where}, status {statuses}"
    if rule is _on_lane:
        return f"lane {target.location}"
    if rule is _through_chokepoint:
        return f"route legs include {target.location}"
    if rule is _on_vessel:
        if target.vessel_ids:
            return f"on vessel {', '.join(sorted(target.vessel_ids))}"
        return f"lane {target.location} (no vessels listed)"
    if rule is _on_road:
        return f"road ahead uses {', '.join(sorted(target.places))}, status {pre_port}"
    if rule is _struck:
        carriers = ", ".join(sorted(target.carrier_ids)) or "no known carriers"
        return f"moved by {carriers}, status {pre_port}"
    if rule is _on_truck:
        return f"on truck {target.location}"
    if rule is _at_facility:
        if target.scope == "full":
            return f"waiting at pack houses {', '.join(sorted(target.places))} for cold-store space"
        return f"held at {target.location}"
    if rule is _held:
        market = target.scope or "any export market"
        return f"at {target.location} or passing it, bound for {market}"
    if rule is _in_market:
        return f"bound for market {target.location}"
    hs = f"{target.hs_prefix}*" if target.hs_prefix else "unknown HS"
    since = f", ETA from {target.effective_from:%Y-%m-%d %H:%M}Z" if target.effective_from else ""
    return f"bound for {target.country}, HS {hs}{since}"
