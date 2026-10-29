"""Pipeline steps 4, 7 and 8 (spec sections 3, 3.3, 3.4): confidence, expected delay, severity.

Pure functions over Signals and ContainerRefs; every weight comes from config/sentinel.yaml.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from agents.sentinel.models import (
    AisPayload,
    CarrierAdvisoryPayload,
    ComplianceFeedPayload,
    ContainerRef,
    FacilityStatusPayload,
    MandiPricePayload,
    NewsPayload,
    PortAuthorityPayload,
    Priority,
    ReeferTelemetryPayload,
    RoadAuthorityPayload,
    Signal,
    TariffPayload,
    TransportUnionPayload,
    WeatherPayload,
)
from agents.sentinel.settings import SentinelConfig

DAY = timedelta(days=1)


def round_half_up(value: float) -> int:
    """2.5 -> 3. Python's round() is banker's rounding and would give 2."""
    return math.floor(value + 0.5)


# --- confidence (step 4) --------------------------------------------------------------------------


def publisher(signal: Signal) -> str:
    """Who issued a signal. Repeats from one publisher are not independent corroboration."""
    p = signal.payload
    if isinstance(p, NewsPayload):
        return p.outlet
    if isinstance(p, CarrierAdvisoryPayload):
        return p.carrier
    return signal.location  # one authority per port, one AIS track per vessel, one feed per market


def reliability(signal: Signal, config: SentinelConfig) -> float:
    """r_source, scaled by the classifier's confidence (spec: news 0.40 × LLM confidence)."""
    return config.reliability.for_source(signal.source) * signal.type_confidence


@dataclass(frozen=True)
class Confidence:
    value: float
    reason: str


def confidence(signals: Iterable[Signal], config: SentinelConfig) -> Confidence:
    """Noisy-OR, conf = 1 − Π(1 − r), over distinct publishers (each counted at its best r)."""
    best: dict[tuple[str, str], float] = {}
    for signal in signals:
        key = (signal.source, publisher(signal))
        best[key] = max(best.get(key, 0.0), reliability(signal, config))
    value = 1 - math.prod(1 - r for r in best.values())
    terms = ", ".join(f"{source} {who} {r:.2f}" for (source, who), r in best.items())
    return Confidence(value, f"confidence {value:.3f} = 1 − Π(1 − r): {terms or 'no sources'}")


# --- expected delay (step 7) ----------------------------------------------------------------------


@dataclass(frozen=True)
class DelayEstimate:
    days: float
    is_estimate: bool  # a configured default rather than a figure stated by a source
    market_unreachable: bool
    cost_impact_pct: float
    reason: str
    shelf_loss_days: float = 0.0  # shelf life burnt on top of the delay (reefer excursions)


def expected_delay(primary: Signal, now: datetime, config: SentinelConfig) -> DelayEstimate:
    """Delay for the disruption's type, read from its primary (type-defining) signal."""
    d, p = config.delay_days, primary.payload
    match primary.type:
        case "PORT_CLOSURE":
            reopen = p.expected_reopen if isinstance(p, PortAuthorityPayload) else None
            if reopen is not None:
                days = max(0.0, (reopen - now) / DAY)
                return DelayEstimate(
                    days,
                    False,
                    False,
                    0,
                    f"delay {days:.1f} d until reopening {reopen:%Y-%m-%d %H:%M}Z",
                )
            return DelayEstimate(
                d.port_closure_unknown_end,
                True,
                True,
                0,
                f"delay {d.port_closure_unknown_end:g} d: reopening unknown, default (estimate); "
                "market unreachable",
            )
        case "PORT_CONGESTION":
            return DelayEstimate(
                d.port_congestion,
                True,
                False,
                0,
                f"delay {d.port_congestion:g} d: congestion default (estimate)",
            )
        case "LANE_SUSPENSION":
            return DelayEstimate(
                d.lane_suspension,
                True,
                True,
                0,
                f"delay {d.lane_suspension:g} d: suspension default (estimate); market unreachable",
            )
        case "ROUTE_REROUTE":
            if isinstance(p, CarrierAdvisoryPayload) and p.added_transit_days > 0:
                return DelayEstimate(
                    p.added_transit_days,
                    False,
                    False,
                    0,
                    f"delay +{p.added_transit_days:g} d: advisory transit time via {p.via}",
                )
            return DelayEstimate(
                d.route_reroute_default,
                True,
                False,
                0,
                f"delay {d.route_reroute_default:g} d: no transit time given, default (estimate)",
            )
        case "VESSEL_DELAY":
            if isinstance(p, AisPayload):
                days = (p.new_eta - p.expected_eta) / DAY
                return DelayEstimate(
                    days, False, False, 0, f"delay {days:.1f} d: AIS new ETA vs plan"
                )
            if isinstance(p, CarrierAdvisoryPayload):
                return DelayEstimate(
                    p.added_transit_days,
                    False,
                    False,
                    0,
                    f"delay +{p.added_transit_days:g} d: carrier advisory",
                )
            return DelayEstimate(
                d.vessel_delay_default,
                True,
                False,
                0,
                f"delay {d.vessel_delay_default:g} d: no ETA given, default (estimate)",
            )
        case "TARIFF_CHANGE":
            if isinstance(p, TariffPayload):
                cost = p.new_rate_pct - p.old_rate_pct
                return DelayEstimate(0, False, False, cost, f"no delay; cost impact +{cost:g} pts")
            return DelayEstimate(0, False, False, 0, "no delay; cost impact unknown until the feed")
        case "WEATHER_ALERT":
            level = p.alert_level if isinstance(p, WeatherPayload) else "orange"
            days = d.weather[level]
            return DelayEstimate(
                days, True, False, 0, f"delay {days:g} d: {level} alert (estimate)"
            )
    domestic = _domestic_delay(primary, now, config)
    if domestic is not None:
        return domestic
    raise ValueError(f"signal {primary.event_id} has no disruption type")


def _until(end: datetime, start: datetime) -> float:
    return max(0.0, (end - start) / DAY)


def _domestic_delay(primary: Signal, now: datetime, config: SentinelConfig) -> DelayEstimate | None:
    """Delay rules for the India-inside types (v2); defaults apply when a feed gives no end."""
    d, p = config.delay_days, primary.payload
    match primary.type:
        case "ROAD_CLOSURE":
            if isinstance(p, RoadAuthorityPayload) and p.expected_reopen is not None:
                days = _until(p.expected_reopen, now)
                return DelayEstimate(
                    days, False, False, 0, f"delay {days:.1f} d until the road reopens"
                )
            return DelayEstimate(
                d.road_closure_unknown_end,
                True,
                False,
                0,
                f"delay {d.road_closure_unknown_end:g} d: reopening unknown, default (estimate)",
            )
        case "TRUCK_STRIKE":
            if isinstance(p, TransportUnionPayload) and p.expected_end is not None:
                days = _until(p.expected_end, max(now, p.effective_from))
                return DelayEstimate(
                    days, False, False, 0, f"delay {days:.1f} d: announced strike length"
                )
            return DelayEstimate(
                d.truck_strike_unknown_end,
                True,
                False,
                0,
                f"delay {d.truck_strike_unknown_end:g} d: strike end unknown, default (estimate)",
            )
        case "REEFER_FAILURE":
            loss, note = 0.0, ""
            if isinstance(p, ReeferTelemetryPayload):
                above = max(0.0, p.temp_c - p.setpoint_c)
                loss = (
                    above * p.minutes_above / 60 * config.severity.reefer_loss_days_per_degree_hour
                )
                note = (
                    f"; ~{loss:.1f} d shelf life burnt (+{above:g} C for {p.minutes_above:g} min)"
                )
            return DelayEstimate(
                d.reefer_swap,
                True,
                False,
                0,
                f"delay {d.reefer_swap:g} d to move the load to another reefer (estimate){note}",
                shelf_loss_days=loss,
            )
        case "FACILITY_OUTAGE":
            if primary.scope == "full":
                return DelayEstimate(
                    d.cold_store_full,
                    True,
                    False,
                    0,
                    f"delay {d.cold_store_full:g} d: waiting for cold-store space (estimate)",
                )
            if isinstance(p, FacilityStatusPayload) and p.expected_restore is not None:
                days = _until(p.expected_restore, now)
                return DelayEstimate(
                    days, False, False, 0, f"delay {days:.1f} d until power returns"
                )
            return DelayEstimate(
                d.facility_power_unknown_end,
                True,
                False,
                0,
                f"delay {d.facility_power_unknown_end:g} d: restore time unknown "
                "(default, estimate)",
            )
        case "COMPLIANCE_HOLD":
            failed = isinstance(p, ComplianceFeedPayload) and p.status == "failed"
            closed = (
                f"; {primary.scope or 'that market'} closed to these lots (failed)"
                if failed
                else ""
            )
            if isinstance(p, ComplianceFeedPayload) and p.expected_clear is not None:
                days = _until(p.expected_clear, now)
                return DelayEstimate(
                    days, False, failed, 0, f"delay {days:.1f} d until cleared{closed}"
                )
            return DelayEstimate(
                d.compliance_hold,
                True,
                failed,
                0,
                f"delay {d.compliance_hold:g} d: re-inspection / re-test "
                f"(default, estimate){closed}",
            )
        case "MARKET_PRICE_CRASH":
            if isinstance(p, MandiPricePayload):
                base = p.baseline_price_inr_per_kg
                drop = round((base - p.modal_price_inr_per_kg) / base * 100, 1)
                return DelayEstimate(
                    0, False, False, drop, f"no delay; price {drop:g}% below baseline"
                )
            return DelayEstimate(0, False, False, 0, "no delay; price drop unknown until the feed")
    return None


def with_missed_sailings(
    delay: DelayEstimate,
    dtype: str,
    containers: Sequence[ContainerRef],
    now: datetime,
    config: SentinelConfig,
) -> DelayEstimate:
    """A box held past its vessel gate-in cut-off waits for the next sailing: a 1-day road closure
    can cost 8 days. Applies to the cutoff_types only."""
    if dtype not in config.matching.cutoff_types:
        return delay
    waiting = set(config.matching.pre_port_statuses)
    ready = now + timedelta(days=delay.days)
    missed = [
        c
        for c in containers
        if c.cutoff_at is not None and c.status in waiting and ready > c.cutoff_at
    ]
    if not missed:
        return delay
    gap = config.delay_days.next_sailing_gap
    return replace(
        delay,
        days=delay.days + gap,
        is_estimate=True,
        reason=(
            f"{delay.reason}; +{gap:g} d for the next sailing: {len(missed)} of "
            f"{len(containers)} containers miss their vessel cut-off"
        ),
    )


# --- severity (step 8) ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Severity:
    level: int
    priority: Priority
    raw: float
    reason: str


def severity(
    delay: DelayEstimate, containers: Sequence[ContainerRef], config: SentinelConfig
) -> Severity:
    """Spec section 3.4: raw = 0.55·shelf + 0.25·cost + 0.20·scale (+0.15 unreachable)."""
    s = config.severity
    known = [c.days_left for c in containers if c.days_left is not None]
    estimated = len(containers) - len(known)
    days_left = known + [s.default_days_left] * estimated
    median_left = statistics.median(days_left) if days_left else s.default_days_left
    shelf = min(1.0, (delay.days + delay.shelf_loss_days) / max(1.0, median_left))
    cost = min(1.0, max(0.0, delay.cost_impact_pct) / s.cost_divisor_pct)
    weight = sum(c.weight_kg for c in containers)
    scale = min(1.0, weight / s.scale_weight_kg)
    bonus = s.market_unreachable_bonus if delay.market_unreachable else 0.0
    raw = s.w_shelf * shelf + s.w_cost * cost + s.w_scale * scale + bonus
    level = 1 + round_half_up(4 * min(1.0, raw))
    priority: Priority = "urgent" if level >= s.urgent_at else "normal"

    burnt = f" + {delay.shelf_loss_days:.1f} d burnt" if delay.shelf_loss_days else ""
    reason = (
        f"severity {level}/5 from raw {raw:.2f}: shelf {shelf:.2f} ({delay.days:g} d{burnt} / "
        f"median {median_left:g} d left), cost {cost:.2f} ({delay.cost_impact_pct:g} pts), scale "
        f"{scale:.2f} ({weight / 1000:g} t)"
    )
    if bonus:
        reason += f", +{bonus:g} market unreachable"
    if not containers:
        reason += f"; no containers, days_left assumed {s.default_days_left:g} (estimated)"
    elif estimated:
        reason += (
            f"; days_left unknown for {estimated}, assumed {s.default_days_left:g} (estimated)"
        )
    return Severity(level, priority, raw, reason)
