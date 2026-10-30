"""Typed view of config/sentinel.yaml. The YAML is the single source: every key is required."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agents.sentinel.models import DISRUPTION_TYPES, DisruptionType, SourceName

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "sentinel.yaml"

Fraction = Annotated[float, Field(ge=0, le=1)]


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ReliabilityConfig(_Section):
    port_authority: Fraction
    tariff_feed: Fraction
    carrier_advisory: Fraction
    weather: Fraction
    ais: Fraction
    ais_port: Fraction = 0.80  # port congestion estimated from live AIS anchorage counts
    gdacs: Fraction = 0.90  # official UN/EU disaster alerts
    news: Fraction
    road_authority: Fraction
    transport_union: Fraction
    reefer_telemetry: Fraction
    facility_status: Fraction
    compliance_feed: Fraction
    mandi_prices: Fraction

    def for_source(self, source: SourceName) -> float:
        return float(getattr(self, source))


class LifecycleConfig(_Section):
    confirm_threshold: Fraction
    authoritative_sources: list[SourceName]
    merge_window_hours: float = Field(gt=0)
    candidate_ttl_hours: float = Field(gt=0)
    resolve_after_hours: float = Field(ge=0)


class ClassificationConfig(_Section):
    vessel_delay_min_hours: float = Field(ge=0)
    tariff_min_delta_pts: float = Field(ge=0)
    weather_alert_levels: list[str]
    reefer_excursion_c: float = Field(ge=0)
    reefer_excursion_minutes: float = Field(ge=0)
    mandi_drop_pct: float = Field(gt=0, lt=100)


class DelayConfig(_Section):
    port_closure_unknown_end: float = Field(ge=0)
    port_congestion: float = Field(ge=0)
    lane_suspension: float = Field(ge=0)
    route_reroute_default: float = Field(ge=0)
    vessel_delay_default: float = Field(ge=0)
    weather: dict[str, float]
    road_closure_unknown_end: float = Field(ge=0)
    truck_strike_unknown_end: float = Field(ge=0)
    reefer_swap: float = Field(ge=0)
    facility_power_unknown_end: float = Field(ge=0)
    cold_store_full: float = Field(ge=0)
    compliance_hold: float = Field(ge=0)
    next_sailing_gap: float = Field(ge=0)


class SeverityConfig(_Section):
    w_shelf: float = Field(ge=0)
    w_cost: float = Field(ge=0)
    w_scale: float = Field(ge=0)
    market_unreachable_bonus: float = Field(ge=0)
    cost_divisor_pct: float = Field(gt=0)
    scale_weight_kg: float = Field(gt=0)
    default_days_left: float = Field(gt=0)
    urgent_at: int = Field(ge=1, le=5)
    reefer_loss_days_per_degree_hour: float = Field(ge=0)


class MatchingConfig(_Section):
    in_transit_statuses: list[str]
    pre_port_statuses: list[str]
    cutoff_types: list[DisruptionType]


class PublishConfig(_Section):
    delay_change_days: float = Field(ge=0)


class NewsKeywords(_Section):
    strong: list[str]
    weak: list[str]


class NewsConfig(_Section):
    strong_confidence: Fraction
    hedged_confidence: Fraction
    hedges: list[str]
    keywords: dict[DisruptionType, NewsKeywords]


class RegionConfig(_Section):
    aliases: list[str]
    members: list[str]


class CountryConfig(_Section):
    aliases: list[str]
    exact: list[str]


class LocationsConfig(_Section):
    chokepoints: list[str]
    reroute_bypass: dict[str, list[str]]
    aliases: dict[str, list[str]]
    regions: dict[str, RegionConfig]
    countries: dict[str, CountryConfig]
    products: dict[str, list[str]]


class SentinelConfig(_Section):
    reliability: ReliabilityConfig
    lifecycle: LifecycleConfig
    classification: ClassificationConfig
    delay_days: DelayConfig
    severity: SeverityConfig
    matching: MatchingConfig
    publish: PublishConfig
    type_rank: list[DisruptionType]
    families: list[list[DisruptionType]]
    actions: dict[str, str]
    options_by_type: dict[DisruptionType, list[str]]
    news: NewsConfig
    locations: LocationsConfig

    @model_validator(mode="after")
    def _covers_every_type(self) -> SentinelConfig:
        for name, listed in (
            ("type_rank", set(self.type_rank)),
            ("families", {t for family in self.families for t in family}),
            ("options_by_type", set(self.options_by_type)),
        ):
            missing = sorted(set(DISRUPTION_TYPES) - listed)
            if missing:
                raise ValueError(f"{name} is missing {missing}")
        unknown = sorted(
            {a for opts in self.options_by_type.values() for a in opts} - set(self.actions)
        )
        if unknown:
            raise ValueError(f"options_by_type names unknown actions {unknown}")
        return self

    def family_of(self, dtype: str) -> int:
        return next(i for i, family in enumerate(self.families) if dtype in family)

    def options_for(self, dtype: str) -> list[dict[str, str]]:
        """The Scenario Planner's option families for a disruption type, with labels."""
        return [{"id": a, "label": self.actions[a]} for a in self.options_by_type.get(dtype, [])]


def load_config(path: Path | str | None = None) -> SentinelConfig:
    """Read config/sentinel.yaml (or `path`) and validate it."""
    text = Path(path or DEFAULT_CONFIG_PATH).read_text(encoding="utf-8")
    return SentinelConfig.model_validate(yaml.safe_load(text))
