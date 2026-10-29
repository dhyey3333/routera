"""Pipeline step 2 (spec section 3.1): decide what kind of disruption a Signal reports.

Structured feeds go through a fixed rule table. Free-text news goes to the optional LLM, whose
reply must pass NewsClassification; with no LLM, or any LLM failure, keyword rules decide.
News is only ever evidence: merge.py never confirms a disruption from news alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from agents.sentinel.models import (
    AisPayload,
    CarrierAdvisoryPayload,
    ClassifiedBy,
    ComplianceFeedPayload,
    DisruptionType,
    FacilityStatusPayload,
    LLMClient,
    MandiPricePayload,
    NewsPayload,
    PortAuthorityPayload,
    ReeferTelemetryPayload,
    RoadAuthorityPayload,
    Signal,
    TariffPayload,
    TransportUnionPayload,
    WeatherPayload,
)
from agents.sentinel.normalise import Gazetteer, describe_validation_error, phrase_pattern
from agents.sentinel.settings import SentinelConfig


class NewsClassification(BaseModel):
    """The schema an LLM's classify_news() reply must satisfy; anything else falls back to keywords.

    type is null (or "NONE") when the article is not a disruption.
    """

    model_config = ConfigDict(extra="ignore")

    type: DisruptionType | None
    location: str | None = None
    confidence: float = Field(ge=0, le=1)

    @field_validator("type", mode="before")
    @classmethod
    def _none_words(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip().upper() in {"", "NONE", "NULL", "N/A"}:
            return None
        return value


@dataclass(frozen=True)
class _KeywordHit:
    type: DisruptionType
    keyword: str
    strong: bool


def _disruption(
    signal: Signal,
    dtype: DisruptionType,
    location: str,
    scope: str,
    reason: str,
    *,
    by: ClassifiedBy = "rules",
    confidence: float = 1.0,
    footprint: list[str] | None = None,
) -> Signal:
    update: dict[str, Any] = {
        "role": "disruption",
        "type": dtype,
        "location": location,
        "scope": scope,
        "reason": reason,
        "classified_by": by,
        "type_confidence": confidence,
    }
    if footprint is not None:
        update["footprint"] = footprint
    return signal.model_copy(update=update)


def _resolution(signal: Signal, reason: str) -> Signal:
    return signal.model_copy(
        update={"role": "resolution", "type": None, "reason": reason, "classified_by": "rules"}
    )


def _noise(signal: Signal, reason: str, by: ClassifiedBy = "rules") -> Signal:
    return signal.model_copy(
        update={"role": "noise", "type": None, "reason": reason, "classified_by": by}
    )


def _tariff_scope(market: str) -> str:
    """Fingerprint scope "country+hs" for a "US:0306" location; empty without an HS code."""
    return market.replace(":", "+") if ":" in market else ""


class Classifier:
    def __init__(
        self, config: SentinelConfig, gazetteer: Gazetteer, llm: LLMClient | None = None
    ) -> None:
        self.config = config
        self.gazetteer = gazetteer
        self.llm = llm
        keywords = config.news.keywords
        self._strong = [(t, phrase_pattern(kw.strong)) for t, kw in keywords.items()]
        self._weak = [(t, phrase_pattern(kw.weak)) for t, kw in keywords.items()]
        self._hedges = phrase_pattern(config.news.hedges)

    def classify(self, signal: Signal) -> Signal:
        p = signal.payload
        if isinstance(p, NewsPayload):
            return self._news(signal, p)
        if isinstance(p, PortAuthorityPayload):
            return self._port_authority(signal, p)
        if isinstance(p, CarrierAdvisoryPayload):
            return self._carrier_advisory(signal, p)
        if isinstance(p, AisPayload):
            return self._ais(signal, p)
        if isinstance(p, TariffPayload):
            return self._tariff(signal, p)
        if isinstance(p, WeatherPayload):
            return self._weather(signal, p)
        if isinstance(p, RoadAuthorityPayload):
            return self._road(signal, p)
        if isinstance(p, TransportUnionPayload):
            return self._union(signal, p)
        if isinstance(p, ReeferTelemetryPayload):
            return self._reefer(signal, p)
        if isinstance(p, FacilityStatusPayload):
            return self._facility(signal, p)
        if isinstance(p, ComplianceFeedPayload):
            return self._compliance(signal, p)
        if isinstance(p, MandiPricePayload):
            return self._mandi(signal, p)
        raise TypeError(f"no classification rule for {type(p).__name__}")

    # --- structured rule table (spec section 3.1) -----------------------------------------------

    def _port_authority(self, s: Signal, p: PortAuthorityPayload) -> Signal:
        port = p.port_code
        if p.status == "closed":
            reopen = (
                f"reopen {p.expected_reopen:%Y-%m-%d %H:%M}Z"
                if p.expected_reopen
                else "reopen unknown"
            )
            return _disruption(
                s, "PORT_CLOSURE", port, "", f"port_authority: {port} closed ({reopen})"
            )
        if p.status == "congested":
            return _disruption(s, "PORT_CONGESTION", port, "", f"port_authority: {port} congested")
        return _resolution(s, f"port_authority: {port} open (reopen signal)")

    def _carrier_advisory(self, s: Signal, p: CarrierAdvisoryPayload) -> Signal:
        lane, days = p.affected_lane, p.added_transit_days
        if p.action == "suspend":
            return _disruption(
                s, "LANE_SUSPENSION", lane, lane, f"carrier_advisory: {p.carrier} suspends {lane}"
            )
        if p.action == "reroute":
            if p.via is None:
                return _noise(s, f"carrier_advisory: {p.carrier} reroutes {lane} without 'via'")
            bypassed = self.gazetteer.bypassed_by(p.via)
            if not bypassed:
                return _disruption(
                    s,
                    "ROUTE_REROUTE",
                    lane,
                    lane,
                    f"carrier_advisory: {p.carrier} reroutes {lane} via {p.via} (+{days:g} d); "
                    "bypassed chokepoint unknown, located on the lane",
                )
            return _disruption(
                s,
                "ROUTE_REROUTE",
                bypassed[0],
                lane,
                f"carrier_advisory: {p.carrier} reroutes {lane} via {p.via} (+{days:g} d), "
                f"bypassing {', '.join(bypassed)}",
            )
        # action == "delay" has no row in the spec's table. It is a VESSEL_DELAY on the lane,
        # matched to the listed vessels, rather than a suspension of the whole lane.
        threshold = self.config.classification.vessel_delay_min_hours
        if days * 24 < threshold:
            return _noise(
                s,
                f"carrier_advisory: {lane} delay +{days:g} d is under {threshold:g} h",
            )
        vessels = ", ".join(s.vessel_ids) or "all vessels"
        return _disruption(
            s,
            "VESSEL_DELAY",
            lane,
            lane,
            f"carrier_advisory: {p.carrier} delays {lane} +{days:g} d for {vessels} "
            "(action=delay -> VESSEL_DELAY on the lane)",
        )

    def _ais(self, s: Signal, p: AisPayload) -> Signal:
        hours = (p.new_eta - p.expected_eta).total_seconds() / 3600
        threshold = self.config.classification.vessel_delay_min_hours
        if hours >= threshold:
            return _disruption(
                s,
                "VESSEL_DELAY",
                p.vessel_id,
                p.vessel_id,
                f"ais: {p.vessel_id} ETA +{hours / 24:.1f} d (>= {threshold:g} h)",
            )
        return _resolution(
            s, f"ais: {p.vessel_id} ETA {hours:+.0f} h, under the {threshold:g} h threshold"
        )

    def _tariff(self, s: Signal, p: TariffPayload) -> Signal:
        delta = p.new_rate_pct - p.old_rate_pct
        threshold = self.config.classification.tariff_min_delta_pts
        change = f"{s.location} {p.old_rate_pct:g}% -> {p.new_rate_pct:g}%"
        if delta >= threshold:
            return _disruption(
                s,
                "TARIFF_CHANGE",
                s.location,
                _tariff_scope(s.location),
                f"tariff_feed: {change} (+{delta:g} pts >= {threshold:g})",
            )
        if delta < 0:
            return _resolution(s, f"tariff_feed: {change} (rate reversal)")
        return _noise(
            s, f"tariff_feed: {change} (+{delta:g} pts) is under the {threshold:g}-pt threshold"
        )

    def _weather(self, s: Signal, p: WeatherPayload) -> Signal:
        levels = self.config.classification.weather_alert_levels
        alert = f"weather: {p.alert_level} {p.hazard} alert for {s.location}"
        if p.alert_level in levels:
            return _disruption(s, "WEATHER_ALERT", s.location, "", alert)
        return _noise(s, f"{alert} is below the alert levels {levels}")

    # --- India-inside rule table (v2) -----------------------------------------------------------

    def _road(self, s: Signal, p: RoadAuthorityPayload) -> Signal:
        until = (
            f"reopen {p.expected_reopen:%Y-%m-%d %H:%M}Z" if p.expected_reopen else "reopen unknown"
        )
        cause = f": {p.reason}" if p.reason else ""
        if p.status == "closed":
            return _disruption(
                s,
                "ROAD_CLOSURE",
                p.segment,
                "",
                f"road_authority: {p.segment} closed{cause} ({until})",
            )
        if p.status == "restricted":
            return _noise(s, f"road_authority: {p.segment} restricted{cause}; traffic still moves")
        return _resolution(s, f"road_authority: {p.segment} open again")

    def _union(self, s: Signal, p: TransportUnionPayload) -> Signal:
        start = f"from {p.effective_from:%Y-%m-%d %H:%M}Z"
        if p.action == "strike":
            end = f"until {p.expected_end:%Y-%m-%d %H:%M}Z" if p.expected_end else "end unknown"
            return _disruption(
                s,
                "TRUCK_STRIKE",
                s.location,
                p.union,
                f"transport_union: {p.union} strike in {s.location} {start}, {end}",
            )
        if p.action == "go_slow":
            return _noise(
                s, f"transport_union: {p.union} go-slow in {s.location}; trucks still run"
            )
        return _resolution(s, f"transport_union: {p.union} strike in {s.location} called off")

    def _reefer(self, s: Signal, p: ReeferTelemetryPayload) -> Signal:
        cls = self.config.classification
        above = p.temp_c - p.setpoint_c
        reading = f"{p.truck_id} at {p.temp_c:g} C vs setpoint {p.setpoint_c:g} C"
        if p.status == "breakdown":
            return _disruption(
                s,
                "REEFER_FAILURE",
                p.truck_id,
                "",
                f"reefer_telemetry: {reading}, unit broken down",
            )
        if above >= cls.reefer_excursion_c and p.minutes_above >= cls.reefer_excursion_minutes:
            return _disruption(
                s,
                "REEFER_FAILURE",
                p.truck_id,
                "",
                f"reefer_telemetry: {reading} for {p.minutes_above:g} min "
                f"(>= +{cls.reefer_excursion_c:g} C "
                f"for {cls.reefer_excursion_minutes:g} min)",
            )
        return _resolution(s, f"reefer_telemetry: {reading}, within limits")

    def _facility(self, s: Signal, p: FacilityStatusPayload) -> Signal:
        if p.status == "power_failure":
            return _disruption(
                s,
                "FACILITY_OUTAGE",
                p.facility_id,
                "power_failure",
                f"facility_status: {p.facility_id} power failure",
            )
        if p.status == "full":
            free = f" ({p.free_capacity_t:g} t free)" if p.free_capacity_t is not None else ""
            return _disruption(
                s,
                "FACILITY_OUTAGE",
                p.facility_id,
                "full",
                f"facility_status: {p.facility_id} full{free}",
            )
        return _resolution(s, f"facility_status: {p.facility_id} back to normal")

    def _compliance(self, s: Signal, p: ComplianceFeedPayload) -> Signal:
        market = p.market_country or "any export market"
        what = f"compliance_feed: {p.authority} {p.status} at {s.location} for {market}"
        if p.status == "cleared":
            return _resolution(s, f"{what}: cleared")
        cause = f": {p.reason}" if p.reason else ""
        return _disruption(
            s, "COMPLIANCE_HOLD", s.location, p.market_country or "", f"{what}{cause}"
        )

    def _mandi(self, s: Signal, p: MandiPricePayload) -> Signal:
        threshold = self.config.classification.mandi_drop_pct
        drop = (
            (p.baseline_price_inr_per_kg - p.modal_price_inr_per_kg)
            / p.baseline_price_inr_per_kg
            * 100
        )
        arrivals = f", arrivals {p.arrivals_t:g} t" if p.arrivals_t is not None else ""
        quote = (
            f"mandi_prices: {p.commodity} at {s.location} Rs {p.modal_price_inr_per_kg:g}/kg vs "
            f"Rs {p.baseline_price_inr_per_kg:g}/kg baseline ({-drop:+.1f}%){arrivals}"
        )
        if drop >= threshold:
            return _disruption(
                s, "MARKET_PRICE_CRASH", s.location, "", f"{quote} (>= {threshold:g}% drop)"
            )
        return _resolution(s, f"{quote}, under the {threshold:g}% crash threshold")

    # --- news (LLM with keyword fallback) -------------------------------------------------------

    def _news(self, s: Signal, p: NewsPayload) -> Signal:
        text = f"{p.headline}\n{p.body}"
        note = ""
        if self.llm is not None:
            try:
                return self._news_by_llm(s, p, text)
            except Exception as exc:  # any LLM failure falls back to keywords
                note = f"LLM classification rejected ({type(exc).__name__}: {_first_line(exc)}); "
        return self._news_by_keywords(s, text, note)

    def _news_by_llm(self, s: Signal, p: NewsPayload, text: str) -> Signal:
        assert self.llm is not None
        result = NewsClassification.model_validate(self.llm.classify_news(p.headline, p.body))
        if result.type is None:
            return _noise(
                s, f"llm: not a disruption (confidence {result.confidence:.2f})", by="llm"
            )
        location = self.gazetteer.resolve(result.location or "")
        if location is None:
            raise ValueError(f"unknown location {result.location!r}")
        if result.type == "ROUTE_REROUTE" and self.gazetteer.is_via_point(location):
            location = self.gazetteer.bypassed_by(location)[0]
        footprint = self.gazetteer.expand([location, *self.gazetteer.find_places(text)])
        scope = _tariff_scope(location) if result.type == "TARIFF_CHANGE" else ""
        return _disruption(
            s,
            result.type,
            location,
            scope,
            f"llm: {result.type} @ {location}, confidence {result.confidence:.2f}",
            by="llm",
            confidence=result.confidence,
            footprint=footprint,
        )

    def _news_by_keywords(self, s: Signal, text: str, note: str) -> Signal:
        hit = self._match_type(text)
        if hit is None:
            return _noise(s, f"{note}keywords: no disruption terms", by="keywords")
        location, footprint = self._news_location(hit.type, text)
        if location is None:
            return _noise(
                s,
                f"{note}keywords: '{hit.keyword}' suggests {hit.type} but names no known place",
                by="keywords",
            )
        hedge = self._hedges.search(text) if self._hedges else None
        news = self.config.news
        confidence = (
            news.strong_confidence if hit.strong and hedge is None else news.hedged_confidence
        )
        detail = "strong" if hit.strong else "weak"
        if hedge is not None:
            detail += f", hedged by '{hedge.group(0).lower()}'"
        scope = _tariff_scope(location) if hit.type == "TARIFF_CHANGE" else ""
        return _disruption(
            s,
            hit.type,
            location,
            scope,
            f"{note}keywords: '{hit.keyword}' ({detail}) -> {hit.type} @ {location}, "
            f"confidence {confidence:.2f}",
            by="keywords",
            confidence=confidence,
            footprint=footprint,
        )

    def _match_type(self, text: str) -> _KeywordHit | None:
        for patterns, strong in ((self._strong, True), (self._weak, False)):
            for dtype, pattern in patterns:
                match = pattern.search(text) if pattern else None
                if match:
                    return _KeywordHit(dtype, match.group(0).lower(), strong)
        return None

    def _news_location(self, dtype: DisruptionType, text: str) -> tuple[str | None, list[str]]:
        gaz = self.gazetteer
        if dtype == "TARIFF_CHANGE":
            country = gaz.find_country(text)
            if country is None:
                return None, []
            hs = gaz.find_hs(text)
            market = f"{country}:{hs}" if hs else country
            return market, [market, country]
        places = gaz.find_places(text)
        if not places:
            return None, []
        primary = places[0]
        if dtype == "ROUTE_REROUTE":
            # "around the Cape" names the detour; the disruption sits at the chokepoint avoided.
            avoided = [code for code in places if not gaz.is_via_point(code)]
            primary = avoided[0] if avoided else (gaz.bypassed_by(primary) or (primary,))[0]
        return primary, gaz.expand([primary, *places])


def _first_line(exc: BaseException) -> str:
    if isinstance(exc, ValidationError):
        return describe_validation_error(exc)[:160]
    lines = str(exc).strip().splitlines()
    return lines[0][:160] if lines else "no detail"
