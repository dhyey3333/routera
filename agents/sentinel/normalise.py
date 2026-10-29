"""Pipeline step 1 (spec section 3): validate a RawEvent's payload and canonicalise its locations.

Malformed events raise NormalisationError; the Sentinel audits them as "reject" and moves on.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import ValidationError

from agents.sentinel.models import (
    PAYLOAD_MODELS,
    AisPayload,
    AnyPayload,
    CarrierAdvisoryPayload,
    ComplianceFeedPayload,
    FacilityStatusPayload,
    MandiPricePayload,
    PortAuthorityPayload,
    RawEvent,
    ReeferTelemetryPayload,
    RoadAuthorityPayload,
    Signal,
    TariffPayload,
    TransportUnionPayload,
    WeatherPayload,
)
from agents.sentinel.settings import LocationsConfig, SentinelConfig

LOCODE_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{3}$")
MARKET_RE = re.compile(r"^([A-Z]{2})(?::(\d{2,10}))?$")
_LANE_SPLIT_RE = re.compile(r"\s*(?:-|>|/|→)\s*")


class NormalisationError(ValueError):
    """A malformed event, with the reason the audit entry will carry."""

    def __init__(self, event_id: str | None, reason: str) -> None:
        super().__init__(reason)
        self.event_id = event_id
        self.reason = reason


def phrase_key(text: str) -> str:
    """Lower-case, single-spaced form used to look matched phrases up."""
    return " ".join(text.lower().split())


def phrase_pattern(phrases: Iterable[str], *, ignore_case: bool = True) -> re.Pattern[str] | None:
    """One regex matching any phrase as whole words, longest first ("suez canal" before "suez")."""
    unique = sorted({p.strip() for p in phrases if p.strip()}, key=len, reverse=True)
    if not unique:
        return None
    body = "|".join(r"\s+".join(map(re.escape, p.split())) for p in unique)
    return re.compile(rf"(?<!\w)(?:{body})(?!\w)", re.IGNORECASE if ignore_case else 0)


class Gazetteer:
    """Canonical places from config: UN/LOCODE ports, chokepoints, regions, countries, HS codes.

    `codes` adds the reference network's ids (road segments, corridors, facilities, markets,
    carriers) so feeds can name them directly.
    """

    def __init__(self, cfg: LocationsConfig, codes: Iterable[str] = ()) -> None:
        self.known_codes = frozenset(codes)
        self.chokepoints = frozenset(cfg.chokepoints)
        self.regions = {code: tuple(region.members) for code, region in cfg.regions.items()}
        self.countries = frozenset(cfg.countries)
        self._bypass = {via: tuple(chokes) for via, chokes in cfg.reroute_bypass.items()}

        self._place_by_alias = {
            phrase_key(alias): code for code, aliases in cfg.aliases.items() for alias in aliases
        }
        for code, region in cfg.regions.items():
            self._place_by_alias.update({phrase_key(alias): code for alias in region.aliases})
        self._place_re = phrase_pattern(self._place_by_alias)

        self._country_by_alias = {
            phrase_key(alias): code for code, c in cfg.countries.items() for alias in c.aliases
        }
        self._country_by_exact = {
            exact: code for code, c in cfg.countries.items() for exact in c.exact
        }
        self._country_re = phrase_pattern(self._country_by_alias)
        self._country_exact_re = phrase_pattern(self._country_by_exact, ignore_case=False)

        self._hs_by_word = {phrase_key(w): hs for hs, words in cfg.products.items() for w in words}
        self._hs_re = phrase_pattern(self._hs_by_word)

    # --- codes ----------------------------------------------------------------------------------

    def canonical_place(self, value: str) -> str:
        """'aejea' -> AEJEA, 'Suez Canal' -> SUEZ, 'bab el mandeb' -> BAB_EL_MANDEB."""
        text = value.strip()
        code = re.sub(r"\s+", "_", text.upper())
        if code in self.chokepoints or code in self.regions or code in self.known_codes:
            return code
        alias = self._place_by_alias.get(phrase_key(text))
        if alias is not None:
            return alias
        if LOCODE_RE.match(code):
            return code
        raise ValueError(f"unknown location {value!r} (expected a UN/LOCODE, chokepoint or alias)")

    def canonical_lane(self, value: str) -> tuple[str, str, str]:
        """'innsa - aejea' -> ('INNSA', 'AEJEA', 'INNSA-AEJEA')."""
        parts = [part for part in _LANE_SPLIT_RE.split(value.strip()) if part]
        if len(parts) != 2:
            raise ValueError(f"lane {value!r} is not ORIGIN-DESTINATION")
        origin, destination = (self.canonical_place(part) for part in parts)
        return origin, destination, f"{origin}-{destination}"

    def bypassed_by(self, via: str) -> tuple[str, ...]:
        """Chokepoints a reroute via `via` avoids (CAPE -> SUEZ, BAB_EL_MANDEB); () if unknown."""
        return self._bypass.get(via, ())

    def is_via_point(self, code: str) -> bool:
        return code in self._bypass

    def expand(self, codes: Iterable[str]) -> list[str]:
        """The codes plus the members of any region among them, in order, without duplicates."""
        out: list[str] = []
        for code in codes:
            out.append(code)
            out.extend(self.regions.get(code, ()))
        return list(dict.fromkeys(out))

    def resolve(self, value: str) -> str | None:
        """Free-form location (e.g. from an LLM) -> canonical place or market ("US", "US:0306")."""
        market = MARKET_RE.match(value.strip().upper())
        if market and market.group(1) in self.countries:
            return market.group(0)
        try:
            return self.canonical_place(value)
        except ValueError:
            return None

    # --- free text ------------------------------------------------------------------------------

    def find_places(self, text: str) -> list[str]:
        """Places named in text: ports/chokepoints first, then regions, each in text order."""
        if self._place_re is None:
            return []
        found = [
            self._place_by_alias[phrase_key(m.group(0))] for m in self._place_re.finditer(text)
        ]
        specific = [code for code in found if code not in self.regions]
        regional = [code for code in found if code in self.regions]
        return list(dict.fromkeys(specific + regional))

    def find_country(self, text: str) -> str | None:
        """The earliest importing country in text ("US", "U.S." are matched case-sensitively)."""
        hits: list[tuple[int, str]] = []
        if self._country_exact_re is not None:
            hits += [
                (m.start(), self._country_by_exact[m.group(0)])
                for m in self._country_exact_re.finditer(text)
            ]
        if self._country_re is not None:
            hits += [
                (m.start(), self._country_by_alias[phrase_key(m.group(0))])
                for m in self._country_re.finditer(text)
            ]
        return min(hits)[1] if hits else None

    def find_hs(self, text: str) -> str | None:
        if self._hs_re is None:
            return None
        match = self._hs_re.search(text)
        return self._hs_by_word[phrase_key(match.group(0))] if match else None


def describe_validation_error(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors():
        where = ".".join(str(part) for part in error["loc"]) or "event"
        parts.append(f"{where}: {error['msg']}")
    return "; ".join(parts)


def _region_code(name: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", name.upper()).strip("_")


class Normaliser:
    """RawEvent (or a feed's plain dict) -> Signal with a typed payload and canonical places."""

    def __init__(self, config: SentinelConfig, gazetteer: Gazetteer | None = None) -> None:
        self.config = config
        self.gazetteer = gazetteer or Gazetteer(config.locations)

    def parse(self, event: RawEvent | Mapping[str, Any]) -> RawEvent:
        if isinstance(event, RawEvent):
            return event
        try:
            return RawEvent.model_validate(event)
        except ValidationError as exc:
            event_id = event.get("event_id") if isinstance(event, Mapping) else None
            raise NormalisationError(
                str(event_id) if event_id else None, describe_validation_error(exc)
            ) from exc

    def normalise(self, event: RawEvent | Mapping[str, Any]) -> Signal:
        raw = self.parse(event)
        try:
            payload = PAYLOAD_MODELS[raw.source].model_validate(raw.payload)
        except ValidationError as exc:
            raise NormalisationError(
                raw.event_id, f"{raw.source} payload: {describe_validation_error(exc)}"
            ) from exc
        try:
            fields = self._canonicalise(payload)  # type: ignore[arg-type]
        except ValueError as exc:
            raise NormalisationError(raw.event_id, f"{raw.source} payload: {exc}") from exc
        return Signal(
            event_id=raw.event_id,
            source=raw.source,
            received_at=raw.received_at,
            kind_hint=raw.kind_hint,
            **fields,
        )

    def _canonicalise(self, p: AnyPayload) -> dict[str, Any]:
        gaz = self.gazetteer
        if isinstance(p, PortAuthorityPayload):
            port = gaz.canonical_place(p.port_code)
            return _fields(p.model_copy(update={"port_code": port}), port, [port])
        if isinstance(p, CarrierAdvisoryPayload):
            origin, destination, lane = gaz.canonical_lane(p.affected_lane)
            via = gaz.canonical_place(p.via) if p.via else None
            footprint = [origin, destination, lane, *(gaz.bypassed_by(via) if via else ())]
            vessels = list(dict.fromkeys(p.vessel_ids))
            canon = p.model_copy(update={"affected_lane": lane, "via": via, "vessel_ids": vessels})
            return _fields(canon, lane, footprint, vessels)
        if isinstance(p, AisPayload):
            return _fields(p, p.vessel_id, [p.vessel_id], [p.vessel_id])
        if isinstance(p, TariffPayload):
            market = f"{p.importing_country}:{p.hs_code}"
            return _fields(p, market, [market, p.importing_country])
        if isinstance(p, WeatherPayload):
            if p.port_code:
                place = gaz.canonical_place(p.port_code)
                canon = p.model_copy(update={"port_code": place})
            else:
                region = p.region or ""
                try:
                    place = gaz.canonical_place(region)
                except ValueError:
                    place = _region_code(region)  # free-form weather regions are allowed
                canon = p.model_copy(update={"region": place})
            return _fields(canon, place, gaz.expand([place]))
        return self._canonicalise_domestic(p) or _fields(p, "", [])  # news: read from the text

    def _canonicalise_domestic(self, p: AnyPayload) -> dict[str, Any] | None:
        """India-inside feeds (v2)."""
        gaz = self.gazetteer
        if isinstance(p, RoadAuthorityPayload):
            segment = gaz.canonical_place(p.segment)
            return _fields(p.model_copy(update={"segment": segment}), segment, [segment])
        if isinstance(p, TransportUnionPayload):
            region = gaz.canonical_place(p.region)
            carriers = list(dict.fromkeys(p.carrier_ids))
            canon = p.model_copy(update={"region": region, "carrier_ids": carriers})
            return _fields(canon, region, [*gaz.expand([region]), p.union])
        if isinstance(p, ReeferTelemetryPayload):
            return _fields(p, p.truck_id, [p.truck_id])
        if isinstance(p, FacilityStatusPayload):
            facility = gaz.canonical_place(p.facility_id)
            return _fields(p.model_copy(update={"facility_id": facility}), facility, [facility])
        if isinstance(p, ComplianceFeedPayload):
            facility = gaz.canonical_place(p.facility_id) if p.facility_id else None
            port = gaz.canonical_place(p.port_code) if p.port_code else None
            place = facility or port or ""
            canon = p.model_copy(update={"facility_id": facility, "port_code": port})
            return _fields(canon, place, [place])
        if isinstance(p, MandiPricePayload):
            market = gaz.canonical_place(p.market_id)
            return _fields(p.model_copy(update={"market_id": market}), market, [market])
        return None


def _fields(
    payload: AnyPayload, location: str, footprint: list[str], vessels: list[str] | None = None
) -> dict[str, Any]:
    return {
        "payload": payload,
        "location": location,
        "footprint": list(dict.fromkeys(footprint)),
        "vessel_ids": vessels or [],
    }
