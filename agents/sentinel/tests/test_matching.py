"""Spec section 3.2: affected-container matching. Gulf closure flags exactly the 12 grape boxes."""

from __future__ import annotations

from typing import Any

import pytest

from agents.sentinel import matching
from agents.sentinel.fakes import JsonContainerRepo
from agents.sentinel.models import ContainerRef
from agents.sentinel.normalise import Gazetteer
from agents.sentinel.replay import load_scenario
from agents.sentinel.settings import SentinelConfig
from agents.sentinel.tests.builders import (
    GULF_GRAPES,
    WHEN,
    advisory,
    ais,
    at,
    news,
    port,
    tariff,
    weather,
)
from agents.sentinel.tests.conftest import ClassifyFn

ARABIAN = ["MV-ARABIAN-01", "MV-ARABIAN-02"]
EUROPE_VIA_SUEZ = ["GRP-13", "GRP-14", "GRP-16", "SHR-10"]
US_SHRIMP = [f"SHR-{i:02d}" for i in range(1, 10)]


@pytest.fixture
def affected(classify: ClassifyFn, gazetteer: Gazetteer, config: SentinelConfig) -> Any:
    """IDs matched by the disruption a payload describes, against a scenario book or all 31."""

    def _run(source: str, payload: dict[str, Any], scenario: str | None = None) -> list[str]:
        ids = load_scenario(scenario).container_ids if scenario else None
        target = matching.target_for(classify(source, payload), gazetteer)
        containers = JsonContainerRepo(container_ids=ids).list_active()
        return [c.container_id for c in matching.match_containers(target, containers, config)]

    return _run


@pytest.mark.parametrize("scenario", ["gulf_closure", None], ids=["gulf-book", "all-31"])
def test_gulf_closure_flags_exactly_the_12_grape_containers(
    affected: Any, scenario: str | None
) -> None:
    assert affected("port_authority", port("AEJEA", "closed"), scenario) == GULF_GRAPES


def test_innsa_congestion_flags_no_gulf_grapes(affected: Any) -> None:
    assert affected("port_authority", port("INNSA", "congested"), "gulf_closure") == ["POM-01"]


def test_port_rules_only_count_containers_in_transit(
    classify: ClassifyFn, gazetteer: Gazetteer, config: SentinelConfig
) -> None:
    base = JsonContainerRepo(container_ids=["GRP-01"]).list_active()[0]
    boxes = [
        base.model_copy(update={"container_id": f"X-{status}", "status": status})
        for status in ("planned", "packed", "at_port", "at_sea", "delivered")
    ]
    target = matching.target_for(classify("port_authority", port("AEJEA", "closed")), gazetteer)
    hits = [c.container_id for c in matching.match_containers(target, boxes, config)]
    assert hits == ["X-at_port", "X-at_sea", "X-packed"]


def test_weather_alert_for_a_region_covers_its_ports(affected: Any) -> None:
    hits = affected("weather", weather("red", region="Gulf"), "gulf_closure")
    assert hits == ["BAN-01", *GULF_GRAPES]


def test_lane_suspension_matches_origin_and_destination(affected: Any) -> None:
    assert affected("carrier_advisory", advisory("suspend")) == GULF_GRAPES
    assert affected("carrier_advisory", advisory("suspend", lane="INMUN-QAHMD")) == ["BAN-01"]


@pytest.mark.parametrize("scenario", ["red_sea_reroute", None], ids=["red-sea-book", "all-31"])
def test_reroute_flags_containers_routed_through_suez(affected: Any, scenario: str | None) -> None:
    payload = advisory("reroute", "CAPE", 25, lane="INNSA-NLRTM")
    assert affected("carrier_advisory", payload, scenario) == EUROPE_VIA_SUEZ


def test_vessel_delay_matches_by_vessel_or_lane(affected: Any) -> None:
    assert affected("ais", ais("MV-ARABIAN-02", WHEN, at(days=3))) == GULF_GRAPES[6:]
    with_vessels = advisory("delay", days=6, vessels=ARABIAN)
    assert affected("carrier_advisory", with_vessels) == GULF_GRAPES
    assert affected("carrier_advisory", advisory("delay", days=6)) == GULF_GRAPES  # lane fallback


@pytest.mark.parametrize("scenario", ["us_tariff_spike", None], ids=["tariff-book", "all-31"])
def test_tariff_flags_us_bound_shrimp_arriving_after_the_effective_date(
    affected: Any, scenario: str | None
) -> None:
    payload = tariff("US", "0306", 10, 50, effective="2026-03-04T04:01:00Z")
    assert affected("tariff_feed", payload, scenario) == US_SHRIMP  # not SHR-11 (lands earlier)


def test_tariff_needs_the_hs_prefix(affected: Any) -> None:
    assert affected("tariff_feed", tariff("US", "0806", 10, 50)) == ["GRP-15"]
    assert affected("news", news("US slaps duties on imports")) == []  # no HS code known


def test_describe_explains_the_rule(
    classify: ClassifyFn, gazetteer: Gazetteer, config: SentinelConfig
) -> None:
    def said(source: str, payload: dict[str, Any]) -> str:
        return matching.describe(matching.target_for(classify(source, payload), gazetteer), config)

    assert said("port_authority", port("AEJEA", "closed")).startswith(
        "destination or route leg AEJEA"
    )
    assert said("carrier_advisory", advisory("reroute", "CAPE", 25)) == "route legs include SUEZ"
    assert "MV-ARABIAN-01" in said("carrier_advisory", advisory("delay", days=6, vessels=ARABIAN))
    assert said("tariff_feed", tariff("US", "0306", 10, 50)).startswith("bound for US, HS 0306*")


def test_container_refs_from_teammates_match_after_normalisation(
    classify: ClassifyFn, gazetteer: Gazetteer, config: SentinelConfig
) -> None:
    box = ContainerRef.model_validate(
        {
            "container_id": "T-1",
            "crop": "grapes",
            "hs_code": "0806.10",
            "weight_kg": 18000,
            "vessel_id": "mv-arabian-01",
            "origin_port": "innsa",
            "destination_port": "aejea",
            "destination_country": "ae",
            "route_legs": ["hormuz", "aejea"],
            "eta": "2026-03-04T08:00:00Z",
            "status": "AT_SEA",
            "days_left": 12,
        }
    )
    target = matching.target_for(classify("port_authority", port("AEJEA", "closed")), gazetteer)
    assert matching.match_containers(target, [box], config) == [box]
