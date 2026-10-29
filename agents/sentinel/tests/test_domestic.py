"""India-inside (v2): the domestic leg's feeds, rules, matching, delays and lifecycle."""

from __future__ import annotations

from typing import Any

import pytest

from agents.sentinel import matching, scoring
from agents.sentinel.fakes import INDIA_CONTAINERS_PATH, JsonContainerRepo, JsonNetworkRepo
from agents.sentinel.normalise import Gazetteer, NormalisationError, Normaliser
from agents.sentinel.replay import load_scenario, make_rig
from agents.sentinel.settings import SentinelConfig, load_config
from agents.sentinel.tests.builders import T0, WHEN, Harness, actions, at, news, port, weather
from agents.sentinel.tests.conftest import ClassifyFn, RawFactory

MAIN = load_scenario("nashik_corridor_strike").container_ids
RIPPLE = load_scenario("gulf_ripple").container_ids
EU_BOXES = [f"NSK-{i:02d}" for i in range(1, 13)]


def road(segment: str, status: str, *, reopen: str | None = None) -> dict[str, Any]:
    return {
        "segment": segment,
        "status": status,
        "reason": "test",
        "effective_from": WHEN,
        "expected_reopen": reopen,
    }


def union(
    action: str, *, effective: str = WHEN, end: str | None = None, carriers: list[str] | None = None
) -> dict[str, Any]:
    return {
        "union": "NSK-TRANSPORTERS",
        "region": "Nashik",
        "action": action,
        "effective_from": effective,
        "expected_end": end,
        "carrier_ids": carriers or [],
    }


def reefer(
    truck: str, temp: float, *, minutes: float = 60, status: str = "excursion"
) -> dict[str, Any]:
    return {
        "truck_id": truck,
        "setpoint_c": 2,
        "temp_c": temp,
        "minutes_above": minutes,
        "status": status,
    }


def facility(fid: str, status: str, free: float | None = None) -> dict[str, Any]:
    return {"facility_id": fid, "status": status, "free_capacity_t": free}


def compliance(status: str, *, place: str = "PH-NPD", market: str | None = None) -> dict[str, Any]:
    return {
        "authority": "NPPO (SIM)",
        "facility_id": place,
        "market_country": market,
        "status": status,
        "reason": "test",
    }


def mandi(market: str, modal: float, baseline: float = 62) -> dict[str, Any]:
    return {
        "market_id": market,
        "commodity": "grapes",
        "modal_price_inr_per_kg": modal,
        "baseline_price_inr_per_kg": baseline,
        "arrivals_t": 900,
    }


@pytest.fixture
def india(config: SentinelConfig) -> Harness:
    """A tracker over the India-inside book, clock at T0."""
    return Harness(config, books=INDIA_CONTAINERS_PATH)


@pytest.fixture
def affected(classify: ClassifyFn, gazetteer: Gazetteer, config: SentinelConfig) -> Any:
    """Containers a payload's disruption matches in an India-inside scenario book."""
    network = JsonNetworkRepo().network()

    def _run(source: str, payload: dict[str, Any], book: list[str] = MAIN) -> list[str]:
        target = matching.target_for(classify(source, payload), gazetteer, network)
        boxes = JsonContainerRepo(INDIA_CONTAINERS_PATH, container_ids=book).list_active()
        return [c.container_id for c in matching.match_containers(target, boxes, config)]

    return _run


# --- the India-inside rule table -----------------------------------------------------------------

DOMESTIC_RULES = [
    pytest.param(
        "road_authority",
        road("Kasara ghat", "closed"),
        "disruption",
        "ROAD_CLOSURE",
        "KASARA_GHAT",
        "",
        id="road-closed",
    ),
    pytest.param(
        "road_authority",
        road("KASARA_GHAT", "restricted"),
        "noise",
        None,
        "KASARA_GHAT",
        "",
        id="road-restricted",
    ),
    pytest.param(
        "road_authority",
        road("kasara", "open"),
        "resolution",
        None,
        "KASARA_GHAT",
        "",
        id="road-open",
    ),
    pytest.param(
        "transport_union",
        union("strike"),
        "disruption",
        "TRUCK_STRIKE",
        "NASHIK",
        "NSK-TRANSPORTERS",
        id="strike",
    ),
    pytest.param("transport_union", union("go_slow"), "noise", None, "NASHIK", "", id="go-slow"),
    pytest.param(
        "transport_union",
        union("called_off"),
        "resolution",
        None,
        "NASHIK",
        "",
        id="strike-called-off",
    ),
    pytest.param(
        "reefer_telemetry",
        reefer("T-1", 12, minutes=180),
        "disruption",
        "REEFER_FAILURE",
        "T-1",
        "",
        id="reefer-excursion",
    ),
    pytest.param(
        "reefer_telemetry",
        reefer("T-1", 2, minutes=0, status="breakdown"),
        "disruption",
        "REEFER_FAILURE",
        "T-1",
        "",
        id="reefer-breakdown",
    ),
    pytest.param(
        "reefer_telemetry",
        reefer("T-1", 4, minutes=180),
        "resolution",
        None,
        "T-1",
        "",
        id="reefer-only-plus-2C",
    ),
    pytest.param(
        "reefer_telemetry",
        reefer("T-1", 12, minutes=20),
        "resolution",
        None,
        "T-1",
        "",
        id="reefer-too-brief",
    ),
    pytest.param(
        "facility_status",
        facility("cs-nsk", "full", 0),
        "disruption",
        "FACILITY_OUTAGE",
        "CS-NSK",
        "full",
        id="cold-store-full",
    ),
    pytest.param(
        "facility_status",
        facility("PH-PMP", "power_failure"),
        "disruption",
        "FACILITY_OUTAGE",
        "PH-PMP",
        "power_failure",
        id="packhouse-power-failure",
    ),
    pytest.param(
        "facility_status",
        facility("CS-NSK", "ok"),
        "resolution",
        None,
        "CS-NSK",
        "",
        id="facility-ok",
    ),
    pytest.param(
        "compliance_feed",
        compliance("delayed", market="NL"),
        "disruption",
        "COMPLIANCE_HOLD",
        "PH-NPD",
        "NL",
        id="phyto-delayed",
    ),
    pytest.param(
        "compliance_feed",
        compliance("cleared"),
        "resolution",
        None,
        "PH-NPD",
        "",
        id="phyto-cleared",
    ),
    pytest.param(
        "mandi_prices",
        mandi("Vashi", 38),
        "disruption",
        "MARKET_PRICE_CRASH",
        "MKT-VASHI",
        "",
        id="mandi-crash",
    ),
    pytest.param(
        "mandi_prices",
        mandi("MKT-VASHI", 55),
        "resolution",
        None,
        "MKT-VASHI",
        "",
        id="mandi-11pct-dip",
    ),
]


@pytest.mark.parametrize(
    ("source", "payload", "role", "dtype", "location", "scope"), DOMESTIC_RULES
)
def test_domestic_rule_table(
    classify: ClassifyFn,
    source: str,
    payload: dict[str, Any],
    role: str,
    dtype: str | None,
    location: str,
    scope: str,
) -> None:
    signal = classify(source, payload)
    assert (signal.role, signal.type, signal.location, signal.scope) == (
        role,
        dtype,
        location,
        scope,
    )
    assert signal.reason


@pytest.mark.parametrize(
    ("source", "payload", "reason_part"),
    [
        pytest.param(
            "road_authority",
            {"segment": "KASARA_GHAT", "status": "closed"},
            "effective_from",
            id="road-no-start",
        ),
        pytest.param("transport_union", union("picket"), "action", id="union-unknown-action"),
        pytest.param(
            "compliance_feed",
            compliance("delayed") | {"facility_id": None},
            "facility_id or a port",
            id="compliance-no-place",
        ),
        pytest.param(
            "mandi_prices", mandi("MKT-VASHI", 38, baseline=0), "baseline", id="mandi-zero"
        ),
        pytest.param(
            "facility_status", facility("CS-XYZ", "full"), "unknown location", id="unknown-facility"
        ),
    ],
)
def test_malformed_domestic_events_are_rejected(
    normaliser: Normaliser,
    make_raw: RawFactory,
    source: str,
    payload: dict[str, Any],
    reason_part: str,
) -> None:
    with pytest.raises(NormalisationError) as caught:
        normaliser.normalise(make_raw(source, payload))
    assert reason_part in caught.value.reason


@pytest.mark.parametrize(
    ("headline", "dtype", "location"),
    [
        pytest.param(
            "Transporters' strike paralyses Nashik grape belt",
            "TRUCK_STRIKE",
            "NASHIK",
            id="strike",
        ),
        pytest.param(
            "Landslide shuts Kasara ghat",
            "ROAD_CLOSURE",
            "KASARA_GHAT",
            id="landslide-not-port-closure",
        ),
        pytest.param(
            "Power cuts hit Nashik pack houses", "FACILITY_OUTAGE", "NASHIK", id="power-cut"
        ),
        pytest.param(
            "Residue test backlog delays Nashik grapes for Europe",
            "COMPLIANCE_HOLD",
            "NASHIK",
            id="residue-backlog-not-congestion",
        ),
        pytest.param(
            "Grape prices crash at Vashi as diverted lots flood in",
            "MARKET_PRICE_CRASH",
            "MKT-VASHI",
            id="mandi-crash",
        ),
        pytest.param("Hailstorm lashes Niphad vineyards", "WEATHER_ALERT", "NASHIK", id="hail"),
        pytest.param(
            "US duties on shrimp after residue findings",
            "TARIFF_CHANGE",
            "US:0306",
            id="tariff-keeps-residue-as-weak",
        ),
    ],
)
def test_domestic_news_keywords(
    classify: ClassifyFn, headline: str, dtype: str, location: str
) -> None:
    signal = classify("news", news(headline))
    assert (signal.type, signal.location) == (dtype, location)


def test_network_entities_and_codes() -> None:
    network = JsonNetworkRepo().network()
    assert network.simulated is True
    assert network.union_members("NSK-TRANSPORTERS") == {"TRK-GODAVARI", "TRK-SAHYADRI"}
    assert network.corridors_through("KASARA_GHAT") == {"NSK-JNPT"}
    cold_store = network.facility("CS-NSK")
    assert cold_store is not None and cold_store.cost_inr_per_t_day == 45
    assert {"KASARA_GHAT", "PH-NPD", "MKT-AZADPUR", "NSK-JNPT-RAIL"} <= network.codes()


def test_config_must_rank_every_type() -> None:
    data = load_config().model_dump()
    data["type_rank"] = data["type_rank"][:-1]
    with pytest.raises(ValueError, match="type_rank is missing"):
        type(load_config()).model_validate(data)


# --- matching on the domestic leg ----------------------------------------------------------------


def test_strike_hits_every_load_moved_by_union_carriers(affected: Any) -> None:
    hits = affected("transport_union", union("strike"))
    assert hits == ["DOM-01", "DOM-02", "DOM-03", "DOM-05", *EU_BOXES]  # not DOM-04 (Pune carrier)
    assert affected("transport_union", union("strike", carriers=["TRK-WESTERN"])) == ["DOM-04"]


def test_road_closure_matches_the_segment_and_its_corridor(affected: Any) -> None:
    assert affected("road_authority", road("KASARA_GHAT", "closed")) == [
        "DOM-01",
        "DOM-02",
        "DOM-03",
        *EU_BOXES,
    ]
    corridor = affected("road_authority", road("NSK-JNPT", "closed"))
    assert corridor == ["DOM-03", *EU_BOXES[:8]]  # boxes not yet on the road


def test_weather_over_a_region_reaches_its_pack_houses(affected: Any) -> None:
    hits = affected("weather", weather("orange", region="Nashik"))
    assert hits == ["DOM-03", "DOM-05", *EU_BOXES[:8]]


def test_facility_outages(affected: Any) -> None:
    assert affected("facility_status", facility("PH-PMP", "power_failure")) == EU_BOXES[:8]
    full = affected("facility_status", facility("CS-NSK", "full", 0))
    assert full == ["DOM-03", "DOM-05", *EU_BOXES[:8]]  # stranded at pack houses in the region


def test_compliance_hold_scopes_by_market(affected: Any) -> None:
    gulf_lots = ["GLF-27", "GLF-28", "GLF-29", "GLF-30"]
    assert affected("compliance_feed", compliance("delayed"), RIPPLE) == gulf_lots
    assert affected("compliance_feed", compliance("delayed", market="AE"), RIPPLE) == gulf_lots
    assert affected("compliance_feed", compliance("delayed", market="NL"), RIPPLE) == []


def test_reefer_and_mandi_matching(affected: Any) -> None:
    assert affected("reefer_telemetry", reefer("TRK-MH15-2201", 12, minutes=180)) == ["DOM-01"]
    assert affected("mandi_prices", mandi("Vashi", 38)) == ["DOM-01", "DOM-02", "DOM-03"]


# --- delays: cut-offs, shelf loss, markets -------------------------------------------------------


def _delay(
    classify: ClassifyFn,
    gazetteer: Gazetteer,
    config: SentinelConfig,
    source: str,
    payload: dict[str, Any],
) -> scoring.DelayEstimate:
    signal = classify(source, payload)
    target = matching.target_for(signal, gazetteer, JsonNetworkRepo().network())
    boxes = JsonContainerRepo(INDIA_CONTAINERS_PATH, container_ids=MAIN).list_active()
    containers = matching.match_containers(target, boxes, config)
    delay = scoring.expected_delay(signal, T0, config)
    return scoring.with_missed_sailings(delay, signal.type or "", containers, T0, config)


def test_missing_the_vessel_cutoff_adds_the_next_sailing(
    classify: ClassifyFn, gazetteer: Gazetteer, config: SentinelConfig
) -> None:
    unknown = _delay(classify, gazetteer, config, "road_authority", road("KASARA_GHAT", "closed"))
    assert unknown.days == 1 + 7 and "12 of 15 containers miss" in unknown.reason
    short = _delay(
        classify,
        gazetteer,
        config,
        "road_authority",
        road("KASARA_GHAT", "closed", reopen=at(hours=6)),
    )
    assert (short.days, short.is_estimate) == (
        pytest.approx(0.25),
        False,
    )  # back before the cut-off


def test_weather_never_adds_a_sailing_gap(
    classify: ClassifyFn, gazetteer: Gazetteer, config: SentinelConfig
) -> None:
    red = _delay(classify, gazetteer, config, "weather", weather("red", region="Nashik"))
    assert red.days == 3  # past the cut-off, but weather is not a cut-off type


def test_announced_strike_length_counts_from_its_start(
    classify: ClassifyFn, gazetteer: Gazetteer, config: SentinelConfig
) -> None:
    strike = union("strike", effective=at(hours=10), end=at(hours=34))
    delay = _delay(classify, gazetteer, config, "transport_union", strike)
    assert delay.days == pytest.approx(1 + 7)  # one day of strike still misses the cut-off


def test_reefer_excursion_burns_shelf_life(classify: ClassifyFn, config: SentinelConfig) -> None:
    delay = scoring.expected_delay(
        classify("reefer_telemetry", reefer("T-1", 12, minutes=180)), T0, config
    )
    assert delay.shelf_loss_days == pytest.approx(10 * 3 * 0.03)
    boxes = JsonContainerRepo(INDIA_CONTAINERS_PATH, container_ids=["DOM-01"]).list_active()
    assert "0.9 d burnt" in scoring.severity(delay, boxes, config).reason


def test_failed_residue_test_closes_that_market(
    classify: ClassifyFn, config: SentinelConfig
) -> None:
    failed = scoring.expected_delay(
        classify("compliance_feed", compliance("failed", market="NL")), T0, config
    )
    assert failed.market_unreachable and "NL closed" in failed.reason
    crash = scoring.expected_delay(classify("mandi_prices", mandi("Vashi", 38)), T0, config)
    assert (crash.days, crash.cost_impact_pct) == (0, 38.7)


# --- lifecycle and merging -----------------------------------------------------------------------


def test_a_rain_alert_does_not_corroborate_a_strike_rumour(india: Harness) -> None:
    india.send("news", news("Transporters threaten strike in Nashik"), minute=0)
    assert actions(india.send("weather", weather("orange", region="Nashik"), minute=3)) == [
        "create"
    ]
    assert india.view().status == "candidate"
    merged = india.send("transport_union", union("strike"), minute=6)
    assert actions(merged) == ["merge", "escalate"]
    assert (india.view().type, india.view().status) == ("TRUCK_STRIKE", "active")


@pytest.mark.parametrize(
    ("source", "start", "end"),
    [
        pytest.param(
            "transport_union", union("strike"), union("called_off"), id="strike-called-off"
        ),
        pytest.param(
            "road_authority",
            road("KASARA_GHAT", "closed"),
            road("KASARA_GHAT", "open"),
            id="road-reopens",
        ),
        pytest.param(
            "facility_status",
            facility("CS-NSK", "full", 0),
            facility("CS-NSK", "ok"),
            id="cold-store-frees-up",
        ),
        pytest.param("mandi_prices", mandi("Vashi", 38), mandi("Vashi", 60), id="prices-recover"),
        pytest.param(
            "reefer_telemetry",
            reefer("TRK-MH15-2201", 12, minutes=180),
            reefer("TRK-MH15-2201", 2, minutes=0, status="ok"),
            id="reefer-back-in-range",
        ),
    ],
)
def test_domestic_disruptions_resolve(
    india: Harness, source: str, start: dict[str, Any], end: dict[str, Any]
) -> None:
    india.send(source, start, minute=0)
    assert actions(india.send(source, end, minute=60)) == ["resolve"]
    assert india.view().status == "resolving"


# --- what the Sentinel adds for the UI and the Scenario Planner ----------------------------------


def test_layer_options_and_links() -> None:
    rig = make_rig(None, T0, books=INDIA_CONTAINERS_PATH)

    def send(event_id: str, source: str, payload: dict[str, Any]) -> None:
        rig.sentinel.ingest(
            {
                "event_id": event_id,
                "source": source,
                "received_at": T0.isoformat(),
                "payload": payload,
            }
        )

    states: list[dict[str, Any]] = []
    rig.sentinel.subscribe(lambda kind, data: states.append(data) if kind == "disruption" else None)
    send("E-1", "reefer_telemetry", reefer("TRK-MH15-2201", 12, minutes=180))
    send("E-2", "mandi_prices", mandi("Vashi", 38))
    send("E-3", "port_authority", port("AEJEA", "closed"))
    send("E-4", "port_authority", port("INNSA", "congested"))

    describe = rig.sentinel.describe
    assert [describe(d)["layer"] for d in ("DSR-0001", "DSR-0002", "DSR-0003", "DSR-0004")] == [
        "india",
        "india",
        "global",
        "india",
    ]
    assert [o["id"] for o in describe("DSR-0002")["options"]] == [
        "split_market",
        "hold_cold_store",
        "processor_salvage",
    ]
    assert describe("DSR-0001")["linked"] == ["DSR-0002"]  # both hold DOM-01
    # When the mandi crash appeared, the reefer card was re-sent with its new link.
    refreshed = [s for s in states if s["disruption"]["disruption_id"] == "DSR-0001"]
    assert refreshed[-1]["linked"] == ["DSR-0002"]
