"""Spec section 8: classification table, news-alone rule, offline keyword fallback, LLM safety."""

from __future__ import annotations

from typing import Any

import pytest

from agents.sentinel.classify import Classifier
from agents.sentinel.fakes import ScriptedLLM
from agents.sentinel.normalise import Gazetteer, NormalisationError, Normaliser
from agents.sentinel.replay import SCENARIO_NAMES, load_expected, load_scenario
from agents.sentinel.settings import SentinelConfig
from agents.sentinel.tests.builders import WHEN, advisory, ais, news, port, tariff, weather
from agents.sentinel.tests.conftest import ClassifyFn, RawFactory

# --- 3.1 rule table ------------------------------------------------------------------------------

RULE_TABLE = [
    pytest.param(
        "port_authority", port("AEJEA", "closed"), "PORT_CLOSURE", "AEJEA", "", id="port-closed"
    ),
    pytest.param(
        "port_authority",
        port("INNSA", "congested"),
        "PORT_CONGESTION",
        "INNSA",
        "",
        id="port-congested",
    ),
    pytest.param(
        "carrier_advisory",
        advisory("suspend"),
        "LANE_SUSPENSION",
        "INNSA-AEJEA",
        "INNSA-AEJEA",
        id="lane-suspend",
    ),
    pytest.param(
        "carrier_advisory",
        advisory("reroute", via="CAPE", days=25, lane="INNSA-NLRTM"),
        "ROUTE_REROUTE",
        "SUEZ",
        "INNSA-NLRTM",
        id="reroute-via-cape",
    ),
    pytest.param(
        "carrier_advisory",
        advisory("delay", days=6, vessels=["MV-ARABIAN-01"]),
        "VESSEL_DELAY",
        "INNSA-AEJEA",
        "INNSA-AEJEA",
        id="advisory-delay",
    ),
    pytest.param(
        "ais",
        ais("MV-ARABIAN-02", WHEN, "2026-03-05T06:00:00Z"),
        "VESSEL_DELAY",
        "MV-ARABIAN-02",
        "MV-ARABIAN-02",
        id="ais-plus-3d",
    ),
    pytest.param(
        "ais",
        ais("MV-ARABIAN-02", WHEN, "2026-03-03T06:00:00Z"),
        "VESSEL_DELAY",
        "MV-ARABIAN-02",
        "MV-ARABIAN-02",
        id="ais-exactly-24h",
    ),
    pytest.param(
        "tariff_feed",
        tariff("US", "0306", 10, 50),
        "TARIFF_CHANGE",
        "US:0306",
        "US+0306",
        id="tariff-plus-40",
    ),
    pytest.param(
        "tariff_feed",
        tariff("US", "0806", 10, 15),
        "TARIFF_CHANGE",
        "US:0806",
        "US+0806",
        id="tariff-exactly-5pts",
    ),
    pytest.param(
        "weather",
        weather("orange", port_code="INMUN"),
        "WEATHER_ALERT",
        "INMUN",
        "",
        id="weather-orange-port",
    ),
    pytest.param(
        "weather",
        weather("red", region="Konkan coast"),
        "WEATHER_ALERT",
        "KONKAN_COAST",
        "",
        id="weather-red-region",
    ),
]


@pytest.mark.parametrize(("source", "payload", "dtype", "location", "scope"), RULE_TABLE)
def test_rule_table_maps_payload_to_type_and_location(
    classify: ClassifyFn,
    source: str,
    payload: dict[str, Any],
    dtype: str,
    location: str,
    scope: str,
) -> None:
    signal = classify(source, payload)
    assert signal.role == "disruption"
    assert (signal.type, signal.location, signal.scope) == (dtype, location, scope)
    assert signal.classified_by == "rules"
    assert signal.type_confidence == 1.0
    assert signal.reason


NOT_DISRUPTIONS = [
    pytest.param("port_authority", port("AEJEA", "open"), "resolution", id="port-reopened"),
    pytest.param("carrier_advisory", advisory("reroute", via=None), "noise", id="reroute-no-via"),
    pytest.param("carrier_advisory", advisory("delay", days=0.5), "noise", id="advisory-delay-12h"),
    pytest.param(
        "ais", ais("MV-ARABIAN-02", WHEN, "2026-03-03T05:00:00Z"), "resolution", id="ais-23h"
    ),
    pytest.param("tariff_feed", tariff("US", "0306", 10, 14), "noise", id="tariff-4pts"),
    pytest.param("tariff_feed", tariff("US", "0306", 50, 10), "resolution", id="tariff-reversal"),
    pytest.param("weather", weather("yellow", port_code="INMUN"), "noise", id="weather-yellow"),
]


@pytest.mark.parametrize(("source", "payload", "role"), NOT_DISRUPTIONS)
def test_below_threshold_and_reopen_signals_are_not_disruptions(
    classify: ClassifyFn, source: str, payload: dict[str, Any], role: str
) -> None:
    signal = classify(source, payload)
    assert (signal.role, signal.type, signal.fingerprint) == (role, None, None)
    assert signal.reason


def test_advisory_delay_keeps_its_vessels_for_matching(classify: ClassifyFn) -> None:
    signal = classify(
        "carrier_advisory", advisory("delay", days=6, vessels=["mv-arabian-01", "MV-ARABIAN-02"])
    )
    assert signal.vessel_ids == ["MV-ARABIAN-01", "MV-ARABIAN-02"]
    assert signal.fingerprint == ("VESSEL_DELAY", "INNSA-AEJEA", "INNSA-AEJEA")


# --- normalisation -------------------------------------------------------------------------------


def test_codes_are_canonicalised(classify: ClassifyFn) -> None:
    assert classify("port_authority", port(" aejea ", "Closed")).location == "AEJEA"
    lane = classify("carrier_advisory", advisory("suspend", lane="innsa - aejea"))
    assert lane.location == "INNSA-AEJEA"
    assert set(lane.footprint) >= {"INNSA", "AEJEA", "INNSA-AEJEA"}
    reroute = classify(
        "carrier_advisory",
        advisory("reroute", via="Cape of Good Hope", days=25, lane="INNSA-NLRTM"),
    )
    assert reroute.payload.via == "CAPE"
    assert (reroute.type, reroute.location) == ("ROUTE_REROUTE", "SUEZ")
    assert {"SUEZ", "BAB_EL_MANDEB"} <= set(reroute.footprint)


def test_naive_timestamps_are_read_as_utc(normaliser: Normaliser, make_raw: RawFactory) -> None:
    raw = make_raw(
        "port_authority", port("AEJEA", "closed") | {"effective_from": "2026-03-02T06:08"}
    )
    raw["received_at"] = "2026-03-02T06:08:00"
    signal = normaliser.normalise(raw)
    assert signal.received_at.utcoffset() is not None
    assert signal.payload.effective_from.utcoffset() is not None


def test_weather_region_expands_to_member_ports(classify: ClassifyFn) -> None:
    signal = classify("weather", weather("red", region="gulf"))
    assert signal.location == "GULF"
    assert "AEJEA" in signal.footprint


MALFORMED = [
    pytest.param("twitter", news("hello"), "source", id="unknown-source"),
    pytest.param("port_authority", port("AEJEA", "maybe"), "status", id="bad-port-status"),
    pytest.param(
        "port_authority", port("AE-JEA", "closed"), "unknown location", id="bad-port-code"
    ),
    pytest.param(
        "carrier_advisory",
        advisory("suspend", lane="INNSA"),
        "ORIGIN-DESTINATION",
        id="lane-without-destination",
    ),
    pytest.param(
        "carrier_advisory",
        advisory("delay", days=-2),
        "added_transit_days",
        id="negative-transit-days",
    ),
    pytest.param(
        "ais", {"vessel_id": "MV-X", "expected_eta": WHEN}, "new_eta", id="ais-missing-eta"
    ),
    pytest.param(
        "tariff_feed",
        {k: v for k, v in tariff("US", "0306", 10, 50).items() if k != "new_rate_pct"},
        "new_rate_pct",
        id="tariff-missing-rate",
    ),
    pytest.param("tariff_feed", tariff("US", "shrimp", 10, 50), "hs_code", id="tariff-bad-hs"),
    pytest.param(
        "tariff_feed", tariff("USA", "0306", 10, 50), "importing_country", id="tariff-bad-country"
    ),
    pytest.param("weather", weather("red"), "region or a port_code", id="weather-no-place"),
    pytest.param("news", news(""), "headline", id="news-empty-headline"),
]


@pytest.mark.parametrize(("source", "payload", "reason_part"), MALFORMED)
def test_malformed_events_are_rejected_with_a_reason(
    normaliser: Normaliser,
    make_raw: RawFactory,
    source: str,
    payload: dict[str, Any],
    reason_part: str,
) -> None:
    with pytest.raises(NormalisationError) as caught:
        normaliser.normalise(make_raw(source, payload))
    assert reason_part in caught.value.reason
    assert caught.value.event_id is not None


def test_event_without_received_at_is_rejected(
    normaliser: Normaliser, make_raw: RawFactory
) -> None:
    raw = make_raw("port_authority", port("AEJEA", "closed"))
    del raw["received_at"]
    with pytest.raises(NormalisationError, match="received_at"):
        normaliser.normalise(raw)


def test_hs_code_given_as_number_is_rejected(normaliser: Normaliser, make_raw: RawFactory) -> None:
    payload = tariff("US", "0306", 10, 50) | {"hs_code": 306}  # leading zero already lost
    with pytest.raises(NormalisationError, match="hs_code"):
        normaliser.normalise(make_raw("tariff_feed", payload))


# --- news: offline keyword fallback --------------------------------------------------------------

NEWS_CASES = [
    pytest.param(
        "Reports of unrest near Gulf ports", "PORT_CLOSURE", "GULF", 0.75, id="vague-hedged-weak"
    ),
    pytest.param(
        "Jebel Ali port closed as Gulf tensions escalate",
        "PORT_CLOSURE",
        "AEJEA",
        0.90,
        id="specific-port-beats-region",
    ),
    pytest.param(
        "Singapore port shut, sources claim", "PORT_CLOSURE", "SGSIN", 0.75, id="rumour-hedged"
    ),
    pytest.param(
        "US slaps 50% duty on Indian shrimp",
        "TARIFF_CHANGE",
        "US:0306",
        0.90,
        id="tariff-country-and-hs",
    ),
    pytest.param(
        "Washington weighs tariffs on imports",
        "TARIFF_CHANGE",
        "US",
        0.75,
        id="tariff-country-only",
    ),
    pytest.param(
        "Ships rerouted around the Cape of Good Hope",
        "ROUTE_REROUTE",
        "SUEZ",
        0.90,
        id="reroute-names-only-the-detour",
    ),
    pytest.param(
        "Carriers divert from the Red Sea", "ROUTE_REROUTE", "RED_SEA", 0.90, id="reroute-region"
    ),
    pytest.param("Cyclone warning for Mundra", "WEATHER_ALERT", "INMUN", 0.90, id="weather"),
    pytest.param(
        "Nhava Sheva congestion worsens", "PORT_CONGESTION", "INNSA", 0.90, id="congestion"
    ),
]


@pytest.mark.parametrize(("headline", "dtype", "location", "confidence"), NEWS_CASES)
def test_news_keyword_fallback_works_offline(
    classify: ClassifyFn, headline: str, dtype: str, location: str, confidence: float
) -> None:
    signal = classify("news", news(headline))
    assert (signal.type, signal.location) == (dtype, location)
    assert signal.type_confidence == pytest.approx(confidence)
    assert signal.classified_by == "keywords"


@pytest.mark.parametrize(
    "headline",
    [
        pytest.param("Grape prices firm at Nashik mandi", id="no-disruption-terms"),
        pytest.param("Port closed, officials say", id="no-known-place"),
        pytest.param("Duties raised again", id="tariff-without-country"),
    ],
)
def test_news_without_type_or_place_is_noise(classify: ClassifyFn, headline: str) -> None:
    signal = classify("news", news(headline))
    assert (signal.role, signal.type) == ("noise", None)
    assert signal.reason.startswith("keywords:")


def test_one_news_report_can_never_reach_the_confirm_threshold(config: SentinelConfig) -> None:
    best_case = config.reliability.news * max(
        config.news.strong_confidence, config.news.hedged_confidence, 1.0
    )
    assert best_case < config.lifecycle.confirm_threshold
    assert "news" not in config.lifecycle.authoritative_sources


# --- news: LLM path and LLM safety ---------------------------------------------------------------


def test_valid_llm_reply_is_used(classify: ClassifyFn) -> None:
    llm = ScriptedLLM(
        news=[
            {
                "type": "PORT_CLOSURE",
                "location": "Jebel Ali",
                "confidence": 0.8,
                "reason": "extra keys are ignored",
            }
        ]
    )
    signal = classify("news", news("Something happened at the port"), llm=llm)
    assert (signal.type, signal.location, signal.classified_by) == ("PORT_CLOSURE", "AEJEA", "llm")
    assert signal.type_confidence == pytest.approx(0.8)
    assert llm.calls == ["classify_news"]


def test_llm_reply_with_market_location(classify: ClassifyFn) -> None:
    llm = ScriptedLLM(news=[{"type": "TARIFF_CHANGE", "location": "us:0306", "confidence": 0.7}])
    signal = classify("news", news("Shrimp duty news"), llm=llm)
    assert (signal.location, signal.scope) == ("US:0306", "US+0306")


def test_llm_can_say_not_a_disruption(classify: ClassifyFn) -> None:
    llm = ScriptedLLM(news=[{"type": "NONE", "location": None, "confidence": 0.95}])
    signal = classify("news", news("Jebel Ali port closed"), llm=llm)
    assert (signal.role, signal.type, signal.classified_by) == ("noise", None, "llm")


BAD_LLM_REPLIES = [
    pytest.param({"type": "EARTHQUAKE", "location": "AEJEA", "confidence": 0.9}, id="unknown-type"),
    pytest.param(
        {"type": "PORT_CLOSURE", "location": "AEJEA", "confidence": 1.7},
        id="confidence-out-of-range",
    ),
    pytest.param({"type": "PORT_CLOSURE", "location": "AEJEA"}, id="missing-confidence"),
    pytest.param(
        {"type": "PORT_CLOSURE", "location": "Atlantis", "confidence": 0.9}, id="unknown-location"
    ),
    pytest.param("PORT_CLOSURE at Jebel Ali, I think", id="prose-not-json"),
    pytest.param(TimeoutError("LLM call timed out after 5 s"), id="timeout"),
    pytest.param(RuntimeError("HTTP 503"), id="server-error"),
]


@pytest.mark.parametrize("reply", BAD_LLM_REPLIES)
def test_malformed_or_failed_llm_reply_falls_back_to_keywords(
    classify: ClassifyFn, reply: Any
) -> None:
    signal = classify("news", news("Jebel Ali port closed"), llm=ScriptedLLM(news=[reply]))
    assert (signal.type, signal.location) == ("PORT_CLOSURE", "AEJEA")
    assert signal.classified_by == "keywords"
    assert signal.reason.startswith("LLM classification rejected")


def test_llm_is_never_asked_about_structured_events(classify: ClassifyFn) -> None:
    llm = ScriptedLLM()
    classify("port_authority", port("AEJEA", "closed"), llm=llm)
    classify("tariff_feed", tariff("US", "0306", 10, 50), llm=llm)
    assert llm.calls == []


# --- scenario data classifies as labelled --------------------------------------------------------


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_scenario_events_classify_as_labelled(
    config: SentinelConfig, gazetteer: Gazetteer, normaliser: Normaliser, name: str
) -> None:
    scenario = load_scenario(name)
    labels = {row["event_id"]: row for row in load_expected(name)["timeline"] if "event_id" in row}
    classifier = Classifier(config, gazetteer)
    events = [step for step in scenario.steps if step.event is not None]
    assert {step.event["event_id"] for step in events} == set(labels)  # type: ignore[index]
    for step in events:
        label = labels[step.event["event_id"]]  # type: ignore[index]
        if label.get("rejected"):
            with pytest.raises(NormalisationError) as caught:
                normaliser.normalise(scenario.raw_event(step))
            assert label["reason_contains"] in caught.value.reason
            continue
        signal = classifier.classify(normaliser.normalise(scenario.raw_event(step)))
        assert {"type": signal.type, "location": signal.location} == label["classified"], step.note
