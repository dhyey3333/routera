"""Spec sections 3 (step 4), 3.3 and 3.4: confidence, expected delay, severity."""

from __future__ import annotations

from typing import Any

import pytest

from agents.sentinel import scoring
from agents.sentinel.fakes import JsonContainerRepo
from agents.sentinel.models import ContainerRef
from agents.sentinel.replay import load_scenario
from agents.sentinel.settings import SentinelConfig
from agents.sentinel.tests.builders import (
    GULF_GRAPES,
    T0,
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


def book(*ids: str) -> list[ContainerRef]:
    return JsonContainerRepo(container_ids=ids).list_active()


# --- confidence -----------------------------------------------------------------------------------


def test_noisy_or_matches_the_spec_formula(classify: ClassifyFn, config: SentinelConfig) -> None:
    rumour = classify("news", news("Reports of unrest near Gulf ports"))
    lane = classify("carrier_advisory", advisory("delay", days=6, vessels=ARABIAN))
    closure = classify("port_authority", port("AEJEA", "closed"))
    assert scoring.reliability(rumour, config) == pytest.approx(0.40 * 0.75)
    assert scoring.confidence([rumour], config).value == pytest.approx(0.30)
    assert scoring.confidence([rumour, lane], config).value == pytest.approx(1 - 0.7 * 0.1)
    three = scoring.confidence([rumour, lane, closure], config)
    assert three.value == pytest.approx(1 - 0.7 * 0.1 * 0.05)
    assert "news Test Wire (SIM) 0.30" in three.reason
    assert "port_authority AEJEA 0.95" in three.reason


@pytest.mark.parametrize(
    ("source", "payload", "r"),
    [
        ("port_authority", port("AEJEA", "closed"), 0.95),
        ("tariff_feed", tariff("US", "0306", 10, 50), 0.95),
        ("carrier_advisory", advisory("suspend"), 0.90),
        ("weather", weather("red", port_code="INMUN"), 0.80),
        ("ais", ais("MV-X", WHEN, at(days=2)), 0.70),
        ("news", news("Jebel Ali port closed"), 0.40 * 0.90),
    ],
)
def test_source_reliability(
    classify: ClassifyFn, config: SentinelConfig, source: str, payload: dict[str, Any], r: float
) -> None:
    signal = classify(source, payload)
    assert scoring.confidence([signal], config).value == pytest.approx(r)


def test_one_outlet_repeating_itself_is_not_corroboration(
    classify: ClassifyFn, config: SentinelConfig
) -> None:
    first = classify("news", news("Jebel Ali port closed", outlet="Wire A (SIM)"))
    repeat = classify("news", news("Jebel Ali port closed", outlet="Wire A (SIM)"))
    other = classify("news", news("Jebel Ali port closed", outlet="Wire B (SIM)"))
    assert scoring.confidence([first, repeat], config).value == pytest.approx(0.36)
    assert scoring.confidence([first, repeat, other], config).value == pytest.approx(1 - 0.64**2)


# --- expected delay -------------------------------------------------------------------------------

DELAYS = [
    pytest.param(
        "port_authority", port("AEJEA", "closed"), 14, True, True, 0, id="closure-unknown"
    ),
    pytest.param(
        "port_authority",
        port("AEJEA", "closed", reopen=at(days=5)),
        5,
        False,
        False,
        0,
        id="closure-with-reopen",
    ),
    pytest.param("port_authority", port("INNSA", "congested"), 3, True, False, 0, id="congestion"),
    pytest.param("carrier_advisory", advisory("suspend"), 14, True, True, 0, id="lane-suspension"),
    pytest.param(
        "carrier_advisory",
        advisory("reroute", "CAPE", 25, lane="INNSA-NLRTM"),
        25,
        False,
        False,
        0,
        id="reroute-from-advisory",
    ),
    pytest.param(
        "carrier_advisory",
        advisory("reroute", "CAPE", 0, lane="INNSA-NLRTM"),
        18,
        True,
        False,
        0,
        id="reroute-default",
    ),
    pytest.param(
        "carrier_advisory",
        advisory("delay", days=6, vessels=ARABIAN),
        6,
        False,
        False,
        0,
        id="advisory-delay",
    ),
    pytest.param("ais", ais("MV-ARABIAN-02", WHEN, at(days=3)), 3, False, False, 0, id="ais"),
    pytest.param("tariff_feed", tariff("US", "0306", 10, 50), 0, False, False, 40, id="tariff"),
    pytest.param("weather", weather("orange", port_code="INMUN"), 1, True, False, 0, id="orange"),
    pytest.param("weather", weather("red", port_code="INMUN"), 3, True, False, 0, id="red"),
    pytest.param("news", news("Jebel Ali port closed"), 14, True, True, 0, id="news-closure"),
]


@pytest.mark.parametrize(("source", "payload", "days", "estimate", "unreachable", "cost"), DELAYS)
def test_expected_delay_per_type(
    classify: ClassifyFn,
    config: SentinelConfig,
    source: str,
    payload: dict[str, Any],
    days: float,
    estimate: bool,
    unreachable: bool,
    cost: float,
) -> None:
    delay = scoring.expected_delay(classify(source, payload), T0, config)
    assert delay.days == pytest.approx(days)
    assert (delay.is_estimate, delay.market_unreachable) == (estimate, unreachable)
    assert delay.cost_impact_pct == cost
    assert delay.reason


# --- severity -------------------------------------------------------------------------------------


def test_gulf_closure_is_severity_5_urgent(classify: ClassifyFn, config: SentinelConfig) -> None:
    delay = scoring.expected_delay(classify("port_authority", port("AEJEA", "closed")), T0, config)
    score = scoring.severity(delay, book(*GULF_GRAPES), config)
    assert (score.level, score.priority) == (5, "urgent")
    assert score.raw == pytest.approx(0.55 + 0.20 + 0.15)
    assert "median 13.25 d left" in score.reason
    assert "market unreachable" in score.reason


def test_gulf_lane_delay_is_about_3(classify: ClassifyFn, config: SentinelConfig) -> None:
    signal = classify("carrier_advisory", advisory("delay", days=6, vessels=ARABIAN))
    score = scoring.severity(scoring.expected_delay(signal, T0, config), book(*GULF_GRAPES), config)
    assert (score.level, score.priority) == (3, "normal")


def test_tariff_spike_is_3_to_4(classify: ClassifyFn, config: SentinelConfig) -> None:
    delay = scoring.expected_delay(
        classify("tariff_feed", tariff("US", "0306", 10, 50)), T0, config
    )
    shrimp = book(*(f"SHR-{i:02d}" for i in range(1, 10)))
    score = scoring.severity(delay, shrimp, config)
    assert 3 <= score.level <= 4
    assert score.raw == pytest.approx(0.25 * 40 / 50 + 0.20)
    assert "days_left unknown for 9" in score.reason


def test_congestion_is_1_to_2(classify: ClassifyFn, config: SentinelConfig) -> None:
    delay = scoring.expected_delay(
        classify("port_authority", port("INNSA", "congested")), T0, config
    )
    assert 1 <= scoring.severity(delay, book("POM-01"), config).level <= 2


def test_red_sea_reroute_matches_the_hand_derivation(
    classify: ClassifyFn, config: SentinelConfig
) -> None:
    signal = classify("carrier_advisory", advisory("reroute", "CAPE", 25, lane="INNSA-NLRTM"))
    affected = book("GRP-13", "GRP-14", "GRP-16", "SHR-10")
    score = scoring.severity(scoring.expected_delay(signal, T0, config), affected, config)
    assert score.level == 3
    assert score.raw == pytest.approx(0.55 * 25 / 30 + 0.20 * 78 / 200)


def test_no_containers_falls_back_to_default_days_left(
    classify: ClassifyFn, config: SentinelConfig
) -> None:
    delay = scoring.expected_delay(
        classify("port_authority", port("INNSA", "congested")), T0, config
    )
    score = scoring.severity(delay, [], config)
    assert score.level == 1
    assert "assumed 30 (estimated)" in score.reason


def test_rounding_is_half_up_not_bankers(classify: ClassifyFn, config: SentinelConfig) -> None:
    assert [scoring.round_half_up(x) for x in (0.5, 1.5, 2.5, 2.49)] == [1, 2, 3, 2]
    weights = config.severity.model_copy(update={"w_shelf": 0, "w_cost": 0.625, "w_scale": 0})
    cost_only = config.model_copy(update={"severity": weights})
    delay = scoring.expected_delay(
        classify("tariff_feed", tariff("US", "0306", 10, 60)), T0, config
    )
    score = scoring.severity(delay, [], cost_only)
    assert score.raw == 0.625  # 4 × raw = 2.5 exactly
    assert (score.level, score.priority) == (4, "urgent")  # round() would have given 3


def test_scenario_books_have_the_days_left_the_derivations_assume() -> None:
    gulf = {c.container_id: c for c in book(*load_scenario("gulf_closure").container_ids)}
    assert sorted(gulf[cid].days_left for cid in GULF_GRAPES)[5:7] == [13, 13.5]
