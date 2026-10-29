"""Shared test helpers: feed payload builders, a tracker harness, and a scenario runner."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from agents.sentinel.classify import Classifier
from agents.sentinel.fakes import (
    DEFAULT_CONTAINERS_PATH,
    JsonContainerRepo,
    JsonNetworkRepo,
    SimClock,
)
from agents.sentinel.merge import Decision, DisruptionTracker
from agents.sentinel.models import Disruption
from agents.sentinel.normalise import Gazetteer, NormalisationError, Normaliser
from agents.sentinel.replay import Scenario, ScenarioStep, check_disruption, load_scenario
from agents.sentinel.settings import SentinelConfig

T0 = datetime(2026, 3, 2, 6, 0, tzinfo=UTC)
WHEN = "2026-03-02T06:00:00Z"
LATER = "2026-03-05T06:00:00Z"
GULF_GRAPES = [f"GRP-{i:02d}" for i in range(1, 13)]


def at(*, minutes: float = 0, hours: float = 0, days: float = 0) -> str:
    return (T0 + timedelta(minutes=minutes, hours=hours, days=days)).isoformat()


# --- payloads -------------------------------------------------------------------------------------


def port(
    code: str, status: str, reopen: str | None = None, *, effective: str = WHEN
) -> dict[str, Any]:
    return {
        "port_code": code,
        "status": status,
        "reason": "test",
        "effective_from": effective,
        "expected_reopen": reopen,
    }


def advisory(
    action: str,
    via: str | None = None,
    days: float = 0,
    vessels: list[str] | None = None,
    lane: str = "INNSA-AEJEA",
    *,
    carrier: str = "Test Line (SIM)",
) -> dict[str, Any]:
    return {
        "carrier": carrier,
        "affected_lane": lane,
        "action": action,
        "via": via,
        "added_transit_days": days,
        "vessel_ids": vessels or [],
    }


def ais(vessel: str, expected: str, new: str) -> dict[str, Any]:
    return {
        "vessel_id": vessel,
        "expected_eta": expected,
        "new_eta": new,
        "position": {"lat": 25.1, "lon": 57.4},
        "speed_knots": 5,
        "deviation_flag": True,
    }


def tariff(
    country: str, hs: str, old: float, new: float, *, effective: str = WHEN
) -> dict[str, Any]:
    return {
        "importing_country": country,
        "hs_code": hs,
        "old_rate_pct": old,
        "new_rate_pct": new,
        "effective_from": effective,
    }


def weather(
    level: str,
    *,
    port_code: str | None = None,
    region: str | None = None,
    valid_from: str = WHEN,
    valid_to: str | None = LATER,
) -> dict[str, Any]:
    return {
        "region": region,
        "port_code": port_code,
        "alert_level": level,
        "hazard": "cyclone",
        "valid_from": valid_from,
        "valid_to": valid_to,
    }


def news(headline: str, body: str = "", outlet: str = "Test Wire (SIM)") -> dict[str, Any]:
    return {"headline": headline, "body": body, "outlet": outlet}


# --- tracker harness ------------------------------------------------------------------------------


class Harness:
    """Runs payloads through normalise -> classify -> DisruptionTracker on a simulated clock."""

    def __init__(
        self,
        config: SentinelConfig,
        container_ids: list[str] | None = None,
        books: Path | str = DEFAULT_CONTAINERS_PATH,
    ) -> None:
        network = JsonNetworkRepo().network()
        gazetteer = Gazetteer(config.locations, network.codes())
        self.clock = SimClock(T0)
        self.normaliser = Normaliser(config, gazetteer)
        self.classifier = Classifier(config, gazetteer)
        repo = JsonContainerRepo(books, container_ids=container_ids)
        self.tracker = DisruptionTracker(config, gazetteer, repo, self.clock, network)
        self._ids = itertools.count(1)

    def _set_time(self, minutes: float) -> None:
        self.clock.set(T0 + timedelta(minutes=minutes))

    def send(
        self,
        source: str,
        payload: dict[str, Any],
        *,
        minute: float | None = None,
        event_id: str | None = None,
    ) -> list[Decision]:
        if minute is not None:
            self._set_time(minute)
        raw = {
            "event_id": event_id or f"H-{next(self._ids):03d}",
            "source": source,
            "received_at": self.clock.now().isoformat(),
            "kind_hint": None,
            "payload": payload,
        }
        return self.tracker.add(self.classifier.classify(self.normaliser.normalise(raw)))

    def tick(self, *, minute: float) -> list[Decision]:
        self._set_time(minute)
        return self.tracker.tick()

    def view(self, disruption_id: str = "DSR-0001") -> Disruption:
        return self.tracker.get(disruption_id).disruption


def actions(decisions: list[Decision]) -> list[str]:
    return [d.action for d in decisions]


# --- scenario runner and expected-file checks -----------------------------------------------------


@dataclass
class StepResult:
    step: ScenarioStep
    decisions: list[Decision] = field(default_factory=list)
    rejected: NormalisationError | None = None
    views: dict[str, Disruption] = field(default_factory=dict)  # every incident after the step


@dataclass
class ScenarioRun:
    scenario: Scenario
    tracker: DisruptionTracker
    steps: list[StepResult]


def run_scenario(config: SentinelConfig, name: str) -> ScenarioRun:
    scenario = load_scenario(name)
    network = JsonNetworkRepo().network()
    gazetteer = Gazetteer(config.locations, network.codes())
    clock = SimClock(scenario.start)
    normaliser = Normaliser(config, gazetteer)
    classifier = Classifier(config, gazetteer)
    repo = JsonContainerRepo(scenario.book_path, container_ids=scenario.container_ids)
    tracker = DisruptionTracker(config, gazetteer, repo, clock, network)
    results = []
    for step in scenario.steps:
        clock.set(scenario.at(step))
        result = StepResult(step)
        if step.tick:
            result.decisions = tracker.tick()
        else:
            try:
                signal = classifier.classify(normaliser.normalise(scenario.raw_event(step)))
            except NormalisationError as exc:
                result.rejected = exc
            else:
                result.decisions = tracker.add(signal)
        result.views = {i.disruption_id: i.disruption for i in tracker.incidents}
        results.append(result)
    return ScenarioRun(scenario, tracker, results)


def check_view(view: Disruption, expected: dict[str, Any]) -> None:
    """Assert a Disruption against a labelled row of data/expected/*.json (keys present only)."""
    problems = check_disruption(view, expected)
    assert not problems, problems
