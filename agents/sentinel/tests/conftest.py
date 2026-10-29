from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

import pytest

from agents.sentinel.classify import Classifier
from agents.sentinel.fakes import JsonNetworkRepo
from agents.sentinel.models import LLMClient, Signal
from agents.sentinel.normalise import Gazetteer, Normaliser
from agents.sentinel.settings import SentinelConfig, load_config
from agents.sentinel.tests.builders import T0, Harness

RawFactory = Callable[[str, dict[str, Any]], dict[str, Any]]
ClassifyFn = Callable[..., Signal]


@pytest.fixture(scope="session")
def config() -> SentinelConfig:
    return load_config()


@pytest.fixture(scope="session")
def gazetteer(config: SentinelConfig) -> Gazetteer:
    return Gazetteer(config.locations, JsonNetworkRepo().network().codes())


@pytest.fixture
def normaliser(config: SentinelConfig, gazetteer: Gazetteer) -> Normaliser:
    return Normaliser(config, gazetteer)


@pytest.fixture
def make_raw() -> RawFactory:
    """A feed message as a dict, stamped at T0, with a fresh event_id."""
    ids = itertools.count(1)

    def _make(source: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "event_id": f"T-{next(ids):03d}",
            "source": source,
            "received_at": T0.isoformat(),
            "kind_hint": None,
            "payload": payload,
        }

    return _make


@pytest.fixture
def classify(
    config: SentinelConfig, gazetteer: Gazetteer, normaliser: Normaliser, make_raw: RawFactory
) -> ClassifyFn:
    """normalise + classify one payload; pass llm=... to use an LLM client."""

    def _run(source: str, payload: dict[str, Any], llm: LLMClient | None = None) -> Signal:
        classifier = Classifier(config, gazetteer, llm=llm)
        return classifier.classify(normaliser.normalise(make_raw(source, payload)))

    return _run


@pytest.fixture
def harness(config: SentinelConfig) -> Harness:
    """A tracker over the full simulated container book, clock at T0."""
    return Harness(config)
