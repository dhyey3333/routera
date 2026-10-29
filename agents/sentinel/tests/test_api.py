"""api.py: REST endpoints, the SSE stream, replays in the background, and the offline UI."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import TypeAdapter

from agents.sentinel.api import STATIC_DIR, SentinelService, create_app
from agents.sentinel.models import Disruption
from agents.sentinel.replay import SCENARIO_NAMES
from agents.sentinel.tests.builders import at, news, port

DISRUPTIONS = TypeAdapter(list[Disruption])


@pytest.fixture
def service() -> Iterator[SentinelService]:
    service = SentinelService()
    yield service
    service.stop_replay()


@pytest.fixture
def client(service: SentinelService) -> TestClient:
    return TestClient(create_app(service))


def event(source: str, payload: dict[str, Any], event_id: str, **extra: Any) -> dict[str, Any]:
    return {"event_id": event_id, "source": source, "payload": payload, **extra}


def read_events(client: TestClient, limit: int) -> list[tuple[str, Any]]:
    """Read the first `limit` SSE events. (TestClient buffers the whole response, so anything
    the stream should carry must already be set in motion, e.g. via service.autostart.)"""
    events: list[tuple[str, Any]] = []
    with client.stream("GET", f"/sentinel/stream?limit={limit}") as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        kind = None
        for line in response.iter_lines():
            if line.startswith("event: "):
                kind = line.removeprefix("event: ")
            elif line.startswith("data: "):
                events.append((kind, json.loads(line.removeprefix("data: "))))
    return events


# --- REST ---------------------------------------------------------------------------------------


def test_post_event_ingests_and_returns_contract_objects(client: TestClient) -> None:
    response = client.post(
        "/sentinel/events", json=event("port_authority", port("AEJEA", "closed"), "A-1")
    )
    assert response.status_code == 200
    body = response.json()
    assert body["simulated"] is True
    assert body["signal"]["status"] == "accepted"
    [disruption] = DISRUPTIONS.validate_python(body["disruptions"])
    assert (disruption.type, disruption.status, disruption.severity) == (
        "PORT_CLOSURE",
        "active",
        5,
    )


def test_malformed_event_is_rejected_with_422_and_audited(
    client: TestClient, service: SentinelService
) -> None:
    response = client.post(
        "/sentinel/events", json=event("tariff_feed", {"importing_country": "US"}, "A-2")
    )
    assert response.status_code == 422
    assert "new_rate_pct" in response.json()["signal"]["reason"]
    assert service.audit[-1]["action"] == "reject"


def test_noise_is_accepted_but_ignored(client: TestClient) -> None:
    response = client.post("/sentinel/events", json=event("news", news("Grape prices firm"), "A-3"))
    assert response.status_code == 200
    assert response.json()["signal"]["status"] == "ignored"
    assert response.json()["disruptions"] == []


def test_received_at_defaults_to_the_simulated_clock(
    client: TestClient, service: SentinelService
) -> None:
    clock = service.rig.clock.now()
    client.post("/sentinel/events", json=event("port_authority", port("AEJEA", "closed"), "A-4"))
    assert service.signals[-1]["received_at"] == clock.isoformat()


def test_disruptions_endpoint_scopes(client: TestClient) -> None:
    client.post("/sentinel/events", json=event("news", news("Singapore port shut"), "A-5"))
    client.post("/sentinel/events", json=event("port_authority", port("AEJEA", "closed"), "A-6"))
    active = DISRUPTIONS.validate_python(client.get("/sentinel/disruptions").json())
    everything = DISRUPTIONS.validate_python(client.get("/sentinel/disruptions?scope=all").json())
    assert [d.location for d in active] == ["AEJEA"]
    assert [d.location for d in everything] == ["SGSIN", "AEJEA"]


def test_tick_runs_the_lifecycle(client: TestClient) -> None:
    client.post("/sentinel/events", json=event("news", news("Singapore port shut"), "A-7"))
    later = event("news", news("Grape prices firm"), "A-8", received_at=at(hours=25))
    client.post("/sentinel/events", json=later)  # moves the simulated clock on
    assert client.post("/sentinel/tick").status_code == 200
    state = client.get("/sentinel/state").json()
    assert state["disruptions"][0]["ignored"] is True
    assert state["audit"][-1]["action"] == "ignore"


def test_replay_runs_in_the_background_and_resets_state(
    client: TestClient, service: SentinelService
) -> None:
    client.post("/sentinel/events", json=event("port_authority", port("QAHMD", "closed"), "A-9"))
    response = client.post("/sentinel/replay/gulf_closure?speed=0")
    assert response.status_code == 202
    assert response.json() == {
        "scenario": "gulf_closure",
        "speed": 0,
        "steps": 8,
        "simulated": True,
    }
    service.wait_for_replay()
    state = client.get("/sentinel/state").json()
    assert state["scenario"]["name"] == "gulf_closure"
    assert state["replay"] == {"running": False, "speed": 0, "step": 8, "steps": 8}
    assert len(state["signals"]) == 7  # the QAHMD event from before the replay is gone
    by_id = {d["disruption"]["disruption_id"]: d for d in state["disruptions"]}
    assert by_id["DSR-0001"]["disruption"]["type"] == "PORT_CLOSURE"
    assert by_id["DSR-0001"]["last_topic"] == "disruption.updated"
    assert by_id["DSR-0003"]["ignored"] is True
    assert state["published"] == 3
    active = client.get("/sentinel/disruptions").json()
    assert [d["disruption_id"] for d in active] == ["DSR-0001", "DSR-0002"]


@pytest.mark.parametrize(
    ("url", "status"),
    [("/sentinel/replay/atlantis", 404), ("/sentinel/replay/gulf_closure?speed=-1", 422)],
)
def test_replay_rejects_bad_requests(client: TestClient, url: str, status: int) -> None:
    assert client.post(url).status_code == status


def test_scenarios_list_feeds_the_dropdown(client: TestClient) -> None:
    names = [s["name"] for s in client.get("/sentinel/scenarios").json()]
    assert names == list(SCENARIO_NAMES)
    assert names[0] == "nashik_corridor_strike"  # India-inside first


# --- SSE ----------------------------------------------------------------------------------------


def test_stream_starts_with_a_snapshot(client: TestClient) -> None:
    [(kind, snapshot)] = read_events(client, limit=1)
    assert kind == "snapshot"
    assert snapshot["simulated"] is True
    assert {"signals", "disruptions", "audit", "replay", "scenarios", "clock"} <= set(snapshot)


def test_stream_carries_a_live_replay(client: TestClient, service: SentinelService) -> None:
    service.autostart = ("gulf_closure", 0)  # starts inside subscribe, after the queue exists
    events = read_events(client, limit=8)
    kinds = [kind for kind, _ in events]
    assert kinds[:2] == ["snapshot", "reset"]
    assert kinds[2:] == ["step", "signal", "decision", "disruption", "step", "signal"]
    assert events[3][1]["event_id"] == "GC-01"
    assert events[5][1]["disruption"]["status"] == "candidate"


def test_demo_replay_starts_when_the_first_viewer_connects(
    client: TestClient, service: SentinelService
) -> None:
    service.autostart = ("gulf_closure", 0)
    read_events(client, limit=1)
    service.wait_for_replay()
    assert service.autostart is None
    assert service.snapshot()["scenario"]["name"] == "gulf_closure"
    read_events(client, limit=1)  # a second viewer does not restart it
    assert service.snapshot()["replay"]["step"] == 8


# --- UI ------------------------------------------------------------------------------------------


def test_index_serves_the_offline_ui(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    html = response.text
    assert "SIMULATED FEEDS" in html
    assert 'new EventSource("/sentinel/stream")' in html
    for speed in ("1×", "10×", "60×"):
        assert speed in html


def test_ui_loads_nothing_from_the_network() -> None:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert not re.search(r"""(src|href)\s*=\s*["']?(https?:)?//""", html)
    assert "@import" not in html
    assert not re.search(r"url\(\s*['\"]?https?:", html)
