"""The map: real geometry for every route and booked path, and live options for any container."""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agents.sentinel.fakes import JsonContainerRepo
from routera.app import RouteraService, create_routera_app
from routera.geo import booked_key, load_geo
from routera.live_feeds import LIVE_BOOKS, TrafficFeed, status
from routera.reroute import load_routes


def test_every_planner_route_and_every_booked_path_has_geometry() -> None:
    geo = load_geo()
    assert {r.id for r in load_routes()} <= set(geo["routes"])
    for c in JsonContainerRepo(LIVE_BOOKS).list_active():
        assert booked_key(c) in geo["booked"], c.container_id
    for path in [*geo["routes"].values(), *geo["booked"].values()]:
        assert path["stretches"] and all(len(s["path"]) >= 2 for s in path["stretches"])


def test_the_paths_are_real_road_and_sea_distances() -> None:
    routes = load_geo()["routes"]
    road, sea = routes["N-JNPT-LK"]["stretches"]
    assert road["mode"] == "road" and 150 < road["km"] < 260  # Nashik to JNPT by road
    assert sea["mode"] == "sea" and 1500 < sea["km"] < 2200  # JNPT to Colombo
    assert routes["N-RAIL-LK"]["stretches"][0]["mode"] == "rail"
    assert routes["G-CAPE"]["stretches"][0]["km"] > 15000  # the long way round Africa


@pytest.fixture
def client() -> Iterator[TestClient]:
    service = RouteraService()
    with TestClient(create_routera_app(service)) as c:
        c.service = service  # type: ignore[attr-defined]
        yield c


def test_the_map_api_and_live_options_for_any_container(client: TestClient) -> None:
    assert "KASARA_GHAT" in client.get("/api/map").json()["places"]
    client.service.start_replay("nashik_corridor_strike", 0)  # type: ignore[attr-defined]
    client.service.wait_for_replay(30)  # type: ignore[attr-defined]
    state = client.get("/api/state").json()["routera"]
    box = state["fleet"][0]
    assert box["path"] in load_geo()["booked"] and box["eta"]
    assert isinstance(state["map"]["blocked"], dict)
    row = client.get(f"/api/containers/{box['container_id']}/options").json()
    assert row["container_id"] == box["container_id"] and "options" in row
    assert client.get("/api/containers/NOPE/options").status_code == 404


def test_what_if_previews_a_closure_without_moving_or_logging_anything(client: TestClient) -> None:
    client.service.start_replay("gulf_closure", 0)  # type: ignore[attr-defined]
    client.service.wait_for_replay(30)  # type: ignore[attr-defined]
    before = client.get("/api/state").json()["routera"]
    preview = client.get("/api/whatif", params={"place": "HORMUZ"}).json()
    after = client.get("/api/state").json()["routera"]
    assert preview["containers"] > 0 and preview["routes"] and "If this shut" in preview["headline"]
    assert preview["delay_days"] == 14
    assert all(r["route_id"] for r in preview["routes"])
    assert after["bookings"] == before["bookings"] and after["chain"]["length"] == before["chain"]["length"]
    assert [p["plan_id"] for p in after["plans"]] == [p["plan_id"] for p in before["plans"]]
    client.service.start_replay("nashik_corridor_strike", 0)  # type: ignore[attr-defined]
    client.service.wait_for_replay(30)  # type: ignore[attr-defined]
    road = client.get("/api/whatif", params={"place": "KASARA_GHAT"}).json()
    assert road["delay_days"] == 1 and road["containers"] > 0
    assert all(r["route_id"] not in ("N-JNPT-LK", "N-JNPT-SG", "N-VASHI") for r in road["routes"])  # the shut road
    assert client.get("/api/whatif", params={"place": "ATLANTIS"}).status_code == 404


def test_a_feed_can_be_refreshed_on_demand_but_not_hammered(client: TestClient) -> None:
    class Counting:
        name, label, interval_s = "count", "Counting (test)", 3600.0
        polls = 0

        def poll(self, now: Any) -> tuple[list[Any], dict[str, Any]]:
            Counting.polls += 1
            return [], status(self.name, self.label, True, f"poll {Counting.polls}")

    service = client.service  # type: ignore[attr-defined]
    service.feeds = [Counting()]
    assert client.post("/api/feeds/count/refresh", json={}).status_code == 409  # live mode is off
    service.start_live()
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and Counting.polls < 1:
            time.sleep(0.05)
        assert client.post("/api/feeds/nope/refresh", json={}).status_code == 404
        assert client.post("/api/feeds/count/refresh", json={}).status_code == 202
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and Counting.polls < 2:
            time.sleep(0.05)
        assert Counting.polls == 2  # polled again without waiting an hour
        assert service.live["feeds"]["count"]["interval_s"] == 3600.0
        assert client.post("/api/feeds/count/refresh", json={}).status_code == 429  # too soon
    finally:
        service.stop_replay()


def test_the_traffic_tile_needs_a_key_and_checks_its_address(client: TestClient) -> None:
    feed = TrafficFeed(env={}, fetch=lambda url: b"PNG")
    client.service.feeds = [feed]  # type: ignore[attr-defined]
    assert client.get("/api/traffic/tile/7/90/56.png").status_code == 404  # no key
    feed.key = "k"
    tile = client.get("/api/traffic/tile/7/90/56.png")
    assert tile.status_code == 200 and tile.content == b"PNG" and tile.headers["content-type"] == "image/png"
    assert client.get("/api/traffic/tile/7/999/56.png").status_code == 422  # off the map

    def down(url: str) -> bytes:
        raise OSError(f"failed {url}")

    feed.fetch = down
    broken = client.get("/api/traffic/tile/8/90/56.png")
    assert broken.status_code == 502 and "key" not in broken.text.replace("no key", "")
