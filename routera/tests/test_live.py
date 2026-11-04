"""Live mode: the real-feed adapters (with canned responses, no network) and the live service."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from agents.sentinel.fakes import JsonContainerRepo
from routera.app import AskedAhead, RouteraService
from routera.compliance import ComplianceRules
from routera.live_feeds import (
    AT_ANCHOR,
    BOOK_EPOCH,
    MANGO_BOOK,
    MOORED,
    AisFeed,
    GdacsFeed,
    NewsFeed,
    TrafficFeed,
    WeatherFeed,
    live_books,
    status,
    weather_hazard,
)
from routera.market import load_markets
from routera.shelf_life import ShelfLifeTracker

NOW = datetime(2026, 10, 29, 10, tzinfo=UTC)


def test_weather_thresholds_follow_imd_categories() -> None:
    assert weather_hazard(4.3, 80, 30) is None
    assert weather_hazard(40, 61, 30) == ("yellow", "rather heavy rain, 40 mm forecast")
    assert weather_hazard(70, 61, 30)[0] == "orange"
    assert weather_hazard(120, 61, 30)[0] == "red"
    assert weather_hazard(0, 96, 30) == ("orange", "thunderstorm with hail")
    assert weather_hazard(0, 1, 41) == ("yellow", "heatwave, 41 C")


def open_meteo(rain: list[float], code: list[int]) -> bytes:
    return json.dumps({
        "current": {"temperature_2m": 25.3, "precipitation": 0, "weather_code": 1},
        "daily": {"time": ["2026-10-29", "2026-10-30", "2026-10-31"], "precipitation_sum": rain,
                  "weather_code": code, "temperature_2m_max": [31, 30, 30]},
    }).encode()


def test_weather_feed_turns_a_heavy_rain_day_into_an_alert() -> None:
    feed = WeatherFeed(fetch=lambda url: open_meteo([0.1, 80.0, 2.9], [51, 63, 55]))
    events, state = feed.poll(NOW)
    assert state["ok"] and len(state["items"]) == 2
    assert {e["payload"]["alert_level"] for e in events} == {"orange"}
    nashik = next(e for e in events if e["payload"]["region"] == "Nashik")
    assert nashik["event_id"] == "WX-NASHIK-2026-10-30-orange"


def test_a_dead_feed_reports_itself_and_emits_nothing() -> None:
    def down(url: str) -> bytes:
        raise OSError("network down")

    events, state = WeatherFeed(fetch=down).poll(NOW)
    assert events == [] and not state["ok"] and "network down" in state["error"]


RSS = b"""<rss><channel><title>q</title>
<item><title>Transporters strike in Nashik from tonight - Lokmat</title><link>https://example.org/a</link>
<pubDate>Thu, 29 Oct 2026 08:00:00 GMT</pubDate><source url="https://lokmat.com">Lokmat</source></item>
<item><title>Grape growers fear rain - Agrowon</title><link>https://example.org/b</link>
<pubDate>Thu, 29 Oct 2026 07:00:00 GMT</pubDate><source url="https://agrowon.com">Agrowon</source></item>
</channel></rss>"""


def test_news_feed_reads_google_news_rss_and_skips_seen_articles() -> None:
    feed = NewsFeed(fetch=lambda url: RSS, queries=("nashik",))
    events, state = feed.poll(NOW)
    assert [e["payload"]["headline"] for e in events] == [
        "Transporters strike in Nashik from tonight", "Grape growers fear rain"]
    assert events[0]["payload"]["outlet"] == "Lokmat" and events[0]["source"] == "news"
    assert state["ok"]
    again, _ = feed.poll(NOW)
    assert again == []
    feed.reset()  # live restarted: the new Sentinel gets today's headlines again
    assert len(feed.poll(NOW)[0]) == 2


def gdacs_event(kind: str, level: str, lat: float, lon: float, name: str = "Storm") -> dict[str, Any]:
    return {"geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"eventtype": kind, "eventid": "77", "episodeid": "1", "alertlevel": level,
                           "iscurrent": "true", "name": name}}


def test_gdacs_only_signals_alerts_near_the_network_and_says_how_far_the_nearest_is() -> None:
    body = {"features": [gdacs_event("TC", "Orange", 19.5, 70.5, "Cyclone Near JNPT"),
                         gdacs_event("EQ", "Green", 53.6, -163.0, "Quake in Alaska")]}
    feed = GdacsFeed(fetch=lambda url: json.dumps(body).encode())
    events, state = feed.poll(NOW)
    assert state["ok"] and "2 live alerts worldwide, 1 near your routes" in state["summary"]
    assert {e["payload"]["port_code"] or e["payload"]["region"] for e in events} == {"INNSA", "INMUN", "Nashik"}
    assert {e["source"] for e in events} == {"gdacs"} and events[0]["payload"]["alert_level"] == "orange"
    assert any("Nearest: Quake in Alaska" in i["title"] for i in state["items"])
    assert {m["near"] for m in state["markers"]} == {True, False}  # both go on the map
    assert all(i["lat"] and i["lon"] for i in state["items"])  # every item can be clicked to its place
    assert feed.poll(NOW)[0] == []  # one signal per alert
    feed.reset()
    assert len(feed.poll(NOW)[0]) == 3  # all within 500 km of the cyclone


def test_an_official_orange_gdacs_alert_confirms_a_real_plan_but_news_alone_never_does() -> None:
    class Alerts:
        name, label, interval_s = "gdacs", "GDACS (test)", 3600.0

        def __init__(self, level: str) -> None:
            self.feed = GdacsFeed(fetch=lambda url: json.dumps(
                {"features": [gdacs_event("TC", level, 26.0, 58.0, "Cyclone Near Hormuz")]}).encode())

        def poll(self, now: datetime) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            return self.feed.poll(now)

    class Headlines:
        name, label, interval_s = "news", "News (test)", 3600.0

        def poll(self, now: datetime) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            titles = [("Tanker on fire and largely submerged in Hormuz", "Seatrade"),
                      ("Ship on fire in Strait of Hormuz", "Firstpost"),
                      ("Strait of Hormuz Watch: Vessel Attacks Persist", "IndustryWeek"),
                      ("Houthis threaten shipping near Hormuz", "AFP")]
            return ([{"event_id": f"N{i}", "source": "news", "kind_hint": None,
                      "payload": {"headline": t, "body": "", "outlet": o}} for i, (t, o) in enumerate(titles)],
                    status(self.name, self.label, True, "4 headlines"))

    def run_live(*feeds: Any) -> dict[str, Any]:
        service = RouteraService(feeds=list(feeds))
        try:
            service.start_live()
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline and len(service.live["feeds"]) < len(feeds):
                time.sleep(0.1)
            return service.snapshot()
        finally:
            service.stop_replay()

    news_only = run_live(Headlines())
    [candidate] = [d["disruption"] for d in news_only["disruptions"]]
    assert candidate["status"] == "candidate" and candidate["location"] == "HORMUZ" and candidate["confidence"] > 0.6
    assert news_only["routera"]["kpis"]["pending_plans"] == 0

    official = run_live(Alerts("Orange"))
    assert any(d["disruption"]["type"] == "WEATHER_ALERT" and d["disruption"]["status"] != "candidate"
               for d in official["disruptions"])
    assert official["routera"]["kpis"]["pending_plans"] >= 1

    green = run_live(Alerts("Green"))
    assert green["disruptions"] == [] and green["routera"]["kpis"]["pending_plans"] == 0


ROAD = [[20.0, 73.79], [19.64, 73.48], [18.95, 72.95]]  # Nashik, Kasara ghat, JNPT
GONE_ROUND = [[20.0, 73.79], [19.5, 73.9], [18.95, 72.95]]  # a live route that leaves the road at the ghat


def tomtom(delay_s: int = 0, incidents: list[dict[str, Any]] | None = None,
           fail_incidents: bool = False, live: list[list[float]] = GONE_ROUND) -> Any:
    """A stand-in for the TomTom API, in the response formats it really returned."""
    def fetch(url: str) -> bytes:
        if "/tile/flow/" in url:
            return b"PNG"
        if "incidentDetails" in url:
            if fail_incidents:
                raise OSError(f"failed {url}")
            return json.dumps({"incidents": incidents if "73.35" in url else []}).encode()
        path = ROAD if "traffic=false" in url else live
        leg = {"points": [{"latitude": y, "longitude": x} for y, x in path]}
        route = {"summary": {"lengthInMeters": 200000, "travelTimeInSeconds": 9000 + delay_s,
                             "trafficDelayInSeconds": delay_s, "noTrafficTravelTimeInSeconds": 9000},
                 "legs": [leg]}
        return json.dumps({"routes": [route, route]}).encode()
    return fetch


def closure(lat: float = 19.64, lon: float = 73.48, category: int = 8,
            roads: tuple[str, ...] = ("AH47", "NH160")) -> dict[str, Any]:
    return {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"iconCategory": category, "from": "Igatpuri", "to": "Kasara", "roadNumbers": list(roads),
                           "events": [{"description": "Road closed", "code": 401, "iconCategory": category}]}}


def test_traffic_is_off_without_a_key_and_never_calls_tomtom() -> None:
    def never(url: str) -> bytes:
        raise AssertionError("called TomTom without a key")

    events, state = TrafficFeed(env={}, fetch=never).poll(NOW)
    assert events == [] and not state["ok"] and "TOMTOM_API_KEY" in state["summary"]


def test_traffic_reports_live_times_alternatives_and_slows_the_planner_only_for_real_delays() -> None:
    feed = TrafficFeed(env={"TOMTOM_API_KEY": "k"}, fetch=tomtom(delay_s=5400, incidents=[]))
    events, state = feed.poll(NOW)
    assert events == [] and state["ok"]
    assert state["live_roads"]["N-JNPT-LK"] == {"minutes": 240, "delay_min": 90}
    assert {r["alternative"] for r in state["roads"]} == {True, False}  # live alternatives come back too
    assert "Nashik → JNPT: 4 h 00 now, +1 h 30 in traffic, 200 km" in [i["title"] for i in state["items"]]
    assert state["slow"]["KASARA_GHAT"] == 90 / 1440
    calm = TrafficFeed(env={"TOMTOM_API_KEY": "k"}, fetch=tomtom(delay_s=1200, incidents=[]))
    assert calm.poll(NOW)[1]["slow"] == {}  # 20 minutes doesn't change a perishable decision


def test_a_line_that_cuts_long_answers_still_gets_every_road_time_on_the_last_shape_seen() -> None:
    good = tomtom(delay_s=600, incidents=[])

    def cutting(url: str) -> bytes:  # the long answer with road shapes never arrives whole; the times-only one does
        if "traffic=true" in url and "routeRepresentation=polyline" in url:
            raise OSError("IncompleteRead")
        if "summaryOnly" in url:
            routes = json.loads(good(url))["routes"]
            return json.dumps({"routes": [{"summary": route["summary"]} for route in routes]}).encode()
        return good(url)

    feed = TrafficFeed(env={"TOMTOM_API_KEY": "k"}, fetch=good)
    shapes = [r["path"] for r in feed.poll(NOW)[1]["roads"]]
    feed.fetch = cutting
    _, state = feed.poll(NOW)
    assert state["ok"] and state["error"] == "" and len(state["live_roads"]) == len(feed.poll(NOW)[1]["live_roads"])
    assert state["live_roads"]["N-JNPT-LK"] == {"minutes": 160, "delay_min": 10}
    assert [r["path"] for r in state["roads"]] == shapes  # drawn where the road was last seen
    _, fresh = TrafficFeed(env={"TOMTOM_API_KEY": "k"}, fetch=cutting).poll(NOW)
    assert fresh["ok"] and fresh["live_roads"] and all(r["path"] == [] for r in fresh["roads"])  # times, no shapes yet


def test_a_closed_side_street_never_shuts_the_corridor() -> None:
    # what real TomTom data looks like: closed streets next to the highway, with no highway number
    side_street = closure(roads=())
    off_the_road = closure(lat=19.70, lon=73.60)  # the highway's number, but not on this stretch
    feed = TrafficFeed(env={"TOMTOM_API_KEY": "k"}, fetch=tomtom(incidents=[side_street, off_the_road]))
    events, state = feed.poll(NOW)
    assert events == [] and state["markers"] == []
    feed = TrafficFeed(env={"TOMTOM_API_KEY": "k"}, fetch=tomtom(incidents=[closure()], live=ROAD))
    events, state = feed.poll(NOW)  # on the highway, but TomTom still routes trucks through it
    assert events == [] and len(state["markers"]) == 1 and feed.closed == {}


def test_a_real_closure_counts_and_it_clears_when_tomtom_says_so() -> None:
    feed = TrafficFeed(env={"TOMTOM_API_KEY": "k"}, fetch=tomtom(incidents=[closure()]))
    [shut] = feed.poll(NOW)[0]
    assert shut["source"] == "tomtom" and shut["payload"]["segment"] == "KASARA_GHAT"
    assert shut["payload"]["status"] == "closed"
    assert feed.poll(NOW)[0] == []  # one signal per closure
    feed.fetch = tomtom(fail_incidents=True)
    events, state = feed.poll(NOW)
    assert events == [] and state["ok"]  # a failed check says nothing about the road
    feed.fetch = tomtom(incidents=[])
    [opened] = feed.poll(NOW)[0]
    assert opened["payload"]["status"] == "open" and opened["payload"]["segment"] == "KASARA_GHAT"


def test_the_tomtom_key_never_appears_in_a_status_and_a_tile_is_cached() -> None:
    feed = TrafficFeed(env={"TOMTOM_API_KEY": "secret-key"}, fetch=tomtom(fail_incidents=True))
    state = feed.poll(NOW)[1]
    assert "secret-key" not in json.dumps(state) and "failed" in state["error"]
    calls = []

    def counting(url: str) -> bytes:
        calls.append(url)
        return b"PNG"

    feed.fetch = counting
    assert feed.tile(7, 90, 56, True) == feed.tile(7, 90, 56, True) == b"PNG"
    assert len(calls) == 1 and "relative0-dark" in calls[0]


def test_a_live_tomtom_closure_confirms_a_plan_and_clearing_it_resolves_the_disruption() -> None:
    feed = TrafficFeed(env={"TOMTOM_API_KEY": "k"}, fetch=tomtom(incidents=[closure()]))
    feed.interval_s = 3600.0
    service = RouteraService(feeds=[feed])
    try:
        service.start_live()
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not service.live["feeds"]:
            time.sleep(0.1)
        state = service.snapshot()
        [road] = [d["disruption"] for d in state["disruptions"] if d["disruption"]["type"] == "ROAD_CLOSURE"]
        assert road["status"] != "candidate" and road["location"] == "KASARA_GHAT"
        assert state["routera"]["kpis"]["pending_plans"] >= 1
        assert "KASARA_GHAT" in state["routera"]["map"]["blocked"]
        feed.fetch = tomtom(incidents=[])
        [opened] = feed.poll(datetime.now(UTC))[0]
        service.ingest(opened)
        after = service.snapshot()["disruptions"]
        [road] = [d["disruption"] for d in after if d["disruption"]["type"] == "ROAD_CLOSURE"]
        assert road["status"] == "resolving"
    finally:
        service.stop_replay()


def ais_message(mmsi: int, lat: float, lon: float, status: int, sog: float) -> str:
    return json.dumps({"MessageType": "PositionReport",
                       "MetaData": {"MMSI": mmsi, "ShipName": f"SHIP {mmsi}", "latitude": lat, "longitude": lon},
                       "Message": {"PositionReport": {"NavigationalStatus": status, "Sog": sog}}})


def test_ais_feed_counts_ships_waiting_at_anchor_and_flags_congestion() -> None:
    _, off = AisFeed(env={}).poll(NOW)
    assert not off["ok"] and "AISSTREAM_API_KEY" in off["summary"]
    waiting = [ais_message(1000 + i, 18.9, 72.9, AT_ANCHOR, 0.1) for i in range(16)]
    berthed = [ais_message(2000, 18.95, 72.95, MOORED, 0.0)]
    elsewhere = [ais_message(3000, 10.0, 60.0, AT_ANCHOR, 0.0)]  # outside every port box
    feed = AisFeed(env={"AISSTREAM_API_KEY": "k"}, stream=lambda sub: [*waiting, *berthed, *elsewhere])
    first, _ = feed.poll(NOW)  # starts the listener, which may already have finished
    feed._thread.join(5)
    second, state = feed.poll(NOW)
    [event] = first + second
    assert event["source"] == "ais_port" and event["payload"]["port_code"] == "INNSA"
    assert event["payload"]["status"] == "congested" and "16 ships waiting" in event["payload"]["reason"]
    assert state["ok"] and "16 waiting at anchor, 1 at berth" in state["items"][0]["title"]
    assert len(state["markers"]) == 17 and "at anchor" in state["markers"][0]["title"]  # real ships on the map
    again, _ = feed.poll(NOW)
    assert again == []  # one signal per port per hour


def test_mangoes_age_at_their_own_setpoint_and_need_hot_water_treatment_for_the_eu() -> None:
    mango = next(c for c in JsonContainerRepo(MANGO_BOOK).list_active() if c.container_id == "MNG-03")
    tracker = ShelfLifeTracker(NOW)
    assert tracker.project(mango, 20.0, [(5.0, 0.5)]) == 15.0  # 0.5 C reefer = mango setpoint, not slower
    check = ComplianceRules().check(mango, "NL")
    assert check.status == "docs" and "hot water treatment" in check.note and "TREATMENT BOOKING" in check.draft
    treated = next(c for c in JsonContainerRepo(MANGO_BOOK).list_active() if c.container_id == "MNG-01")
    assert ComplianceRules().check(treated, "NL").status == "ok"
    assert load_markets()["NLRTM"].price("mangoes") > load_markets()["NLRTM"].price("grapes")


def test_live_books_move_the_simulated_fleet_to_today() -> None:
    [india] = [p for p in live_books(NOW) if p.name == "containers_india.json"]
    row = next(r for r in json.loads(india.read_text(encoding="utf-8")) if r["container_id"] == "NSK-01")
    eta = datetime.fromisoformat(row["eta"])
    assert eta - NOW == datetime(2026, 3, 6, 12, tzinfo=UTC) - BOOK_EPOCH


class OneShotFeed:
    """Sends the Kasara ghat closing, as the road authority would, then nothing."""

    name, label, interval_s = "road", "Road authority (test)", 3600.0

    def poll(self, now: datetime) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        event = {"event_id": "LIVE-ROAD-1", "source": "road_authority", "kind_hint": "ROAD_STATUS",
                 "payload": {"segment": "Kasara ghat", "status": "closed", "reason": "landslide",
                             "effective_from": (now - timedelta(minutes=5)).isoformat(),
                             "expected_reopen": None}}
        return [event], status(self.name, self.label, True, "1 closure")


def test_live_mode_feeds_the_same_sentinel_planner_and_chain() -> None:
    service = RouteraService(feeds=[OneShotFeed()])
    try:
        assert service.start_live()["live"] is True
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not service.live["feeds"]:
            time.sleep(0.1)
        state = service.snapshot()
        assert state["live"]["on"] and state["live"]["feeds"]["road"]["ok"]
        assert state["scenario"]["name"] == "live" and state["replay"]["speed"] == 1
        assert any(d["disruption"]["type"] == "ROAD_CLOSURE" for d in state["disruptions"])
        assert state["routera"]["kpis"]["pending_plans"] >= 1
        assert state["routera"]["chain"]["ok"]
    finally:
        service.stop_replay()


class HeadlineFeed:
    """Three new headlines, padded the way RSS titles sometimes are."""

    name, label, interval_s = "news", "Google News (test)", 3600.0

    def poll(self, now: datetime) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        events = [{"event_id": f"GN-{i}", "source": "news", "kind_hint": None,
                   "payload": {"headline": f"  Jebel Ali port closed, day {i} ", "body": "", "outlet": "Test"}}
                  for i in range(3)]
        return events, status(self.name, self.label, True, "3 headlines")


class LockWatchingLLM:
    """Says whether the service's lock was free each time it was asked."""

    def __init__(self) -> None:
        self.service: RouteraService | None = None
        self.asked: list[tuple[str, bool]] = []

    def classify_news(self, headline: str, body: str) -> dict[str, Any]:
        assert self.service is not None
        free = self.service._lock.acquire(blocking=False)
        if free:
            self.service._lock.release()
        self.asked.append((headline, free))
        return {"type": "PORT_CLOSURE", "location": "Jebel Ali", "confidence": 0.9}

    def summarise(self, disruption: dict[str, Any]) -> str:
        return ""


def test_live_headlines_go_to_the_llm_before_the_lock_and_only_once() -> None:
    llm = LockWatchingLLM()
    service = RouteraService(feeds=[HeadlineFeed()], llm=llm, llm_label="genai_hub:gpt-4o-mini")
    llm.service = service
    try:
        service.start_live()
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not service.live["feeds"]:
            time.sleep(0.1)
        state = service.snapshot()
        assert sorted(llm.asked) == [(f"Jebel Ali port closed, day {i}", True) for i in range(3)]
        assert state["llm"] == "genai_hub:gpt-4o-mini"
        assert {s["classified_by"] for s in state["signals"] if s["source"] == "news"} == {"llm"}
    finally:
        service.stop_replay()


def test_an_llm_failure_asked_ahead_falls_back_once_then_asks_again() -> None:
    asked: list[str] = []

    class Slow:
        def classify_news(self, headline: str, body: str) -> dict[str, Any]:
            asked.append(headline)
            raise TimeoutError("no reply within 8 s")

        def summarise(self, disruption: dict[str, Any]) -> str:
            return ""

    llm = AskedAhead(Slow())
    llm.ahead([{"source": "news", "payload": {"headline": "Port shut", "body": ""}},
               {"source": "road_authority", "payload": {"segment": "Kasara ghat"}}])
    with pytest.raises(TimeoutError):
        llm.classify_news("Port shut", "")  # the answer given ahead: no second wait
    assert asked == ["Port shut"]
    with pytest.raises(TimeoutError):
        llm.classify_news("Port shut", "")  # used up, so a later copy asks again
    assert asked == ["Port shut", "Port shut"]
