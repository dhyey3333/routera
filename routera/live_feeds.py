"""Real feeds for Live mode. Each poll returns raw Sentinel events plus a status for the UI.

    Open-Meteo   real weather forecast for Nashik and JNPT (no key); IMD-style thresholds
    GDACS        official UN/EU disaster alerts (cyclones, floods, earthquakes, no key) near a port, sea lane
                 or corridor of the network; orange and red alerts confirm a disruption on their own
    TomTom       live road times, delays and alternative routes on the corridors, and closures on the Nashik to
                 JNPT road (free key in TOMTOM_API_KEY); a closure confirms a road disruption on its own
    Google News  real headlines on the corridor, the ports and the global lanes (no key)
    AIS          real ship positions off JNPT, Mundra, Jebel Ali and Rotterdam via aisstream.io (free key in
                 AISSTREAM_API_KEY, needs `pip install websockets`); ships waiting at anchor above a
                 threshold become a port-congestion signal

Only the feeds are real. The fleet (containers, bookings) stays the simulated book, shifted so that
its timings start today.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable, Iterator, Mapping
from datetime import UTC, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

from agents.sentinel.fakes import ALL_CONTAINER_PATHS
from routera.geo import load_geo

USER_AGENT = "ROUTERA-Hackfest/1.0 (+control room)"
IST = timezone(timedelta(hours=5, minutes=30))
MANGO_BOOK = Path(__file__).parent / "data" / "containers_mango.json"
LIVE_BOOKS = (*ALL_CONTAINER_PATHS, MANGO_BOOK)
BOOK_EPOCH = datetime(2026, 3, 2, 6, tzinfo=UTC)  # every simulated book starts here
Fetch = Callable[[str], bytes]


def http_get(url: str, timeout: float = 12.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def status(name: str, label: str, ok: bool, summary: str, items: list[dict[str, Any]] | None = None,
           error: str = "", markers: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"name": name, "label": label, "ok": ok, "summary": summary, "items": items or [],
            "error": error, "polled_at": datetime.now(UTC).isoformat(), "markers": markers or []}


# --- weather --------------------------------------------------------------------------------------

WEATHER_PLACES = (  # (label, lat, lon, region, port_code)
    ("Nashik", 19.997, 73.79, "Nashik", None),
    ("JNPT (Nhava Sheva)", 18.95, 72.95, None, "INNSA"),
)
HAIL, THUNDER = {96, 99}, {95}
LEVELS = ("yellow", "orange", "red")


def weather_hazard(rain_mm: float, code: int, tmax_c: float) -> tuple[str, str] | None:
    """(alert level, hazard) for one forecast day, using IMD rainfall categories, or None."""
    found: list[tuple[str, str]] = []
    if rain_mm >= 115.6:
        found.append(("red", f"very heavy rain, {rain_mm:.0f} mm forecast"))
    elif rain_mm >= 64.5:
        found.append(("orange", f"heavy rain, {rain_mm:.0f} mm forecast"))
    elif rain_mm >= 35.6:
        found.append(("yellow", f"rather heavy rain, {rain_mm:.0f} mm forecast"))
    if code in HAIL:
        found.append(("orange", "thunderstorm with hail"))
    elif code in THUNDER:
        found.append(("yellow", "thunderstorm"))
    if tmax_c >= 45:
        found.append(("orange", f"severe heat, {tmax_c:.0f} C"))
    elif tmax_c >= 40:
        found.append(("yellow", f"heatwave, {tmax_c:.0f} C"))
    if not found:
        return None
    level = max(found, key=lambda f: LEVELS.index(f[0]))[0]
    return level, "; ".join(h for _, h in found)


class WeatherFeed:
    name, label, interval_s = "weather", "Open-Meteo weather", 600.0

    def __init__(self, fetch: Fetch = http_get) -> None:
        self.fetch = fetch

    def poll(self, now: datetime) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        events, items, markers = [], [], []
        try:
            for label, lat, lon, region, port in WEATHER_PLACES:
                url = ("https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode({
                    "latitude": lat, "longitude": lon, "timezone": "Asia/Kolkata", "forecast_days": 3,
                    "current": "temperature_2m,precipitation,weather_code",
                    "daily": "weather_code,precipitation_sum,temperature_2m_max"}))
                data = json.loads(self.fetch(url))
                cur, daily = data["current"], data["daily"]
                rains = [round(r or 0.0, 1) for r in daily["precipitation_sum"]]
                items.append({"title": f"{label}: {cur['temperature_2m']:.0f} °C now, "
                              f"rain next 3 days {' / '.join(f'{r:g}' for r in rains)} mm",
                              "lat": lat, "lon": lon, "zoom": 9, "place": port})
                markers.append({"lat": lat, "lon": lon, "title": items[-1]["title"]})
                for day, rain, code, tmax in zip(daily["time"], rains, daily["weather_code"],
                                                 daily["temperature_2m_max"], strict=True):
                    hazard = weather_hazard(rain, int(code or 0), float(tmax or 0))
                    if hazard is None:
                        continue
                    level, text = hazard
                    start = datetime.fromisoformat(day).replace(tzinfo=IST).astimezone(UTC)
                    events.append({
                        "event_id": f"WX-{(region or port or label).upper()}-{day}-{level}",
                        "source": "weather", "kind_hint": "WEATHER_ALERT",
                        "payload": {"region": region, "port_code": port, "alert_level": level,
                                    "hazard": f"{text} (Open-Meteo forecast for {day})",
                                    "valid_from": max(start, now).isoformat(),
                                    "valid_to": (start + timedelta(days=1)).isoformat()},
                    })
        except Exception as exc:  # a feed must never take the control room down
            return [], status(self.name, self.label, False, "unreachable", items, repr(exc)[:200])
        alerts = f"{len(events)} alert(s)" if events else "no IMD threshold crossed (≥35.6 mm/day, hail, ≥40 °C)"
        return events, status(self.name, self.label, True, alerts, items, markers=markers)


# --- news -----------------------------------------------------------------------------------------

GDACS_URL = ("https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH?"
             "eventlist=TC;FL;EQ;VO;WF&alertlevel=Green;Orange;Red")
GDACS_PLACES = (  # (label, lat, lon, region, port or chokepoint code): every node a container can pass
    ("Nashik", 20.0, 73.79, "Nashik", None),
    ("JNPT", 18.95, 72.95, None, "INNSA"),
    ("Mundra", 22.75, 69.7, None, "INMUN"),
    ("Jebel Ali", 25.0, 55.06, None, "AEJEA"),
    ("Strait of Hormuz", 26.57, 56.25, None, "HORMUZ"),
    ("Salalah", 16.94, 54.0, None, "OMSLL"),
    ("Bab el-Mandeb", 12.58, 43.33, None, "BAB_EL_MANDEB"),
    ("Suez", 30.0, 32.55, None, "SUEZ"),
    ("Colombo", 6.95, 79.84, None, "LKCMB"),
    ("Port Klang", 3.0, 101.39, None, "MYPKG"),
    ("Singapore", 1.26, 103.84, None, "SGSIN"),
    ("Rotterdam", 51.95, 4.14, None, "NLRTM"),
    ("Cape of Good Hope", -34.36, 18.47, None, "CAPE"),
)
GDACS_RADIUS_KM = {"TC": 500, "FL": 300, "EQ": 200, "VO": 100, "WF": 100}  # ASSUMPTION: how far each hazard reaches
GDACS_DAYS = {"TC": 3, "FL": 3, "EQ": 1, "VO": 2, "WF": 2}  # ASSUMPTION: how long an alert stays open
GDACS_TIMEOUT_S = 40.0
GDACS_LEVEL = {"Red": "red", "Orange": "orange", "Green": "yellow"}  # the Sentinel acts on orange and red only


def km_between(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    a = (math.sin(math.radians(lat2 - lat1) / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6371 * math.asin(math.sqrt(a))


class GdacsFeed:
    """GDACS (UN and European Commission): official alerts for live disasters worldwide. Only those near
    the network become Sentinel signals; the card also says how close the nearest one is."""

    name, label, interval_s = "gdacs", "GDACS disaster alerts (UN/EU)", 900.0

    def __init__(self, fetch: Fetch | None = None) -> None:
        self.fetch = fetch or (lambda url: http_get(url, timeout=GDACS_TIMEOUT_S))
        self.sent: set[str] = set()

    def reset(self) -> None:
        """A new live run has a new Sentinel: send it every current alert again."""
        self.sent.clear()

    def poll(self, now: datetime) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        try:
            features = json.loads(self.fetch(GDACS_URL)).get("features", [])
        except Exception:  # GDACS is slow at times: try once more before reporting it down
            try:
                features = json.loads(self.fetch(GDACS_URL)).get("features", [])
            except Exception as exc:  # a feed must never take the control room down
                return [], status(self.name, self.label, False, "unreachable", [], repr(exc)[:200])
        events, near, far, markers = [], [], [], []
        for feature in features:
            p = feature.get("properties", {})
            kind, level = p.get("eventtype"), p.get("alertlevel")
            if kind not in GDACS_RADIUS_KM or level not in GDACS_LEVEL or str(p.get("iscurrent")).lower() != "true":
                continue
            lon, lat = feature["geometry"]["coordinates"][:2]
            spots = sorted((km_between(lat, lon, plat, plon), place) for place in GDACS_PLACES
                           for plat, plon in [place[1:3]])
            km, closest = spots[0]
            name = str(p.get("name") or kind)[:80]
            title = f"{name} ({level}): {km:,.0f} km from {closest[0]}"
            markers.append({"lat": lat, "lon": lon, "title": title, "level": level, "kind": kind,
                            "near": km <= GDACS_RADIUS_KM[kind]})
            spot = {"lat": lat, "lon": lon, "zoom": 6, "place": closest[4], "radius_km": GDACS_RADIUS_KM[kind]}
            if km > GDACS_RADIUS_KM[kind]:
                far.append((km, title, spot))
                continue
            near.append((title, spot))
            for dist, (label, _, _, region, port) in spots:
                event_id = f"GDACS-{kind}{p.get('eventid')}-{p.get('episodeid')}-{port or label.upper()}-{level}"
                if dist > GDACS_RADIUS_KM[kind] or event_id in self.sent:
                    continue
                self.sent.add(event_id)
                events.append({
                    "event_id": event_id, "source": "gdacs", "kind_hint": "WEATHER_ALERT",
                    "payload": {"region": region, "port_code": port, "alert_level": GDACS_LEVEL[level],
                                "hazard": f"{name}, {dist:,.0f} km from {label} (GDACS {level} alert)",
                                "valid_from": now.isoformat(),
                                "valid_to": (now + timedelta(days=GDACS_DAYS[kind])).isoformat()},
                })
        nearest = sorted(far, key=lambda found: found[0])[:3]
        items = [{"title": f"⚠ {title}", **spot} for title, spot in near]
        items += [{"title": f"Nearest: {title}", **spot} for _, title, spot in nearest]
        summary = f"{len(near) + len(far)} live alerts worldwide, {len(near)} near your routes"
        return events, status(self.name, self.label, True, summary, items[:8], markers=markers)


NEWS_QUERIES = (
    '("truck strike" OR "transporters strike" OR "chakka jam") (Nashik OR Maharashtra OR Mumbai)',
    '"Kasara ghat" OR "Mumbai-Nashik highway" OR "Nashik highway"',
    '(JNPT OR JNPA OR "Nhava Sheva") (congestion OR delay OR closed OR strike)',
    '(grape OR grapes) (Nashik OR export) (price OR rain OR export)',
    '("Red Sea" OR "Bab el-Mandeb" OR Hormuz OR "Jebel Ali") shipping',
    '(mango OR mangoes OR Alphonso OR Kesar) (export OR Vashi OR price OR cyclone OR irradiation)',
)
PER_QUERY = 5


def parse_rss(xml_bytes: bytes) -> list[dict[str, str]]:
    root = ET.fromstring(xml_bytes)
    items = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        source = (item.findtext("source") or "").strip()
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3]
        items.append({"title": title, "link": (item.findtext("link") or "").strip(),
                      "published": (item.findtext("pubDate") or "").strip(), "outlet": source or "Google News"})
    return items


MAX_NEWS_ITEMS = 30  # the card scrolls, and shows the headlines the Sentinel acted on first


class NewsFeed:
    name, label, interval_s = "news", "Google News", 300.0

    def __init__(self, fetch: Fetch = http_get, queries: tuple[str, ...] = NEWS_QUERIES) -> None:
        self.fetch, self.queries = fetch, queries
        self.seen: set[str] = set()

    def reset(self) -> None:
        """A new live run has a new Sentinel: send it every current headline again."""
        self.seen.clear()

    def poll(self, now: datetime) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        events, items, errors = [], [], []
        for query in self.queries:
            url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
                {"q": f"{query} when:2d", "hl": "en-IN", "gl": "IN", "ceid": "IN:en"})
            try:
                found = parse_rss(self.fetch(url))[:PER_QUERY]
            except Exception as exc:
                errors.append(repr(exc)[:120])
                continue
            for article in found:
                key = hashlib.sha1((article["link"] or article["title"]).encode()).hexdigest()[:12]
                items.append(article)
                if key in self.seen or not article["title"]:
                    continue
                self.seen.add(key)
                published = article["published"]
                with contextlib.suppress(TypeError, ValueError):
                    published = parsedate_to_datetime(published).astimezone(IST).strftime("%d %b %H:%M IST")
                events.append({"event_id": f"GN-{key}", "source": "news", "kind_hint": None,
                               "payload": {"headline": article["title"][:300],
                                           "body": f"Published {published}. {article['link']}"[:600],
                                           "outlet": article["outlet"][:80]}})
        ok = len(errors) < len(self.queries)
        summary = f"{len(items)} headlines from the last 2 days, {len(events)} new" if ok else "unreachable"
        return events, status(self.name, self.label, ok, summary, items[:MAX_NEWS_ITEMS], "; ".join(errors))


# --- ships (AIS) ----------------------------------------------------------------------------------

AIS_URL = "wss://stream.aisstream.io/v0/stream"
AIS_PORTS = {  # port -> (label, [[lat_min, lon_min], [lat_max, lon_max]], ships waiting = congested)
    "INNSA": ("JNPT / Mumbai anchorage", [[18.80, 72.70], [19.05, 73.02]], 15),
    "INMUN": ("Mundra", [[22.60, 69.55], [22.85, 69.85]], 15),
    "AEJEA": ("Jebel Ali", [[24.85, 54.85], [25.15, 55.15]], 20),
    "NLRTM": ("Rotterdam anchorages", [[51.85, 3.30], [52.15, 4.20]], 30),
}
# aisstream.io is crowd-sourced: Europe and North America are well covered, Indian and Gulf waters
# barely (checked 29 Oct 2026: 0 of 727 positions in 20 s). Those cards show 0 rather than a guess.
AT_ANCHOR, MOORED = 1, 5  # AIS navigational status codes
AIS_STATE = {AT_ANCHOR: "at anchor", MOORED: "at berth"}
AIS_MAX_MARKERS = 300
FRESH_S = 1800  # anchored ships report every few minutes; forget a ship after 30 min of silence


def port_for(lat: float | None, lon: float | None) -> str | None:
    if lat is None or lon is None:
        return None
    for code, (_, ((lat0, lon0), (lat1, lon1)), _) in AIS_PORTS.items():
        if lat0 <= lat <= lat1 and lon0 <= lon <= lon1:
            return code
    return None


class AisFeed:
    """Keeps one websocket to aisstream.io open in the background; each poll counts the ships."""

    name, label, interval_s = "ais", "AIS ships (aisstream.io)", 300.0

    def __init__(self, env: Mapping[str, str] | None = None,
                 stream: Callable[[str], Iterable[str]] | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.key = (os.environ if env is None else env).get("AISSTREAM_API_KEY", "")
        self.stream, self.clock = stream, clock
        self.vessels: dict[Any, dict[str, Any]] = {}
        self.sent: set[str] = set()
        self.error = ""
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def reset(self) -> None:
        """A new live run: signal congestion again (the ships seen so far are still counted)."""
        self.sent.clear()

    def record(self, message: Mapping[str, Any]) -> None:
        meta = message.get("MetaData") or {}
        report = (message.get("Message") or {}).get("PositionReport") or {}
        lat = meta.get("latitude", report.get("Latitude"))
        lon = meta.get("longitude", report.get("Longitude"))
        port, mmsi = port_for(lat, lon), meta.get("MMSI") or report.get("UserID")
        if port is None or mmsi is None:
            return
        with self._lock:
            self.vessels[mmsi] = {"port": port, "name": str(meta.get("ShipName") or mmsi).strip(),
                                  "status": report.get("NavigationalStatus"), "sog": report.get("Sog"),
                                  "lat": lat, "lon": lon, "seen": self.clock()}

    def _messages(self, subscription: str) -> Iterator[str]:
        if self.stream is not None:
            yield from self.stream(subscription)
            return
        from websockets.sync.client import connect

        with connect(AIS_URL, open_timeout=20) as ws:
            ws.send(subscription)
            yield from ws

    def _listen(self) -> None:
        subscription = json.dumps({"APIKey": self.key, "FilterMessageTypes": ["PositionReport"],
                                   "BoundingBoxes": [box for _, box, _ in AIS_PORTS.values()]})
        while True:
            try:
                for raw in self._messages(subscription):
                    self.record(json.loads(raw))
                    self.error = ""
            except Exception as exc:
                self.error = repr(exc)[:200]
            if self.stream is not None:
                return
            time.sleep(15)  # reconnect

    def poll(self, now: datetime) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not self.key:
            return [], status(self.name, self.label, False, "off: set AISSTREAM_API_KEY (free at aisstream.io)")
        if self.stream is None:
            try:
                import websockets  # noqa: F401
            except ImportError:
                return [], status(self.name, self.label, False, "off: pip install websockets")
        if self._thread is None:
            self._thread = threading.Thread(target=self._listen, name="ais", daemon=True)
            self._thread.start()
        with self._lock:
            fresh = [v for v in self.vessels.values() if self.clock() - v["seen"] < FRESH_S]
        events, items = [], []
        for code, (label, box, threshold) in AIS_PORTS.items():
            here = [v for v in fresh if v["port"] == code]
            waiting = [v for v in here if v["status"] == AT_ANCHOR]  # not tugs and barges standing still
            berthed = sum(v["status"] == MOORED for v in here)
            items.append({"title": f"{label}: {len(here)} ships, {len(waiting)} waiting at anchor, "
                                   f"{berthed} at berth", "place": code, "zoom": 9,
                          "lat": (box[0][0] + box[1][0]) / 2, "lon": (box[0][1] + box[1][1]) / 2})
            event_id = f"AIS-{code}-{now:%Y%m%d%H}"
            if len(waiting) >= threshold and event_id not in self.sent:
                self.sent.add(event_id)
                events.append({"event_id": event_id, "source": "ais_port", "kind_hint": "PORT_STATUS",
                               "payload": {"port_code": code, "status": "congested",
                                           "effective_from": now.isoformat(),
                                           "reason": f"live AIS: {len(waiting)} ships waiting at anchor off "
                                                     f"{label} (ROUTERA threshold {threshold})"}})
        if not fresh and self.error:
            return [], status(self.name, self.label, False, "unreachable", items, self.error)
        summary = (f"{len(fresh)} ships tracked in the last 30 min" if fresh
                   else "listening… (anchored ships report every few minutes)")
        markers = [{"lat": v["lat"], "lon": v["lon"],
                    "title": f"{v['name']}: {AIS_STATE.get(v['status'], 'under way')}, {v['sog'] or 0:g} kn"}
                   for v in fresh[:AIS_MAX_MARKERS]]
        return events, status(self.name, self.label, True, summary, items, markers=markers)


# --- TomTom live road traffic ---------------------------------------------------------------------

TOMTOM = "https://api.tomtom.com"
TRAFFIC_CORRIDORS = (  # (label, from, to, planner routes on this road, place the planner slows down)
    ("Nashik → JNPT", "NASHIK", "INNSA", ("N-JNPT-LK", "N-JNPT-SG", "N-HOLD-LK"), "KASARA_GHAT"),
    ("Nashik → Mundra", "NASHIK", "INMUN", ("N-MUN-LK",), "NSK-MUN"),
    ("Nashik → Pune", "NASHIK", "MKT-PUNE", ("N-PUNE",), None),
    ("Nashik → Bengaluru", "NASHIK", "MKT-BLR", ("N-BLR",), None),
    ("Nashik → Hyderabad", "NASHIK", "MKT-HYD", ("N-HYD",), None),
    ("Nashik → Delhi", "NASHIK", "MKT-AZADPUR", ("N-DEL",), None),
    ("JNPT → Delhi", "INNSA", "MKT-AZADPUR", ("J-DEL", "J-HOLD", "G-DEL", "G-HOLD"), None),
    ("JNPT → Bengaluru", "INNSA", "MKT-BLR", ("J-BLR",), None),
)
# The incident API allows 10,000 km2 per box, so the Nashik to JNPT road takes two boxes (min lon, min lat, ...).
INCIDENT_BOXES = ("73.35,19.55,74.15,20.25", "72.85,18.85,73.55,19.75")
INCIDENT_FIELDS = ("{incidents{type,geometry{type,coordinates},properties{iconCategory,magnitudeOfDelay,"
                   "events{description,code,iconCategory},startTime,endTime,from,to,length,delay,roadNumbers}}}")
JAM, ROAD_CLOSED = 6, 8
ROAD_ALERTS = {1, 3, JAM, 7, ROAD_CLOSED, 9, 11, 14}  # accident, danger, jam, lane or road closed, works, flood...
# TomTom lists hundreds of closed side streets around this road, so a closure must pass three tests: it carries the
# highway's number, it lies on the road itself, and TomTom's own live route has actually gone around it.
CORRIDOR_ROADS = {"NH160", "AH47", "NH348"}
ON_ROAD_KM = 0.1
KASARA_KM = 12.0
TRAFFIC_POINTS = {"INMUN": {"lat": 22.79, "lon": 69.705}}  # where trucks reach it: the port's own point has no road
SLOW_FROM_MIN = 60  # below this a delay doesn't change a perishable decision
TILE_TTL_S, TILE_CACHE_MAX = 300.0, 1000


def km_to_path(lat: float, lon: float, path: list[list[float]]) -> float:
    """Shortest distance from a point to a polyline of [lat, lon] points, on a local flat projection."""
    scale = math.cos(math.radians(lat))
    best = math.inf
    for (lat1, lon1), (lat2, lon2) in pairwise(path):
        ax, ay, bx, by = (lon1 - lon) * scale, lat1 - lat, (lon2 - lon) * scale, lat2 - lat
        dx, dy = bx - ax, by - ay
        t = max(0.0, min(1.0, -(ax * dx + ay * dy) / ((dx * dx + dy * dy) or 1e-12)))
        best = min(best, math.hypot(ax + t * dx, ay + t * dy) * 111.2)
    return best


def clock_time(minutes: float) -> str:
    hours, mins = divmod(round(minutes), 60)
    return f"{hours} h {mins:02d}" if hours else f"{mins} min"


class TrafficFeed:
    """TomTom Traffic. Each poll takes 10 requests, so every 10 minutes is 1,440 of the free 2,500 a day
    (plus one per run for the road's own path).

    The key stays on the server: it never reaches the browser, and map tiles come through tile() below.
    Only the Nashik to JNPT road is checked for closures; TomTom's data for an Indian road may be thin."""

    name, label, interval_s = "traffic", "TomTom live traffic", 600.0

    def __init__(self, env: Mapping[str, str] | None = None, fetch: Fetch = http_get) -> None:
        self.key = (os.environ if env is None else env).get("TOMTOM_API_KEY", "")
        self.fetch = fetch
        self.closed: dict[str, str] = {}  # segment -> what closed it
        self.corridor: list[list[float]] | None = None  # the Nashik to JNPT road when nothing is closed
        self._tiles: dict[tuple[int, int, int, bool], tuple[float, bytes]] = {}
        self._lock = threading.Lock()

    def reset(self) -> None:
        """A new live run has a new Sentinel: tell it about current closures again."""
        self.closed.clear()
        self.corridor = None

    def _clean(self, exc: Exception) -> str:
        return repr(exc).replace(self.key, "***")[:200] if self.key else repr(exc)[:200]

    def _route_url(self, a: dict[str, Any], b: dict[str, Any]) -> str:
        return (f"{TOMTOM}/routing/1/calculateRoute/{a['lat']},{a['lon']}:{b['lat']},{b['lon']}/json?"
                + urllib.parse.urlencode({"key": self.key, "traffic": "true", "travelMode": "truck",
                                          "vehicleCommercial": "true", "computeTravelTimeFor": "all",
                                          "maxAlternatives": 2, "routeRepresentation": "polyline"}))

    def _corridor(self, geo: dict[str, Any]) -> list[list[float]]:
        """The road itself, in full detail: via Kasara, ignoring live traffic. Fetched once per run."""
        if self.corridor is None:
            a, ghat, b = (geo["places"][code] for code in ("NASHIK", "KASARA_GHAT", "INNSA"))
            url = (f"{TOMTOM}/routing/1/calculateRoute/{a['lat']},{a['lon']}:{ghat['lat']},{ghat['lon']}:"
                   f"{b['lat']},{b['lon']}/json?"
                   + urllib.parse.urlencode({"key": self.key, "traffic": "false", "travelMode": "truck",
                                             "vehicleCommercial": "true", "routeRepresentation": "polyline"}))
            legs = json.loads(self.fetch(url))["routes"][0]["legs"]
            self.corridor = [[p["latitude"], p["longitude"]] for leg in legs for p in leg["points"]]
        return self.corridor

    def _incidents(self) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        for box in INCIDENT_BOXES:
            query = urllib.parse.urlencode({"key": self.key, "bbox": box, "fields": INCIDENT_FIELDS,
                                            "language": "en-GB", "timeValidityFilter": "present"})
            url = f"{TOMTOM}/traffic/services/5/incidentDetails?{query}"
            found += json.loads(self.fetch(url)).get("incidents", [])
        return found

    def poll(self, now: datetime) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not self.key:
            return [], status(self.name, self.label, False, "off: set TOMTOM_API_KEY (free at developer.tomtom.com)")
        geo = load_geo()
        roads, items, errors = [], [], []
        live_path: list[list[float]] | None = None  # where TomTom sends a truck from Nashik to JNPT right now
        live_roads: dict[str, dict[str, float]] = {}
        slow: dict[str, float] = {}
        for label, a, b, route_ids, place in TRAFFIC_CORRIDORS:
            try:
                start, end = (TRAFFIC_POINTS.get(code) or geo["places"][code] for code in (a, b))
                routes = json.loads(self.fetch(self._route_url(start, end))).get("routes") or []
            except Exception as exc:
                errors.append(self._clean(exc))
                continue
            for i, route in enumerate(routes):
                s = route["summary"]
                minutes, delay = s["travelTimeInSeconds"] / 60, s.get("trafficDelayInSeconds", 0) / 60
                clear = s.get("noTrafficTravelTimeInSeconds", s["travelTimeInSeconds"]) / 60
                points = [[p["latitude"], p["longitude"]]
                          for leg in route.get("legs", []) for p in leg.get("points", [])]
                roads.append({"label": label, "alternative": i > 0, "km": round(s["lengthInMeters"] / 1000),
                              "minutes": round(minutes), "delay_min": round(delay), "clear_min": round(clear),
                              "path": points[:: max(1, len(points) // 150)] + points[-1:]})
                if i:
                    continue
                if place == "KASARA_GHAT":
                    live_path = points
                late = f"+{clock_time(delay)} in traffic" if delay >= 1 else "no delay"
                km = s["lengthInMeters"] / 1000
                items.append({"title": f"{label}: {clock_time(minutes)} now, {late}, {km:,.0f} km",
                              "road": label, "place": place, "via": place or b})
                for route_id in route_ids:
                    live_roads[route_id] = {"minutes": round(minutes), "delay_min": round(delay)}
                if place and delay >= SLOW_FROM_MIN:
                    slow[place] = round(delay / 15) * 15 / 1440  # in days, in steps of 15 min so plans stay steady
        if len(errors) == len(TRAFFIC_CORRIDORS):
            return [], status(self.name, self.label, False, "unreachable, over the free limit, or a bad key", [],
                              errors[0])
        events: list[dict[str, Any]] = []
        markers: list[dict[str, Any]] = []
        try:
            events, markers = self._closures(self._incidents(), now, geo, self._corridor(geo), live_path)
        except Exception as exc:  # unknown is not the same as clear: no road is reopened on a failed check
            errors.append(self._clean(exc))
        worst = max((r["delay_min"] for r in roads if not r["alternative"]), default=0)
        closures = sum(m["category"] == ROAD_CLOSED for m in markers)
        summary = f"{len(items)} roads live, " + (f"worst +{clock_time(worst)}" if worst >= 1 else "no delays")
        summary += f" · {len(markers)} incident(s) on NH160" + (f", {closures} closed" if closures else "")
        worst_first = sorted(markers, key=lambda m: m["category"] != ROAD_CLOSED)
        alerts = [{"title": f"⚠ {m['title']}", "lat": m["lat"], "lon": m["lon"], "zoom": 13, "place": m["place"]}
                  for m in worst_first[:4]]
        state = status(self.name, self.label, True, summary, alerts + items, "; ".join(errors), markers)
        state.update(roads=roads, live_roads=live_roads, slow=slow)
        return events, state

    def _closures(self, incidents: list[dict[str, Any]], now: datetime, geo: dict[str, Any],
                  corridor: list[list[float]], live: list[list[float]] | None,
                  ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        ghat = geo["places"]["KASARA_GHAT"]
        events, markers = [], []
        closed_now: dict[str, str] = {}
        for incident in incidents:
            p = incident.get("properties", {})
            category = p.get("iconCategory")
            coords = (incident.get("geometry") or {}).get("coordinates") or []
            points = coords if coords and isinstance(coords[0], list) else [coords]
            points = [c for c in points if len(c) >= 2]
            if category not in ROAD_ALERTS or not points:
                continue
            on_highway = not CORRIDOR_ROADS.isdisjoint(p.get("roadNumbers") or [])
            if not on_highway and category != JAM:  # a jam may have no road number; anything else is a side road
                continue
            if max(km_to_path(c[1], c[0], corridor) for c in points) > ON_ROAD_KM:
                continue
            what = ((p.get("events") or [{}])[0].get("description") or "Traffic incident")[:80]
            where = " to ".join(x for x in (p.get("from"), p.get("to")) if x)[:80]
            late = f", +{clock_time(p['delay'] / 60)}" if p.get("delay") else ""
            near_ghat = min(km_between(c[1], c[0], ghat["lat"], ghat["lon"]) for c in points) <= KASARA_KM
            segment = "KASARA_GHAT" if near_ghat else "NSK-JNPT"
            markers.append({"lat": points[0][1], "lon": points[0][0], "category": category, "place": segment,
                            "title": f"{what}: {where}{late}" if where else f"{what}{late}"})
            middle = points[len(points) // 2]
            still_routed = not live or km_to_path(middle[1], middle[0], live) <= ON_ROAD_KM
            if category != ROAD_CLOSED or not on_highway or still_routed:
                continue  # if TomTom's live route still runs through it, trucks are getting through
            closed_now[segment] = markers[-1]["title"]
            if segment in self.closed:
                continue
            reopen = None
            with contextlib.suppress(TypeError, ValueError):
                end = datetime.fromisoformat(str(p.get("endTime")).replace("Z", "+00:00"))
                reopen = end.isoformat() if end > now else None
            title = markers[-1]["title"]
            events.append({"event_id": f"TT-{segment}-{hashlib.sha1(title.encode()).hexdigest()[:10]}",
                           "source": "tomtom", "kind_hint": "ROAD_STATUS",
                           "payload": {"segment": segment, "status": "closed", "reason": f"TomTom live: {title}",
                                       "effective_from": now.isoformat(), "expected_reopen": reopen}})
        for segment in set(self.closed) - set(closed_now):
            events.append({"event_id": f"TT-{segment}-open-{now:%Y%m%d%H%M}", "source": "tomtom",
                           "kind_hint": "ROAD_STATUS",
                           "payload": {"segment": segment, "status": "open", "reason": "TomTom live: closure cleared",
                                       "effective_from": now.isoformat(), "expected_reopen": None}})
        self.closed = closed_now
        return events, markers

    def tile(self, z: int, x: int, y: int, dark: bool) -> bytes:
        """A traffic-flow map tile, fetched here so the key stays on the server, and kept for 5 minutes."""
        if not self.key:
            raise LookupError("no TomTom key")
        if not (3 <= z <= 22 and 0 <= x < 2 ** z and 0 <= y < 2 ** z):
            raise ValueError("no such tile")
        slot = (z, x, y, dark)
        with self._lock:
            held = self._tiles.get(slot)
            if held and time.monotonic() - held[0] < TILE_TTL_S:
                return held[1]
        style = "relative0-dark" if dark else "relative0"
        data = self.fetch(f"{TOMTOM}/traffic/map/4/tile/flow/{style}/{z}/{x}/{y}.png?key={self.key}")
        with self._lock:
            if len(self._tiles) >= TILE_CACHE_MAX:
                self._tiles.clear()
            self._tiles[slot] = (time.monotonic(), data)
        return data


# --- the fleet, moved to today --------------------------------------------------------------------


def live_books(now: datetime, books: tuple[Path, ...] = LIVE_BOOKS) -> list[Path]:
    """Copies of the simulated container books with every timestamp shifted so they start `now`."""
    shift = now - BOOK_EPOCH
    out = Path(tempfile.mkdtemp(prefix="routera-live-"))
    paths = []
    for book in books:
        rows = json.loads(book.read_text(encoding="utf-8"))
        for row in rows:
            for field in ("eta", "cutoff_at"):
                if row.get(field):
                    when = datetime.fromisoformat(row[field].replace("Z", "+00:00")) + shift
                    row[field] = when.isoformat()
        path = out / book.name
        path.write_text(json.dumps(rows), encoding="utf-8")
        paths.append(path)
    return paths


def default_feeds() -> list[Any]:
    return [WeatherFeed(), GdacsFeed(), TrafficFeed(), NewsFeed(), AisFeed()]

