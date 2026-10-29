"""Real feeds for Live mode. Each poll returns raw Sentinel events plus a status for the UI.

    Open-Meteo   real weather forecast for Nashik and JNPT (no key); IMD-style thresholds
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
from pathlib import Path
from typing import Any

from agents.sentinel.fakes import ALL_CONTAINER_PATHS

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
           error: str = "") -> dict[str, Any]:
    return {"name": name, "label": label, "ok": ok, "summary": summary, "items": items or [],
            "error": error, "polled_at": datetime.now(UTC).isoformat()}


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
        events, items = [], []
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
                              f"rain next 3 days {' / '.join(f'{r:g}' for r in rains)} mm"})
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
        return events, status(self.name, self.label, True, alerts, items)


# --- news -----------------------------------------------------------------------------------------

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


class NewsFeed:
    name, label, interval_s = "news", "Google News", 300.0

    def __init__(self, fetch: Fetch = http_get, queries: tuple[str, ...] = NEWS_QUERIES) -> None:
        self.fetch, self.queries = fetch, queries
        self.seen: set[str] = set()

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
        return events, status(self.name, self.label, ok, summary, items[:8], "; ".join(errors))


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
                                  "seen": self.clock()}

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
        for code, (label, _, threshold) in AIS_PORTS.items():
            here = [v for v in fresh if v["port"] == code]
            waiting = [v for v in here if v["status"] == AT_ANCHOR]  # not tugs and barges standing still
            berthed = sum(v["status"] == MOORED for v in here)
            items.append({"title": f"{label}: {len(here)} ships, {len(waiting)} waiting at anchor, "
                                   f"{berthed} at berth"})
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
        return events, status(self.name, self.label, True, summary, items)


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
    return [WeatherFeed(), NewsFeed(), AisFeed()]

