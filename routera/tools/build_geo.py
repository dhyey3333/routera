"""Dev tool: builds routera/data/geo.json, the map's places and real route geometry.

Roads come from OSRM (OpenStreetMap data), sea lanes from the searoute package (open marine network).
Rail has no free router, so its path is hand-drawn through the real junctions. Run it only when routes
change; the app reads the JSON and needs neither service at runtime.

    pip install searoute
    python -m routera.tools.build_geo
"""

from __future__ import annotations

import json
import math
import time
import urllib.request
from itertools import pairwise
from pathlib import Path
from typing import Any

import searoute

from agents.sentinel.fakes import JsonContainerRepo
from routera.geo import GEO_PATH, booked_key
from routera.live_feeds import LIVE_BOOKS
from routera.reroute import load_routes

Point = list[float]
PLACES: dict[str, tuple[str, float, float, str]] = {
    "INNSA": ("JNPT (Nhava Sheva)", 18.95, 72.95, "port"),
    "INMUN": ("Mundra", 22.75, 69.70, "port"),
    "INVTZ": ("Visakhapatnam", 17.69, 83.29, "port"),
    "INCOK": ("Kochi", 9.97, 76.24, "port"),
    "AEJEA": ("Jebel Ali", 25.01, 55.06, "port"),
    "QAHMD": ("Hamad", 25.01, 51.60, "port"),
    "OMSLL": ("Salalah", 16.94, 54.00, "port"),
    "LKCMB": ("Colombo", 6.95, 79.84, "port"),
    "SGSIN": ("Singapore", 1.26, 103.84, "port"),
    "MYPKG": ("Port Klang", 3.00, 101.39, "port"),
    "NLRTM": ("Rotterdam", 51.95, 4.14, "port"),
    "GBLGP": ("London Gateway", 51.50, 0.48, "port"),
    "DEHAM": ("Hamburg", 53.55, 9.97, "port"),
    "USLAX": ("Los Angeles", 33.74, -118.27, "port"),
    "USNYC": ("New York", 40.67, -74.04, "port"),
    "HORMUZ": ("Strait of Hormuz", 26.57, 56.25, "chokepoint"),
    "BAB_EL_MANDEB": ("Bab el-Mandeb", 12.58, 43.33, "chokepoint"),
    "SUEZ": ("Suez Canal", 30.00, 32.55, "chokepoint"),
    "CAPE": ("Cape of Good Hope", -34.36, 18.47, "chokepoint"),
    "KASARA_GHAT": ("Kasara ghat", 19.64, 73.48, "road"),
    "PH-PMP": ("Pimpalgaon pack house", 20.17, 73.99, "facility"),
    "PH-NPD": ("Niphad pack house", 20.08, 74.11, "facility"),
    "CS-NSK": ("Nashik cold store", 19.99, 73.79, "facility"),
    "CS-JNPT": ("JNPT cold store", 18.93, 72.97, "facility"),
    "PROC-NSK": ("Nashik processors", 20.03, 73.85, "market"),
    "MKT-VASHI": ("Vashi APMC, Mumbai", 19.08, 73.00, "market"),
    "MKT-PUNE": ("Gultekdi, Pune", 18.50, 73.87, "market"),
    "MKT-BLR": ("Bengaluru", 12.97, 77.59, "market"),
    "MKT-HYD": ("Hyderabad", 17.33, 78.50, "market"),
    "MKT-AZADPUR": ("Azadpur, Delhi", 28.71, 77.18, "market"),
    "NASHIK": ("Nashik grape belt", 20.00, 73.79, "region"),
    "RED_SEA": ("Red Sea", 20.00, 38.50, "region"),
    "GULF": ("Persian Gulf", 26.50, 52.00, "region"),
}
HIDDEN = {"GULF_SEA": (23.00, 62.50)}  # where Gulf-bound boxes are taken to be when a diversion starts
RAIL_NASHIK_JNPT = [[19.95, 73.84], [19.70, 73.56], [19.65, 73.47], [19.24, 73.13], [18.99, 73.12], [18.95, 72.95]]
ROAD_NODES = {"NASHIK", "KASARA_GHAT", "INNSA", "INMUN", "PH-PMP", "PH-NPD", "CS-NSK", "CS-JNPT", "PROC-NSK",
              "MKT-VASHI", "MKT-PUNE", "MKT-BLR", "MKT-HYD", "MKT-AZADPUR"}
GHAT = ("NASHIK", "KASARA_GHAT", "INNSA")
ROUTE_PLANS: dict[str, list[tuple[str, ...]]] = {  # (mode, node, node, ...) per stretch
    "G-SG": [("sea", "GULF_SEA", "OMSLL", "SGSIN")],
    "G-LK": [("sea", "GULF_SEA", "LKCMB")],
    "G-MY": [("sea", "GULF_SEA", "OMSLL", "MYPKG")],
    "G-CAPE": [("sea", "GULF_SEA", "CAPE", "NLRTM")],
    "G-HOLD": [("sea", "GULF_SEA", "INNSA"), ("road", "INNSA", "MKT-AZADPUR")],
    "G-DEL": [("sea", "GULF_SEA", "INNSA"), ("road", "INNSA", "MKT-AZADPUR")],
    "G-PROC": [("sea", "GULF_SEA", "INNSA"), ("road", "INNSA", "PROC-NSK")],
    "J-SG": [("sea", "INNSA", "SGSIN")],
    "J-LK": [("sea", "INNSA", "LKCMB")],
    "J-MY": [("sea", "INNSA", "MYPKG")],
    "J-HOLD": [("road", "INNSA", "MKT-AZADPUR")],
    "J-DEL": [("road", "INNSA", "MKT-AZADPUR")],
    "J-BLR": [("road", "INNSA", "MKT-BLR")],
    "J-PROC": [("road", "INNSA", "PROC-NSK")],
    "N-JNPT-LK": [("road", *GHAT), ("sea", "INNSA", "LKCMB")],
    "N-RAIL-LK": [("rail",), ("sea", "INNSA", "LKCMB")],
    "N-MUN-LK": [("road", "NASHIK", "INMUN"), ("sea", "INMUN", "LKCMB")],
    "N-HOLD-LK": [("road", *GHAT), ("sea", "INNSA", "LKCMB")],
    "N-JNPT-SG": [("road", *GHAT), ("sea", "INNSA", "SGSIN")],
    "N-RAIL-SG": [("rail",), ("sea", "INNSA", "SGSIN")],
    "N-VASHI": [("road", "NASHIK", "KASARA_GHAT", "MKT-VASHI")],
    "N-PUNE": [("road", "NASHIK", "MKT-PUNE")],
    "N-BLR": [("road", "NASHIK", "MKT-BLR")],
    "N-HYD": [("road", "NASHIK", "MKT-HYD")],
    "N-DEL": [("road", "NASHIK", "MKT-AZADPUR")],
    "N-PROC": [("road", "NASHIK", "PROC-NSK")],
}
_cache: dict[tuple[str, ...], tuple[list[Point], float]] = {}


def coords(code: str) -> tuple[float, float]:
    return HIDDEN[code] if code in HIDDEN else PLACES[code][1:3]


def km_between(a: Point, b: Point) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


def simplify(points: list[Point], tolerance: float) -> list[Point]:
    """Douglas-Peucker, in degrees, so a path keeps its shape without thousands of points."""
    keep = {0, len(points) - 1}
    stack = [(0, len(points) - 1)]
    while stack:
        first, last = stack.pop()
        (y1, x1), (y2, x2) = points[first], points[last]
        worst, index = 0.0, -1
        for i in range(first + 1, last):
            y, x = points[i]
            dx, dy = x2 - x1, y2 - y1
            length = (dx * dx + dy * dy) or 1e-12
            t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / length))
            distance = math.hypot(x - (x1 + t * dx), y - (y1 + t * dy))
            if distance > worst:
                worst, index = distance, i
        if worst > tolerance:
            keep.add(index)
            stack += [(first, index), (index, last)]
    return [points[i] for i in sorted(keep)]


def unwrap(points: list[Point]) -> list[Point]:
    """Keep longitudes continuous, so a Pacific crossing doesn't draw a line across the whole map."""
    out = [points[0]]
    for lat, lon in points[1:]:
        while lon - out[-1][1] > 180:
            lon -= 360
        while lon - out[-1][1] < -180:
            lon += 360
        out.append([lat, lon])
    return out


def road(nodes: tuple[str, ...]) -> tuple[list[Point], float]:
    if ("road", *nodes) not in _cache:
        spots = ";".join(f"{coords(n)[1]},{coords(n)[0]}" for n in nodes)
        url = f"https://router.project-osrm.org/route/v1/driving/{spots}?overview=full&geometries=geojson"
        request = urllib.request.Request(url, headers={"User-Agent": "ROUTERA map builder"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    best = json.load(response)["routes"][0]
                break
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(5 * (attempt + 1))
        time.sleep(1.1)  # the public OSRM server asks for gentle use
        path = simplify([[lat, lon] for lon, lat in best["geometry"]["coordinates"]], 0.012)
        _cache["road", *nodes] = (path, best["distance"] / 1000)
    return _cache["road", *nodes]


def sea(a: str, b: str) -> tuple[list[Point], float]:
    if ("sea", a, b) not in _cache:
        (lat1, lon1), (lat2, lon2) = coords(a), coords(b)
        found = searoute.searoute([lon1, lat1], [lon2, lat2], units="km", append_orig_dest=True)
        path = unwrap([[lat, lon] for lon, lat in found["geometry"]["coordinates"]])
        _cache["sea", a, b] = (simplify(path, 0.12), found["properties"]["length"])
    return _cache["sea", a, b]


def stretch(mode: str, nodes: tuple[str, ...]) -> dict[str, Any]:
    if mode == "rail":
        points = RAIL_NASHIK_JNPT
        km = sum(km_between(a, b) for a, b in pairwise(points))
        return {"mode": "rail", "km": round(km), "path": points}
    if mode == "road":
        path, km = road(nodes)
        return {"mode": "road", "km": round(km), "path": [[round(y, 3), round(x, 3)] for y, x in path]}
    path, km = [], 0.0
    for a, b in pairwise(nodes):
        part, part_km = sea(a, b)
        path += part if not path else part[1:]
        km += part_km
    return {"mode": "sea", "km": round(km), "path": [[round(y, 2), round(x, 2)] for y, x in unwrap(path)]}


def booked_stretches(origin: str, legs: list[str]) -> list[dict[str, Any]]:
    """A booked path, from its stored legs: road between road nodes, sea for everything else."""
    legs = [leg for leg in legs if leg != "NSK-JNPT"]
    start = "NASHIK" if legs and legs[0] == "KASARA_GHAT" and origin in ("INNSA", "INMUN") else origin
    nodes = [start, *legs]
    nodes = [n for i, n in enumerate(nodes) if i == 0 or n != nodes[i - 1]]
    runs: list[tuple[str, list[str]]] = []
    for a, b in pairwise(nodes):
        mode = "road" if a in ROAD_NODES and b in ROAD_NODES else "sea"
        if runs and runs[-1][0] == mode:
            runs[-1][1].append(b)
        else:
            runs.append((mode, [a, b]))
    return [stretch(mode, tuple(run)) for mode, run in runs]


def main() -> None:
    routes = {r.id: r for r in load_routes()}
    missing = set(routes) - set(ROUTE_PLANS)
    if missing:
        raise SystemExit(f"add these routes to ROUTE_PLANS: {sorted(missing)}")
    out: dict[str, Any] = {
        "note": "Road paths: OSRM, © OpenStreetMap contributors. Sea lanes: searoute (open marine network). "
                "Rail: hand-drawn through the real junctions.",
        "places": {code: {"name": n, "lat": la, "lon": lo, "kind": k} for code, (n, la, lo, k) in PLACES.items()},
        "routes": {},
        "booked": {},
    }
    for route_id, plan in ROUTE_PLANS.items():
        r = routes[route_id]
        out["routes"][route_id] = {"days": r.days + r.hold_days,
                                   "stretches": [stretch(p[0], tuple(p[1:])) for p in plan]}
        print(route_id, [(s["mode"], s["km"]) for s in out["routes"][route_id]["stretches"]])
    seen: set[str] = set()
    for c in JsonContainerRepo(LIVE_BOOKS).list_active():
        key = booked_key(c)
        if key not in seen:
            seen.add(key)
            out["booked"][key] = {"stretches": booked_stretches(c.origin_port, c.route_legs)}
            print(key, [(s["mode"], s["km"]) for s in out["booked"][key]["stretches"]])
    Path(GEO_PATH).write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {GEO_PATH} ({Path(GEO_PATH).stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
