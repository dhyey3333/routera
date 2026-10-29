"""Reroute: route / vessel / truck lookup plus a capacity and availability check."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from agents.sentinel.models import ContainerRef

DATA_DIR = Path(__file__).parent / "data"
GULF_PORTS = frozenset({"AEJEA", "AEAUH", "QAHMD", "SADMM", "KWSWK", "BHKBS"})


@dataclass(frozen=True)
class Route:
    id: str
    origin: str
    destination: str
    action: str
    label: str
    days: float
    cost_inr: float
    capacity: int
    via: tuple[str, ...] = ()
    carrier: str | None = None
    temp_c: float = 0.5
    hold_days: float = 0.0
    hold_facility: str | None = None
    sell_after_recovery: bool = False


def load_routes(path: Path = DATA_DIR / "routes.json") -> list[Route]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Route(**(r | {"via": tuple(r.get("via", ()))})) for r in raw["routes"]]


def origin_node(c: ContainerRef) -> str:
    """Where a container can be moved from: at sea (Gulf-bound), at a port, or in the grape belt."""
    if c.status == "at_sea":
        return "GULF_SEA" if c.destination_port in GULF_PORTS else "SEA"
    if c.status == "at_port":
        return c.origin_port
    return "NASHIK"


def unavailable(
    route: Route,
    blocked: Mapping[str, str],
    struck: Mapping[str, str],
    used: Mapping[str, int],
) -> str | None:
    """Why the route can't take another container right now, or None."""
    for place in (route.destination, *route.via):
        if place in blocked:
            return blocked[place]
    if route.carrier and route.carrier in struck:
        return struck[route.carrier]
    if used.get(route.id, 0) >= route.capacity:
        return f"no slots left (capacity {route.capacity})"
    return None
