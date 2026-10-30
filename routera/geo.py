"""The map's data: places and real route geometry, built once by routera/tools/build_geo.py."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from agents.sentinel.models import ContainerRef

GEO_PATH = Path(__file__).parent / "data" / "geo.json"


@lru_cache(maxsize=1)
def load_geo() -> dict[str, Any]:
    return json.loads(GEO_PATH.read_text(encoding="utf-8"))


def booked_key(c: ContainerRef) -> str:
    """Which stored path a container's booking follows."""
    return "|".join([c.origin_port, *c.route_legs])
