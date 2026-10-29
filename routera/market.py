"""Market: list prices, buyer minimums and the saturation formula.

realised price = list price x (1 - crash) x saturation
saturation     = max(floor, 1 - elasticity x (tonnes already sent + half this load) / weekly absorption)
So the more a plan sends to one market, the less each extra container earns there.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
HUMAN_FLOOR_RATIO = 0.6  # below 60% of list price only a human may decide (deck: "Human only")


@dataclass(frozen=True)
class Market:
    id: str
    name: str
    country: str
    kind: str  # export | domestic | processor
    price_inr_per_kg: float
    absorb_t: float
    elasticity: float
    floor_ratio: float
    min_arrival_days: float
    prices: dict[str, float] = field(default_factory=dict)  # per crop; grapes use price_inr_per_kg

    def price(self, crop: str) -> float:
        return self.prices.get(crop, self.price_inr_per_kg)


def load_markets(path: Path = DATA_DIR / "markets.json") -> dict[str, Market]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {m["id"]: Market(**m) for m in raw["markets"]}


def realised_price(
    m: Market, added_t: float, load_t: float, crash_pct: float = 0.0, crop: str = "grapes"
) -> tuple[float, float]:
    """(Rs/kg, saturation factor) for one more load of `load_t` tonnes."""
    if m.absorb_t <= 0 or m.elasticity <= 0:
        sat = 1.0
    else:
        sat = max(m.floor_ratio, 1 - m.elasticity * (added_t + load_t / 2) / m.absorb_t)
    return m.price(crop) * (1 - crash_pct / 100) * sat, sat
