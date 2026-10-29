"""Shelf-Life Tracker: days of sellable life per container, now (estimate) and on arrival (project).

Temperature-driven decay (Q10 model): a day at the crop's reference temperature uses one day of
life; a day at T uses q10 ** ((T - T_ref) / 10) days. Crop parameters are ILLUSTRATIVE.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from agents.sentinel.models import ContainerRef


@dataclass(frozen=True)
class Crop:
    name: str
    ref_temp_c: float
    q10: float
    frozen: bool = False


CROPS = {
    "grapes": Crop("grapes", 0.5, 3.0),
    "pomegranate": Crop("pomegranate", 5.0, 2.5),
    "onion": Crop("onion", 1.0, 2.0),
    "banana": Crop("banana", 13.0, 2.5),
    "mangoes": Crop("mangoes", 12.0, 2.5),  # chilling-sensitive: reefers run at 10-13 C
    "shrimp": Crop("shrimp", -18.0, 2.0, frozen=True),
}
DEFAULT_CROP = Crop("produce", 2.0, 2.5)
UNTRACKED_DAYS = {True: 180.0, False: 30.0}  # frozen goods / fresh produce without a reading
COLD_STORE_TEMP_C = 1.0


def rate(crop: Crop, temp_c: float) -> float:
    """Days of life used per day at temp_c."""
    return float(crop.q10 ** ((temp_c - crop.ref_temp_c) / 10))


@dataclass(frozen=True)
class ShelfState:
    container_id: str
    days_left: float
    rate: float
    estimated: bool  # True when the tracker had no reading and used a default


class ShelfLifeTracker:
    """days_left in ContainerRef is the reading at `start`; time and excursions use it up."""

    def __init__(self, start: datetime) -> None:
        self.start = start
        self.extra_loss: dict[str, float] = {}

    @staticmethod
    def crop(c: ContainerRef) -> Crop:
        return CROPS.get(c.crop, DEFAULT_CROP)

    def estimate(self, c: ContainerRef, now: datetime) -> ShelfState:
        crop = self.crop(c)
        elapsed = max(0.0, (now - self.start).total_seconds() / 86400)
        estimated = c.days_left is None
        base = UNTRACKED_DAYS[crop.frozen] if c.days_left is None else c.days_left
        left = base - elapsed - self.extra_loss.get(c.container_id, 0.0)
        return ShelfState(c.container_id, round(left, 3), 1.0, estimated)

    def project(
        self, c: ContainerRef, left_now: float, segments: Sequence[tuple[float, float]]
    ) -> float:
        """Life left after (days, temp_c) segments, e.g. [(hold, 1.0), (transit, 0.5)]."""
        crop = self.crop(c)
        if crop.frozen:
            return left_now - sum(days for days, _ in segments)
        # A reefer is set to the crop's own setpoint: colder is chilling injury, not slower ageing.
        return left_now - sum(days * rate(crop, max(temp, crop.ref_temp_c)) for days, temp in segments)

    def record_excursion(self, c: ContainerRef, above_c: float, minutes: float) -> float:
        """Extra life used by `minutes` at `above_c` over the reference temperature."""
        crop = self.crop(c)
        extra = max(0.0, rate(crop, crop.ref_temp_c + above_c) - 1) * minutes / 1440
        self.extra_loss[c.container_id] = self.extra_loss.get(c.container_id, 0.0) + extra
        return extra
