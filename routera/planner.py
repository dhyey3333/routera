"""Scenario Planner: every action x destination for each at-risk container, scored by value.

value   = kg x realised price x quality - freight - paperwork - cold-store holding
quality = 0.7 at the buyer's minimum shelf life on arrival, rising to 1.0 with 10 more days;
          below the minimum the option is not feasible (fruit arrives soft or is rejected)
Containers are planned most-vulnerable first. Each pick adds volume to its market (saturation)
and uses a slot on its route (capacity), so the plan spreads load instead of everyone dumping in
Delhi. "Without ROUTERA" is the baseline: wait as booked if the fruit survives, else dump in Delhi.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from agents.sentinel.models import ContainerRef
from routera.compliance import ComplianceRules
from routera.market import HUMAN_FLOOR_RATIO, Market, realised_price
from routera.paperwork import change_set
from routera.reroute import Route, origin_node, unavailable
from routera.shelf_life import COLD_STORE_TEMP_C, ShelfLifeTracker

DAY_S = 86400


@dataclass
class Context:
    """What the open disruptions mean for planning (built by the orchestrator)."""

    now: datetime
    blocked: dict[str, str] = field(default_factory=dict)  # place -> why (closed port, road, full store)
    struck: dict[str, str] = field(default_factory=dict)  # carrier -> why
    crashes: dict[str, float] = field(default_factory=dict)  # market -> % below normal
    slow: dict[str, float] = field(default_factory=dict)  # congested place -> extra days
    doc_delay: dict[str, float] = field(default_factory=dict)  # facility -> paperwork backlog days
    delay: dict[str, float] = field(default_factory=dict)  # container -> wait-as-booked delay


def fmt_inr(value: float) -> str:
    sign = "-" if value < 0 else ""
    v = abs(value)
    if v >= 1e7:
        return f"{sign}₹{v / 1e7:.2f} Cr"
    if v >= 1e5:
        return f"{sign}₹{v / 1e5:.1f} L"
    return f"{sign}₹{v:,.0f}"


class Planner:
    def __init__(
        self,
        markets: Mapping[str, Market],
        routes: list[Route],
        rules: ComplianceRules,
        tracker: ShelfLifeTracker,
        cold_store_cost: Mapping[str, float],
    ) -> None:
        self.markets = markets
        self.routes = routes
        self.rules = rules
        self.tracker = tracker
        self.cold_store_cost = cold_store_cost  # facility -> Rs per tonne-day

    # --- one option ---------------------------------------------------------------------------

    def evaluate(
        self,
        c: ContainerRef,
        route: Route,
        life_now: float,
        ctx: Context,
        added_t: Mapping[str, float],
        used: Mapping[str, int],
    ) -> dict[str, Any]:
        market = self.markets[route.destination]
        tonnes = c.weight_kg / 1000
        option: dict[str, Any] = {
            "route_id": route.id,
            "label": route.label,
            "action": route.action,
            "destination": market.id,
            "destination_name": market.name,
            "kind": market.kind,
            "feasible": False,
            "reason": "",
            "arrival_days_left": None,
            "price_inr_per_kg": None,
            "saturation": None,
            "value_inr": None,
            "compliance": {"status": "ok", "note": ""},
            "human_only": False,
            "market_switch": market.id != c.destination_port,
        }
        check = self.rules.check(c, market.country)
        option["compliance"] = {"status": check.status, "note": check.note, "rules": list(check.rules)}
        if check.status == "block":
            option["reason"] = f"compliance block: {check.note}"
            return option
        why_not = unavailable(route, ctx.blocked, ctx.struck, used)
        if why_not:
            option["reason"] = why_not
            return option
        paper_days = check.extra_days + (ctx.doc_delay.get(c.facility_id or "", 0.0) if check.draft else 0)
        slow = sum(ctx.slow.get(place, 0.0) for place in route.via)
        segments = [(route.hold_days, COLD_STORE_TEMP_C), (route.days + slow + paper_days, route.temp_c)]
        arrival = self.tracker.project(c, life_now, segments)
        option["arrival_days_left"] = round(arrival, 2)
        if arrival < market.min_arrival_days:
            option["reason"] = (
                f"arrives with {max(arrival, 0):.1f} d left; buyers need {market.min_arrival_days:g}"
            )
            return option
        crash = 0.0 if route.sell_after_recovery else ctx.crashes.get(market.id, 0.0)
        price, sat = realised_price(market, added_t.get(market.id, 0.0), tonnes, crash, c.crop)
        quality = min(1.0, 0.7 + 0.3 * (arrival - market.min_arrival_days) / 10)
        hold_cost = self.cold_store_cost.get(route.hold_facility or "", 0.0) * tonnes * route.hold_days
        cost = route.cost_inr + check.extra_cost_inr + hold_cost
        option.update(
            feasible=True,
            price_inr_per_kg=round(price, 1),
            saturation=round(sat, 3),
            quality=round(quality, 3),
            cost_inr=round(cost),
            value_inr=round(c.weight_kg * price * quality - cost),
            human_only=check.status == "review" or price < market.price(c.crop) * HUMAN_FLOOR_RATIO,
            draft=check.draft,
        )
        option["paperwork"] = change_set(c, option, market, self.rules.lot(c)["phyto"])
        return option

    # --- the "without ROUTERA" baseline -------------------------------------------------------

    def baseline(
        self, c: ContainerRef, life_now: float, ctx: Context, crowd_t: float = 0.0
    ) -> dict[str, Any]:
        """What happens without ROUTERA: keep the booking if the fruit survives the delay, else
        dump in Delhi along with everyone else (crowd_t tonnes arriving at once)."""
        market = self.markets.get(c.destination_port)
        delay = ctx.delay.get(c.container_id, 0.0)
        remaining = max(0.0, (c.eta - ctx.now).total_seconds() / DAY_S)
        if market is not None:
            arrival = self.tracker.project(c, life_now, [(delay + remaining, 0.5)])
            if arrival >= market.min_arrival_days:
                price = market.price(c.crop) * (1 - ctx.crashes.get(market.id, 0.0) / 100)
                quality = min(1.0, 0.7 + 0.3 * (arrival - market.min_arrival_days) / 10)
                late = f", {delay:g} d late" if delay else ""
                return {
                    "kind": "wait",
                    "label": f"Keep the booking: sell in {market.name}{late}",
                    "value_inr": round(c.weight_kg * price * quality),
                    "arrival_days_left": round(arrival, 2),
                }
        dump = next(
            (r for r in self.routes if r.origin == origin_node(c) and r.destination == "MKT-AZADPUR"
             and r.action == "split_market"),
            None,
        )
        if dump is not None:
            crowd = Context(now=ctx.now, crashes=ctx.crashes)
            option = self.evaluate(c, dump, life_now, crowd, {"MKT-AZADPUR": crowd_t}, {})
            if option["feasible"]:
                return {
                    "kind": "dump",
                    "label": "Dump in Delhi (Azadpur) along with everyone else",
                    "value_inr": option["value_inr"],
                    "arrival_days_left": option["arrival_days_left"],
                }
        if market is None:
            return {"kind": "unmodelled", "label": "Keep the booking (no alternative modelled)",
                    "value_inr": 0, "arrival_days_left": None}
        salvage = round(c.weight_kg * self.markets["PROC-NSK"].price(c.crop) * 0.5)
        return {"kind": "salvage", "label": "Spoils in transit; salvage only", "value_inr": salvage,
                "arrival_days_left": 0}

    # --- the plan -------------------------------------------------------------------------------

    def plan(self, containers: Iterable[ContainerRef], ctx: Context) -> dict[str, Any]:
        boxes = list(containers)
        life = {c.container_id: self.tracker.estimate(c, ctx.now).days_left for c in boxes}
        added_t: dict[str, float] = defaultdict(float)
        used: dict[str, int] = defaultdict(int)
        crowd_t = sum(c.weight_kg for c in boxes) / 1000  # without ROUTERA everyone dumps at once
        rows = []
        for c in sorted(boxes, key=lambda b: (life[b.container_id], b.container_id)):  # most vulnerable first
            node = origin_node(c)
            domestic = c.destination_country == "IN"
            booked_name = self.markets[c.destination_port].name if c.destination_port in self.markets \
                else c.destination_port
            options = [
                self.evaluate(c, r, life[c.container_id], ctx, added_t, used)
                for r in self.routes
                if r.origin == node and (not domestic or self.markets[r.destination].kind != "export")
            ]
            base = self.baseline(c, life[c.container_id], ctx, crowd_t)
            if base["kind"] in ("wait", "unmodelled"):  # keeping the booking is an option too
                options.append({
                    "route_id": "STAY", "label": base["label"], "action": "keep_booking",
                    "destination": c.destination_port, "destination_name": booked_name,
                    "kind": "stay", "feasible": True, "reason": "",
                    "arrival_days_left": base["arrival_days_left"], "price_inr_per_kg": None,
                    "saturation": None, "value_inr": base["value_inr"], "cost_inr": 0,
                    "compliance": {"status": "ok", "note": "booking unchanged"},
                    "human_only": False, "market_switch": False, "paperwork": None,
                })
            feasible = [o for o in options if o["feasible"] and o["compliance"]["status"] != "review"]
            choice = max(feasible, key=lambda o: o["value_inr"]) if feasible else None
            if choice is not None and choice["route_id"] != "STAY":
                added_t[choice["destination"]] += c.weight_kg / 1000
                used[choice["route_id"]] += 1
            options.sort(key=lambda o: (not o["feasible"], -(o["value_inr"] or 0)))
            rows.append({
                "container_id": c.container_id,
                "crop": c.crop,
                "weight_kg": c.weight_kg,
                "days_left_now": round(life[c.container_id], 2),
                "booked_destination": c.destination_port,
                "booked_name": booked_name,
                "choice": choice,
                "baseline": base,
                "value_saved_inr": (choice["value_inr"] - base["value_inr"]) if choice else 0,
                "options": options,
            })
        return {"rows": rows, **summarise(rows)}


def by_place(found: Mapping[str, set[str]]) -> str:
    """'Singapore for GLF-23, NSK-05; Rotterdam for GRP-09'."""
    return "; ".join(f"{place} for {', '.join(sorted(ids))}" for place, ids in sorted(found.items()))


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Totals, the split by destination, paperwork drafts, guardrails and a one-line headline."""
    by_dest: dict[str, dict[str, Any]] = {}
    docs, papers, switches, human_only = [], [], 0, 0
    flagged: dict[str, dict[str, set[str]]] = {"block": defaultdict(set), "review": defaultdict(set)}
    for row in rows:
        choice = row["choice"]
        for o in row["options"]:  # which containers each block or review is about
            if o["compliance"]["status"] in flagged:
                flagged[o["compliance"]["status"]][o["destination_name"]].add(row["container_id"])
        if choice is None:
            continue
        stay = choice["route_id"] == "STAY"
        slot = by_dest.setdefault("STAY" if stay else choice["destination"], {
            "destination": "STAY" if stay else choice["destination"],
            "name": "Keep booking" if stay else choice["destination_name"],
            "kind": choice["kind"], "count": 0, "tonnes": 0.0, "value_inr": 0,
        })
        slot["count"] += 1
        slot["tonnes"] += row["weight_kg"] / 1000
        slot["value_inr"] += choice["value_inr"]
        switches += choice["market_switch"]
        human_only += choice["human_only"]
        if choice.get("paperwork"):
            papers.append(choice["paperwork"])
        if choice.get("draft"):
            docs.append({"container_id": row["container_id"], "to": choice["destination_name"],
                         "draft": choice["draft"]})
    value = sum(r["choice"]["value_inr"] for r in rows if r["choice"])
    baseline = sum(r["baseline"]["value_inr"] for r in rows)
    split = sorted(by_dest.values(), key=lambda s: -s["count"])
    guardrails = []
    if switches:
        guardrails.append(f"Human approves: {switches} market switch{'es' if switches > 1 else ''}")
    if human_only:
        guardrails.append(f"Human only: {human_only} container(s) below the floor price")
    if flagged["block"]:
        guardrails.append(f"Compliance blocks (verified law, cannot be overridden): {by_place(flagged['block'])}")
    if flagged["review"]:
        guardrails.append(
            f"Human review (failing rule not yet verified, never auto-picked): {by_place(flagged['review'])}")
    paperwork = {
        "containers": len(papers),
        **{k: sum(len(p[k]) for p in papers) for k in ("keep", "regenerate", "new", "obsolete")},
        "hours": max((p["hours"] for p in papers), default=0.0),
    }
    if papers:
        guardrails.append(
            f"Paperwork drafted by the agent: {paperwork['regenerate'] + paperwork['new']} documents for "
            f"{len(papers)} moves, ready in about {paperwork['hours']:g} h; a human signs and files them")
    moves = ", ".join(f"{s['count']} → {s['name']}" for s in split) or "no feasible moves"
    needs_approval = any(r["choice"] and r["choice"]["route_id"] != "STAY" for r in rows)
    if needs_approval:
        headline = f"{len(rows)} containers at risk. Plan: {moves}. Saves {fmt_inr(value - baseline)}."
    else:
        headline = f"{len(rows)} containers affected. Nothing beats keeping the bookings; monitoring."
    return {
        "needs_approval": needs_approval,
        "split": split,
        "value_inr": value,
        "baseline_inr": baseline,
        "saved_inr": value - baseline,
        "docs": docs,
        "paperwork": paperwork,
        "guardrails": guardrails,
        "headline": headline,
    }
