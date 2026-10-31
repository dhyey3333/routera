"""Orchestrator: turns Sentinel disruptions into plans, waits for a human, then executes and logs.

On every disruption.* message it groups open disruptions that share containers into one
situation (a global shock and its domestic ripples become one plan), re-plans what is still
pending, and emits a fresh snapshot. Approval books the moves (SAP TM, simulated), drafts the
paperwork and logs decision and payout to the hash-chained audit.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from agents.sentinel.models import ContainerRef, ReeferTelemetryPayload
from agents.sentinel.sentinel import DisruptionSentinel
from routera.audit import HashChainAudit
from routera.compliance import FPO_BY_CROP, ComplianceRules
from routera.geo import booked_key
from routera.market import load_markets
from routera.paperwork import draft_pack
from routera.planner import Context, Planner, fmt_inr, summarise
from routera.reroute import load_routes
from routera.shelf_life import ShelfLifeTracker

PLANNING_STATUSES = ("confirmed", "active")
CONGESTION_DAYS = 3.0
PAPERWORK_BACKLOG_DAYS = 3.0
TYPE_LABEL = {
    "PORT_CLOSURE": "Port closure", "PORT_CONGESTION": "Port congestion",
    "LANE_SUSPENSION": "Lane suspension", "ROUTE_REROUTE": "Route reroute",
    "VESSEL_DELAY": "Vessel delay", "TARIFF_CHANGE": "Tariff change",
    "WEATHER_ALERT": "Weather alert", "ROAD_CLOSURE": "Road closure",
    "TRUCK_STRIKE": "Truck strike", "REEFER_FAILURE": "Reefer failure",
    "FACILITY_OUTAGE": "Facility outage", "COMPLIANCE_HOLD": "Compliance hold",
    "MARKET_PRICE_CRASH": "Mandi price crash",
}
# Who a pending plan waits on, and for how many hours, before it escalates to the next person. From the
# team's approval policy: silence or a timeout is never consent, so a plan only ever escalates.
ESCALATION = (("Operations lead (Sunil P.)", 2.0), ("Export manager", 2.0), ("FPO director", 4.0))
Emit = Callable[[str, dict[str, Any]], None]


class Orchestrator:
    def __init__(
        self, sentinel: DisruptionSentinel, chain: HashChainAudit, emit: Emit, llm: Any = None
    ) -> None:
        self.sentinel = sentinel
        self.chain = chain
        self.emit = emit
        self.llm = llm
        self.clock = sentinel.clock
        self.tracker = ShelfLifeTracker(self.clock.now())
        costs = {
            f.facility_id: f.cost_inr_per_t_day or 0.0
            for f in sentinel.network.facilities
            if f.kind == "cold_store"
        }
        self.planner = Planner(load_markets(), load_routes(), ComplianceRules(), self.tracker, costs)
        self.plans: dict[str, dict[str, Any]] = {}
        self.plan_for_root: dict[str, str] = {}
        self.assigned: dict[str, dict[str, Any]] = {}
        self.bookings: list[dict[str, Any]] = []
        self._excursions: set[str] = set()
        self.live_slow: dict[str, float] = {}  # place -> extra days from live road traffic

    # --- inputs ---------------------------------------------------------------------------------

    def containers(self) -> dict[str, ContainerRef]:
        return {c.container_id: c for c in self.sentinel.tracker.containers.list_active()}

    def _incidents(self, statuses: tuple[str, ...] = PLANNING_STATUSES) -> list[Any]:
        return [
            i for i in self.sentinel.tracker.incidents
            if i.is_open and i.disruption.status in statuses
        ]

    def context(self) -> Context:
        ctx = Context(now=self.clock.now())
        network = self.sentinel.network
        for incident in self._incidents():
            d = incident.disruption
            tag = f"{TYPE_LABEL[d.type].lower()} at {d.location} ({d.disruption_id})"
            if d.type in ("PORT_CLOSURE", "LANE_SUSPENSION", "FACILITY_OUTAGE"):
                ctx.blocked[d.location] = tag
            elif d.type == "ROAD_CLOSURE":
                for place in {d.location, *network.corridors_through(d.location)}:
                    ctx.blocked[place] = tag
            elif d.type == "TRUCK_STRIKE":
                for union in {s.scope for s in incident.signals if s.type == "TRUCK_STRIKE" and s.scope}:
                    for carrier in network.union_members(union):
                        ctx.struck[carrier] = tag
            elif d.type == "PORT_CONGESTION":
                ctx.slow[d.location] = CONGESTION_DAYS
            elif d.type == "MARKET_PRICE_CRASH":
                ctx.crashes[d.location] = d.cost_impact_pct
            elif d.type == "COMPLIANCE_HOLD":
                ctx.doc_delay[d.location] = PAPERWORK_BACKLOG_DAYS
            for cid in d.affected_container_ids:
                ctx.delay[cid] = max(ctx.delay.get(cid, 0.0), d.expected_delay_days)
        for place, days in self.live_slow.items():
            ctx.slow[place] = ctx.slow.get(place, 0.0) + days
        return ctx

    def set_live_slow(self, slow: dict[str, float]) -> None:
        """Live road delays feed the arrival times of pending plans; re-plan only when they change."""
        if slow != self.live_slow:
            self.live_slow = slow
            self.refresh()

    def _record_excursions(self, boxes: dict[str, ContainerRef]) -> None:
        for incident in self.sentinel.tracker.incidents:
            for s in incident.signals:
                p = s.payload
                if s.type != "REEFER_FAILURE" or not isinstance(p, ReeferTelemetryPayload):
                    continue
                if s.event_id in self._excursions:
                    continue
                self._excursions.add(s.event_id)
                for cid in incident.disruption.affected_container_ids:
                    if cid in boxes:
                        self.tracker.record_excursion(boxes[cid], p.temp_c - p.setpoint_c, p.minutes_above)

    def situations(self) -> list[list[Any]]:
        """Open disruptions grouped by shared containers (union-find)."""
        incidents = [i for i in self._incidents() if i.disruption.affected_container_ids]
        parent = list(range(len(incidents)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        owner: dict[str, int] = {}
        for idx, incident in enumerate(incidents):
            for cid in incident.disruption.affected_container_ids:
                if cid in owner:
                    parent[find(idx)] = find(owner[cid])
                else:
                    owner[cid] = idx
        groups: dict[int, list[Any]] = {}
        for idx, incident in enumerate(incidents):
            groups.setdefault(find(idx), []).append(incident)
        return list(groups.values())

    # --- planning -------------------------------------------------------------------------------

    def on_bus(self, topic: str, message: dict[str, Any]) -> None:
        self.refresh()

    def refresh(self) -> None:
        boxes = self.containers()
        self._record_excursions(boxes)
        ctx = self.context()
        live: dict[str, str] = {}  # pending plan id -> the plan that now covers it
        for group in self.situations():
            ids = sorted(i.disruption_id for i in group)
            known = [self.plans[p] for p in dict.fromkeys(self.plan_for_root.get(i) for i in ids) if p]
            plan = next((p for p in known if p["status"] == "pending"), None)
            decided = {cid for p in known if p["status"] != "pending" for cid in p["container_ids"]}
            pending = sorted(
                {cid for i in group for cid in i.disruption.affected_container_ids}
                - set(self.assigned) - decided
            )
            pending = [cid for cid in pending if cid in boxes]
            if not pending:
                continue
            built = self.planner.plan([boxes[cid] for cid in pending], ctx)
            if plan is None:
                plan = {"plan_id": f"PLN-{len(self.plans) + 1:04d}", "version": 0, "status": "pending",
                        "created_at": ctx.now.isoformat()}
                self.plans[plan["plan_id"]] = plan
            for other in known:  # a plan whose situation merged into this one
                if other["status"] == "pending" and other is not plan:
                    live[other["plan_id"]] = plan["plan_id"]
            live[plan["plan_id"]] = plan["plan_id"]
            for disruption_id in ids:
                self.plan_for_root[disruption_id] = plan["plan_id"]
            if plan.get("rows") != built["rows"] or plan.get("disruption_ids") != ids:
                plan["version"] += 1
                plan["updated_at"] = ctx.now.isoformat()
            root_view = next(i.disruption for i in group if i.disruption_id == ids[0])
            linked = f" + {len(ids) - 1} linked" if len(ids) > 1 else ""
            plan.update(built, disruption_ids=ids, container_ids=pending,
                        title=f"{TYPE_LABEL[root_view.type]} · {root_view.location}{linked}")
        for plan in self.plans.values():
            if plan["status"] != "pending" or live.get(plan["plan_id"]) == plan["plan_id"]:
                continue
            plan["status"] = "closed"
            target = live.get(plan["plan_id"])
            plan["closed_reason"] = (
                f"merged into {target}" if target else "the disruption cleared before anyone had to act"
            )
        self.emit("routera", self.snapshot())

    # --- the human decision ---------------------------------------------------------------------

    def approve(
        self, plan_id: str, approver: str, overrides: dict[str, str] | None = None
    ) -> dict[str, Any]:
        plan = self.plans[plan_id]
        if plan["status"] != "pending":
            raise ValueError(f"{plan_id} is {plan['status']}, not pending")
        rows = {r["container_id"]: r for r in plan["rows"]}
        reviewed: dict[str, list[str]] = {}
        for cid, route_id in (overrides or {}).items():
            row = rows[cid]
            option = next(o for o in row["options"] if o["route_id"] == route_id)
            if option["compliance"]["status"] == "block":
                raise PermissionError(f"{cid}: compliance block cannot be overridden ({option['compliance']['note']})")
            if not option["feasible"]:
                raise ValueError(f"{cid}: {option['reason']}")
            row["choice"] = option
            row["value_saved_inr"] = option["value_inr"] - row["baseline"]["value_inr"]
            row["overridden_by"] = approver
            if option["compliance"]["status"] == "review":
                reviewed[cid] = option["compliance"]["rules"]
        plan.update(summarise(plan["rows"]))
        now = self.clock.now().isoformat()
        created = []
        for row in plan["rows"]:
            if row["choice"] is None or row["choice"]["route_id"] == "STAY":
                continue  # keeping the booking needs no new booking
            booking = {
                "booking_id": f"TM-{len(self.bookings) + 1:05d}",
                "container_id": row["container_id"],
                "plan_id": plan_id,
                "route_id": row["choice"]["route_id"],
                "label": row["choice"]["label"],
                "destination": row["choice"]["destination"],
                "destination_name": row["choice"]["destination_name"],
                "action": row["choice"]["action"],
                "at": now,
            }
            self.bookings.append(booking)
            self.assigned[row["container_id"]] = booking
            created.append(booking["booking_id"])
        plan.update(status="approved", approved_by=approver, approved_at=now)
        moves = ", ".join(f"{s['count']} → {s['name']}" for s in plan["split"])
        checked = f"; reviewed unverified rule(s) for {', '.join(reviewed)}" if reviewed else ""
        self.chain.log(approver, "plan.approved", {"plan_id": plan_id, "split": plan["split"],
                       "overrides": overrides or {}, "reviewed_rules": reviewed},
                       f"{approver} approved {plan_id}: {moves}; saves {fmt_inr(plan['saved_inr'])}{checked}")
        self.chain.log("reroute", "booking.created", {"plan_id": plan_id, "bookings": created,
                       "system": "SAP TM (simulated)"},
                       f"{len(created)} moves booked in SAP TM (simulated) for {plan_id}")
        packs = {r["container_id"]: self.paperwork(plan_id, r["container_id"]) for r in plan["rows"]
                 if r["choice"] and r["choice"]["route_id"] != "STAY"}
        plan["packs"] = {cid: p["sha256"] for cid, p in packs.items()}
        if packs:
            papers = plan["paperwork"]
            drafted = sum(len(p["documents"]) for p in packs.values())
            to_fill = sum(len(p["to_fill"]) for p in packs.values())
            self.chain.log("compliance", "documents.drafted",
                           {"plan_id": plan_id, "packs": plan["packs"], "documents": drafted,
                            "regenerate": papers["regenerate"], "new": papers["new"],
                            "superseded": papers["obsolete"], "critical_path_h": papers["hours"],
                            "fields_to_fill": to_fill},
                           f"{drafted} documents drafted for {len(packs)} moves ({papers['regenerate']} regenerated, "
                           f"{papers['new']} new, {papers['obsolete']} superseded), ready in about "
                           f"{papers['hours']:g} h; {to_fill} field(s) left for a human; pack hashes logged")
        fpos = ", ".join(sorted({FPO_BY_CROP.get(r["crop"], "Grower FPO (SIM)") for r in plan["rows"]}))
        self.chain.log("approval_audit", "payout.logged",
                       {"plan_id": plan_id, "fpo": fpos, "value_saved_inr": plan["saved_inr"]},
                       f"{fmt_inr(plan['saved_inr'])} value saved credited to {fpos} growers")
        self.refresh()
        return plan

    def reject(self, plan_id: str, approver: str, reason: str) -> dict[str, Any]:
        plan = self.plans[plan_id]
        if plan["status"] != "pending":
            raise ValueError(f"{plan_id} is {plan['status']}, not pending")
        plan.update(status="rejected", rejected_by=approver, reject_reason=reason or "no reason given")
        self.chain.log(approver, "plan.rejected", {"plan_id": plan_id},
                       f"{approver} rejected {plan_id}: {plan['reject_reason']}")
        self.refresh()
        return plan

    def paperwork(self, plan_id: str, container_id: str, route_id: str | None = None) -> dict[str, Any]:
        """The draft filing pack for one moved container. A pending plan can preview another route;
        an approved plan's pack is rebuilt as approved, so its hash matches the one on the chain."""
        plan = self.plans[plan_id]
        row = next(r for r in plan["rows"] if r["container_id"] == container_id)
        approved = plan["status"] == "approved"
        option = row["choice"]
        if route_id and not approved:
            option = next(o for o in row["options"] if o["route_id"] == route_id)
        if option is None or option["route_id"] == "STAY":
            raise ValueError(f"{container_id} keeps its booking: no new paperwork")
        c = self.containers()[container_id]
        market = self.planner.markets[option["destination"]]
        route = next(r for r in self.planner.routes if r.id == option["route_id"])
        rules = self.planner.rules
        at = datetime.fromisoformat(plan["approved_at"]) if approved else self.clock.now()
        citations = [r.cite() for r in rules.book.applicable(market.country, c.crop)]
        pack = draft_pack(c, option, market, route, rules.lot(c), rules.exporters.get(c.crop, {}),
                          rules.markets.get(market.country, {}).get("mrl_standard"),
                          rules.treatment_note(c, market.country), plan_id, plan["title"], at, approved,
                          citations)
        pack["on_chain"] = plan.get("packs", {}).get(container_id)
        return pack

    def options(self, container_id: str) -> dict[str, Any]:
        """Every route for one container right now, priced on the live closures: any box, any time."""
        row = self.planner.plan([self.containers()[container_id]], self.context())["rows"][0]
        return {**row, "booking": self.assigned.get(container_id)}

    def escalation(self, plan: dict[str, Any], now: datetime) -> dict[str, Any]:
        """Who the plan waits on now. Past the last stage it is an SLA breach, and it still waits."""
        created = datetime.fromisoformat(plan["created_at"])
        elapsed = (now - created).total_seconds() / 3600
        edge, level = 0.0, len(ESCALATION) - 1
        for i, (_, hours) in enumerate(ESCALATION):
            edge += hours
            if elapsed <= edge:
                level = i
                break
        role = ESCALATION[level][0]
        return {"level": level, "role": role, "elapsed_h": round(elapsed, 2), "breached": elapsed > edge,
                "due_at": (created + timedelta(hours=edge)).isoformat(),
                "next": ESCALATION[level + 1][0] if level + 1 < len(ESCALATION) else None}

    def _escalate(self, now: datetime) -> None:
        for plan in self.plans.values():
            if plan["status"] != "pending":
                continue
            before = plan.get("escalation") or {"level": 0, "breached": False}
            now_at = plan["escalation"] = self.escalation(plan, now)
            if now_at["level"] > before["level"] or (now_at["breached"] and not before["breached"]):
                what = "SLA breached; still waiting on" if now_at["breached"] else "escalated to"
                self.chain.log("approval_audit", "plan.escalated",
                               {"plan_id": plan["plan_id"], **now_at},
                               f"{plan['plan_id']} waited {now_at['elapsed_h']:g} h without a decision: {what} "
                               f"{now_at['role']}. Silence is never consent; nothing moves until someone approves")

    def why(self, plan_id: str, container_id: str) -> dict[str, str]:
        """'Ask why', answered from plan data (an LLM may rephrase it; numbers are kept)."""
        row = next(r for r in self.plans[plan_id]["rows"] if r["container_id"] == container_id)
        base, choice = row["baseline"], row["choice"]
        lines = [f"{container_id} has about {row['days_left_now']:.1f} days of shelf life left."]
        if choice and choice["route_id"] == "STAY":
            lines.append(f"Plan: keep the booking ({choice['label']}), worth {fmt_inr(choice['value_inr'])}.")
        elif choice:
            lines.append(
                f"Plan: {choice['label']}. It arrives with {choice['arrival_days_left']:.1f} days "
                f"left, earns about ₹{choice['price_inr_per_kg']:g}/kg after saturation "
                f"({choice['saturation'] * 100:.0f}% of list price), worth {fmt_inr(choice['value_inr'])}."
            )
        else:
            lines.append("No modelled route beats holding course.")
        lines.append(f"Without ROUTERA: {base['label']}, worth {fmt_inr(base['value_inr'])}.")
        others = [o for o in row["options"] if not choice or o["route_id"] != choice["route_id"]][:4]
        for o in others:
            verdict = fmt_inr(o["value_inr"]) if o["feasible"] else f"not possible: {o['reason']}"
            lines.append(f"{o['destination_name']} ({o['label']}): {verdict}.")
        return {"text": " ".join(lines), "by": "template"}

    # --- output ---------------------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        now = self.clock.now()
        self._escalate(now)
        ctx = self.context()
        boxes = self.containers()
        open_ids: dict[str, list[str]] = {}
        for incident in self._incidents(("confirmed", "active", "resolving")):
            for cid in incident.disruption.affected_container_ids:
                open_ids.setdefault(cid, []).append(incident.disruption_id)
        fleet = []
        for c in boxes.values():
            state = self.tracker.estimate(c, now)
            fleet.append({
                "container_id": c.container_id, "crop": c.crop, "weight_kg": c.weight_kg,
                "status": c.status, "origin": c.origin_port, "destination": c.destination_port,
                "destination_name": self.planner.markets[c.destination_port].name
                if c.destination_port in self.planner.markets else c.destination_port,
                "facility": c.facility_id, "carrier": c.carrier_id, "vessel": c.vessel_id,
                "days_left": state.days_left, "rate": state.rate, "estimated": state.estimated,
                "affected_by": open_ids.get(c.container_id, []),
                "booking": self.assigned.get(c.container_id),
                "path": booked_key(c), "eta": c.eta.isoformat(),
                "cutoff_at": c.cutoff_at.isoformat() if c.cutoff_at else None,
            })
        fleet.sort(key=lambda f: (not f["affected_by"] or bool(f["booking"]), f["days_left"]))
        plans = sorted(self.plans.values(), key=lambda p: (p["status"] != "pending", p["plan_id"]))
        at_risk = [f for f in fleet if f["affected_by"] and not f["booking"]]
        return {
            "as_of": now.isoformat(),
            "fleet": fleet,
            "plans": plans,
            "bookings": self.bookings[-50:],
            "map": {"blocked": ctx.blocked, "slow": ctx.slow},
            "kpis": {
                "containers": len(fleet),
                "at_risk": len(at_risk),
                "min_days_left": min((f["days_left"] for f in at_risk), default=None),
                "on_table_inr": sum(p["saved_inr"] for p in plans if p["status"] == "pending"),
                "protected_inr": sum(p["saved_inr"] for p in plans if p["status"] == "approved"),
                "pending_plans": sum(p["status"] == "pending" for p in plans),
                "rebooked": len(self.assigned),
            },
            "chain": {**self.chain.verify(), "tail": self.chain.tail(80)},
        }
