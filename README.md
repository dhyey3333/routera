# ROUTERA (SAP Hackfest 2026, Team Code Catalyst)

**Agentic AI that saves perishable farm exports when shipping routes, markets or tariffs break
mid-season.** The deck's loop runs end to end: **Sense → Shelf-life clock → Decide → Approve & log**.

**India-inside is the core.** That is the leg ROUTERA can act on: pack house, truck, cold store,
Indian port, mandi and paperwork. Global shocks (Gulf closure, Red Sea reroute, US tariff) run on the
**same engine** as a second scenario layer.

> **All feeds, prices, capacities, containers and bookings are SIMULATED or ILLUSTRATIVE.**
> Validate the numbers in `routera/data/` and `agents/sentinel/data/` with a real exporter before
> relying on them.

## Run it

You need Python 3.11. Run these from this folder:

```bash
python -m pip install -r requirements.txt
```

```bash
python -m routera
```

This opens http://localhost:8000: the sign-in page, with the map rerouting a broken route behind it.
Give your name and role and press **Sign in**. It plays the India-inside Nashik corridor
scenario live at 60× (one simulated hour per real minute). Open **Sandbox tools** in the header to pick another
scenario and press **Replay**, or press **Simulate an event** to inject your own. The engine and every button work
offline; the map tiles, fonts and GSAP motion come from the internet, and the page runs without them.

### Signing in

The name you sign in with goes on the audit log next to every decision you make, with your role
("Asha K. (Export manager)"). These are not accounts: the team shares one access code. Set it before
starting the app, and only the sign-in page is reachable until someone types it:

```bash
export ROUTERA_ACCESS_CODE="something-your-team-shares"
```

Without it the control room is open, as before, and signing in only records who's at the desk. Five wrong
codes from one address lock it out for five minutes. A sign-in lasts one shift (12 hours) in a signed cookie;
set `ROUTERA_SESSION_SECRET` to keep people signed in across restarts.

| What | Command (`make` target) |
|---|---|
| Control room, port 8000 | `python -m routera` (`make routera`) |
| Tests: engine, app and Sentinel (372) | `python -m pytest routera/tests agents/sentinel/tests -q` (`make routera-test`) |
| Lint | `python -m ruff check routera agents/sentinel` (`make routera-lint`) |
| Sentinel console only, port 8001 | `python -m agents.sentinel.api --demo` (`make sentinel-demo`) |

### Live mode: real feeds

```bash
python -m routera --live
```

You can also pick **LIVE** in the scenario dropdown. The same Sentinel, planner and audit chain then run on real data, polled in the background:

| Feed | Source | Key | Becomes |
|---|---|---|---|
| Weather for Nashik and JNPT, 3-day forecast | Open-Meteo | none | `WEATHER_ALERT` when an IMD threshold is crossed (≥35.6 mm/day rain, hail, ≥40 °C) |
| Official disaster alerts (cyclones, floods, earthquakes) near any port, sea lane or corridor of the network | GDACS (UN and European Commission) | none | `WEATHER_ALERT` (source `gdacs`, reliability 0.90). Orange and red alerts confirm a disruption on their own; green ones only show as signals |
| Live road times, delays and alternative routes on 8 corridors, plus closures on the Nashik to JNPT road | TomTom Traffic | `TOMTOM_API_KEY` (free: 2,500 requests and 50,000 map tiles a day) | A closure becomes `ROAD_CLOSURE` (source `tomtom`, reliability 0.85), which confirms on its own, but only if it carries the highway's number, lies on the road and TomTom's own live route has gone around it: real data lists hundreds of closed side streets nearby. Delays of an hour or more slow the planner's routes through that road. The key stays on the server; map tiles come through `/api/traffic/tile/...` |
| Headlines on the corridor, JNPT, grapes, Red Sea / Hormuz | Google News RSS | none | news the Sentinel classifies (rules, or the LLM) |
| Ships off Rotterdam, JNPT, Mundra and Jebel Ali | AIS via aisstream.io (`pip install websockets`) | `AISSTREAM_API_KEY` (free, GitHub sign-up) | `PORT_CONGESTION` when ships at anchor pass a threshold (source `ais_port`, reliability 0.80: an estimate, not an official notice). aisstream.io is crowd-sourced and barely covers Indian and Gulf waters, so those ports usually show 0 ships; Rotterdam, where the mangoes go, is well covered |

Click any item on a feed card to see it on the map, which containers it touches, and what a closure there would mean. Only the feeds are real. The fleet is still the simulated container book (grapes, shrimp, pomegranate, banana and onion, plus Konkan mangoes in `routera/data/containers_mango.json`), moved so its ETAs and cut-offs start today, and bookings stay simulated. Mangoes age at their own reefer setpoint (12 °C; colder means chilling injury, not longer life). They also need hot water treatment for the EU and irradiation for the US, which ROUTERA drafts as a booking request. All mango prices and treatment rules are illustrative. On a quiet day, live mode correctly shows "no decision needed". Use **Feed the Sentinel** to add an event on top.

The LLM is off by default (`SENTINEL_LLM=off`). With it off, the keyword rules classify the news
and the summaries come from templates. See `agents/sentinel/README.md` to switch it on.

### Approvals on your real phone

Optional. A plan buzzes your phone in Telegram, with Approve, Reject and Why? buttons, and the big screen updates when you tap. Create a bot with @BotFather, set `TELEGRAM_BOT_TOKEN`, start the app, then send the `/pair` code shown in the Phone approvals panel. The bot only makes outbound calls, so the app needs no public address.

Only the paired private chat can decide, and only the plan version it was shown. Every decision is on the audit chain under the name of the Telegram account that paired, as "Asha K. via Telegram" (set `TELEGRAM_APPROVER` to fix one name). Set `TELEGRAM_CHAT_ID` to skip pairing after a restart. The on-screen buttons keep working, whichever comes first counts, and route changes stay on the big screen.

## What is an agent and what is an algorithm

Only the Sentinel is an LLM agent. Everything that decides money or compliance is deterministic
and explainable.

| Deck component | Here | How it works |
|---|---|---|
| **Disruption Sentinel** (LLM agent) | `agents/sentinel/` | Normalise, classify, merge, score with noisy-OR confidence, match containers, publish. The LLM only classifies free-text news and writes summaries; both are validated and fall back to rules. |
| **Shelf-Life Tracker** | `routera/shelf_life.py` | Q10 model: `rate = q10^((T − Tref)/10)`. `estimate()` gives life left now, `project()` gives life on arrival over (days, temp) legs. Reefer excursions use up extra life. |
| **Scenario Planner** | `routera/planner.py` | Scores every action × destination for every container at risk. Value = kg × saturated price × quality − costs, compared with what happens without ROUTERA. Most vulnerable boxes are allocated first, greedily, against capacity. |
| **Reroute** | `routera/reroute.py`, `data/routes.json` | Route lookup plus availability: blocked port, road or cold store, struck carrier, slots left. |
| **Market** | `routera/market.py`, `data/markets.json` | Saturation formula `price × max(floor, 1 − elasticity × (added + load/2) / absorb)`, plus mandi crashes. Below 60% of list price, only a human may choose. |
| **Compliance** | `routera/compliance.py`, `routera/rulebook.py`, `routera/rulepacks/` | Rules engine over the lot's facts and the team's YAML rule packs, each rule citing its law (EU 2016/2031, 7 CFR 319.56-3, ISPM 12 ...). **Only a verified rule may block**, and nobody can override a block (e.g. the EU residue limit). A failing rule that is not yet verified (e.g. the Singapore MRL) goes to **human review**: the planner never picks it, but a person may, and the approval logs it. Phyto, permits, treatments and traceability gaps are drafted for sign-off. `GET /api/rules/to-verify` lists every rule still to verify. |
| **Paperwork** | `routera/paperwork.py` | The team's document model ([routera_compliance](https://github.com/kokilacg30/routera_compliance)), fitted to ROUTERA: 12 trade documents with their standards (UN/CEFACT, ISPM 12, WCO, FIATA). Each move gets a change set (keep, regenerate, new, superseded) and a critical path (the slowest document). Approval drafts the filing pack and logs its sha256 on the chain. A field ROUTERA does not know, such as the new buyer, is left for a human, never invented. Cross-document checks cover weight, HS code, value and consignee. Each pack downloads as a ZIP with a cover sheet (summary, human checklist, legal basis), every document as JSON and a DRAFT-stamped PDF (`routera/filing.py`, reportlab), and a manifest of sha256 hashes. |
| **Live map** | `routera/geo.py`, `data/geo.json`, `routera/tools/build_geo.py` | Every container, placed from its ETA and moving with the clock, on an OpenStreetMap base. Click a box to see its booked route and every alternative, priced now on the live closures; click a route to change the plan. Real ships (AIS) and real alerts (GDACS, weather) are drawn on top. Road paths come from OSRM and sea lanes from searoute, generated once into `geo.json`. With a TomTom key, live mode adds traffic colours, live road times and live alternative roads. The fleet itself stays simulated. |
| **Escalation** | `routera/orchestrator.py` | From the team's approval policy: a pending plan waits on the operations lead for 2 h, then the export manager for 2 h, then the FPO director for 4 h, then it is an SLA breach. Each step is logged on the chain. **Silence is never consent**: a plan never approves itself. |
| **Approval & Audit** | `routera/orchestrator.py`, `routera/audit.py` | One plan per disruption, holding only the containers it hits, re-planned live. A box hit by two disruptions can sit in both plans until one of them moves it, and approved moves use up route slots and market room for later plans. Nothing moves until a human approves. Approval books SAP TM moves (simulated), drafts paperwork and logs the FPO payout. Every entry is sha256-chained to the previous one. |

## API (port 8000)

| Endpoint | Purpose |
|---|---|
| `GET /` | Intro and sign-in |
| `GET /room` | Control room, once signed in |
| `GET`, `POST`, `DELETE /api/session` | Who's on duty. `POST` body `{"name": "Asha K.", "role": "ops", "code": "..."}`, roles `ops`, `export`, `fpo`. With `ROUTERA_ACCESS_CODE` set, everything else answers 401 until this succeeds |
| `GET /sentinel/stream` | Server-sent events: `snapshot`, `reset`, `step`, `signal`, `disruption`, `routera` (fleet, plans, KPIs), `audit` |
| `POST /sentinel/events` | Inject a raw feed event |
| `POST /sentinel/replay/{scenario}?speed=60` | Start a scenario |
| `GET /api/state` | Full snapshot |
| `POST /api/plans/{id}/approve` | Body `{"overrides": {"NSK-03": "N-RAIL-SG"}}`. The approver is the signed-in person; `"approver"` in the body only counts when nobody is signed in. A compliance block returns 403. |
| `POST /api/plans/{id}/reject` | Body `{"approver": "...", "reason": "..."}` |
| `GET /api/plans/{id}/why?container=NSK-03` | "Ask why", answered from the plan's own numbers |
| `GET /api/plans/{id}/paperwork?container=NSK-03` | The draft filing pack for one moved container (`&route=` previews another option on a pending plan) |
| `GET /api/plans/{id}/filing-pack.zip?container=NSK-03` | The pack as a ZIP: cover sheet, JSON, PDFs, manifest |
| `GET /api/plans/{id}/paperwork/{doc_type}.pdf?container=NSK-03` | One draft as a PDF |
| `GET /api/map` | Places and real route geometry for the map |
| `GET /api/whatif?place=HORMUZ` | The plan if a place shut today. A preview: nothing is booked or logged |
| `POST /api/feeds/{name}/refresh` | Fetch one feed now (at most every 30 s) |
| `POST /api/decisions/open` | Body `{"place": "HORMUZ", "approver": "Asha K.", "reason": "..."}`. A named person treats a report as a real closure: a real plan opens, on the audit chain, and is approved like any other |
| `POST /api/decisions/close` | The undo: the place is open again, and a plan nobody approved closes by itself |
| `GET /api/traffic/tile/{z}/{x}/{y}.png` | Live traffic-flow tile, fetched by the server so the key stays private |
| `GET /api/containers/{id}/options` | Every route for one container right now, on the live closures |
| `GET /api/rules`, `GET /api/rules/to-verify` | Every rule with its citation and freshness; the rules still to verify, as Markdown |
| `GET /api/audit` | The whole chain, re-verified |
| `GET /docs` | OpenAPI |

## Scenarios

| Scenario | Layer | What it shows |
|---|---|---|
| `nashik_corridor_strike` | India-inside | Strike, Kasara ghat landslide, reefer excursion, full cold store and a Vashi price crash. The plan moves boxes to rail and Mundra and keeps the rest of the bookings. |
| `gulf_closure` | Global | Jebel Ali shuts. Boxes at sea go to Singapore or Colombo. Blocked lots and Malaysia (no import permit) are never chosen. |
| `gulf_ripple` | Global → India | The Gulf shock lands as domestic damage: each disruption gets its own plan. |
| `red_sea_reroute`, `us_tariff_spike` | Global | Reroute and tariff on the same engine. |
