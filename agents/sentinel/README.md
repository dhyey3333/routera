# Disruption Sentinel · ROUTERA (SAP Hackfest 2026)

The Sentinel turns a stream of raw, duplicated, sometimes wrong signals into a small set of
trustworthy **Disruption** objects: type, severity, confidence, expected delay and the exact
containers at risk. It publishes them on the bus for the orchestrator and writes every decision to
the audit log with a reason.

**India-inside is the core.** Inside India ROUTERA can act: pick another port, truck, cold store or
market, and fix the paperwork. So the Sentinel's main layer watches the domestic leg (farm → pack
house → pre-cool → truck → port): truck strikes, road closures, rain and hail, reefer failures,
pack-house and cold-store trouble, phyto / residue delays and mandi price crashes. Global shocks
(port closures abroad, reroutes, tariffs) run **on the same engine** as a second layer, and the
chained scenario shows how a global shock lands as domestic damage.

**All feeds are simulated.** Every scenario, outlet, carrier, facility, price and container in
`data/` is illustrative, and every output carries `simulated: true`. India-inside transit times,
cut-offs, capacities, costs and prices are assumptions to validate with a real exporter.

Rules and math decide; the LLM only explains. Classification of structured events, confidence,
delay, severity and container matching are deterministic Python. An LLM is optional (off by
default) and only classifies free-text news and writes one-line summaries, both validated, with a
fallback to keyword rules and a template.

---

## Run it

From the monorepo root (the folder that contains `agents/`), with Python 3.11:

```bash
python -m pip install -r agents/sentinel/requirements.txt
```

| What | Command (`make` target in brackets) |
|---|---|
| Demo: UI at http://localhost:8001, plays the India-inside corridor scenario | `python -m agents.sentinel.api --demo` (`make sentinel-demo`) |
| Print every decision of the India-inside scenario at 60× | `python -m agents.sentinel.replay nashik_corridor_strike --speed 60` (`make sentinel-replay`) |
| Replay all five scenarios and compare with `data/expected/` | `python -m agents.sentinel.replay all --speed 0 --check` (`make sentinel-check`) |
| Tests | `python -m pytest agents/sentinel/tests -q` (`make sentinel-test`) |
| Lint | `python -m ruff check agents/sentinel` and `python -m ruff format --check agents/sentinel` (`make sentinel-lint`) |

No `make` on Windows: run the command itself.

**Demo notes**

- The replay starts when the page connects, so the audience sees it from the first signal. The
  server opens a browser tab; `--no-browser` skips that. `--demo gulf_closure`, `--speed 10` or
  `--port 8080` change the defaults.
- Speed is simulated minutes per real minute: at 60× the India-inside scenario takes about 40 s,
  the chained Gulf ripple about 25 s. Fast-forwards (an hour or more of simulated time) wait at
  most 3 s.
- The dropdown, 1×/10×/60× and Replay restart on any scenario. Reloading the page restores the
  full state.
- Offline: with `SENTINEL_LLM=off` (the default) nothing touches the network, and the page loads
  no external fonts or scripts (a test checks this).

**Suggested 10-minute demo**

1. **Main (about 3 min): `nashik_corridor_strike`.** Strike rumour → rain over Nashik → union
   strike notice → Kasara ghat landslide → reefer excursion on a stuck truck → Nashik cold store
   full → Vashi price crash → strike starts → ghat reopens → strike called off → resolved.
2. **Same engine, global shock (about 1 min): `gulf_closure` or `us_tariff_spike`.**
3. **Chained (optional): `gulf_ripple`.** Jebel Ali closes → JNPT reefer plugs fill → Nashik cold
   store fills → phyto re-inspection backlog → Delhi mandi glut. Cards marked "Shares containers
   with" show the layers connecting.

---

## The five scenarios

| Scenario | Layer | Result |
|---|---|---|
| `nashik_corridor_strike` | India-inside | The strike rumour is held as a candidate until the union notice confirms it. A rain alert is **not** allowed to confirm a strike. TRUCK_STRIKE: severity 4, urgent, 16 loads; the 12 EU boxes miss their vessel cut-off, so 3 d of strike costs 10 d. ROAD_CLOSURE at the Kasara ghat: severity 3, 1 d → 8 d via the cut-off. Reefer failure (2), cold store full (2), Vashi price crash −38.7% (2). The strike and closure resolve when the union calls it off and the ghat reopens. |
| `gulf_ripple` | India-inside (chained) | One global PORT_CLOSURE at AEJEA (severity 4) and four domestic knock-ons: JNPT congestion (3), Nashik cold store full (2), phyto re-inspection hold (3), Delhi mandi crash −37.9% (2). Three of the four share containers with the closure. |
| `gulf_closure` | Global | One PORT_CLOSURE at AEJEA from 5 signals: severity 5, urgent, exactly the 12 Gulf grape containers. The Singapore rumour expires as "ignored – uncorroborated". |
| `us_tariff_spike` | Global | TARIFF_CHANGE US:0306, +40 pts, severity 3, the 9 US-bound shrimp containers that land after the effective date. A malformed message is rejected and audited. |
| `red_sea_reroute` | Global | One ROUTE_REROUTE at SUEZ, severity 3, the 4 Europe-bound containers routed through Suez. |

`data/expected/*.json` are hand-labelled, not generated by the code, down to each step's audit
actions, bus messages, layer and links. Each severity comes with its derivation. `--check` and the
tests compare against them.

---

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/sentinel/events` | Ingest one RawEvent. `received_at` defaults to the simulated clock. 422 if malformed (still audited). |
| `GET` | `/sentinel/disruptions?scope=active\|all` | Disruption contract objects (default: confirmed, active and resolving only). |
| `GET` | `/sentinel/stream` | Server-sent events: a `snapshot` first, then `signal`, `decision`, `publish`, `disruption`, `step`, `replay`, `reset`. |
| `POST` | `/sentinel/replay/{name}?speed=60` | Restart on a scenario in the background. |
| `GET` | `/sentinel/state` | The same snapshot the stream starts with. |
| `GET` | `/sentinel/scenarios` | Scenario names and titles. |
| `POST` | `/sentinel/tick` | Run the clock-driven lifecycle now (expiry, start times, resolution). |

`disruption` events and the snapshot also carry `layer` (`india` / `global`), `options` (the
planner's option families) and `linked` (disruptions sharing containers). Before any replay the
API uses both container books, so POSTed events of either layer match containers.

```bash
curl -s -X POST http://localhost:8001/sentinel/events -H "Content-Type: application/json" -d '{"event_id":"X-1","source":"road_authority","payload":{"segment":"Kasara ghat","status":"closed","effective_from":"2026-03-02T06:40:00Z"}}'
```

---

## How it works

| Step | File | What it does |
|---|---|---|
| 1 Normalise | `normalise.py` | Validates the payload for its source; maps places to canonical codes (UN/LOCODE, chokepoints, regions like GULF / NASHIK, road segments, facilities, markets, `US:0306`). Malformed → rejected and audited. |
| 2 Classify | `classify.py` | Structured feeds use the rule tables below. News goes to the LLM (validated) or to keyword rules. |
| 3 Dedupe / merge | `merge.py` | Folds signals about the same event into one disruption (rules below). |
| 4 Confidence | `scoring.py` | Noisy-OR, `1 − Π(1 − r)`, over distinct publishers. |
| 5 Lifecycle | `merge.py` | candidate → confirmed → active → resolving → resolved; candidates expire after 24 h. |
| 6 Matching | `matching.py` | Which containers each disruption affects; only confirmed (or later) disruptions list them. |
| 7–8 Delay, severity | `scoring.py` | Spec 3.3 and 3.4, plus the India-inside rules below. |
| 9 Summary | `summary.py` | Validated LLM sentence or the spec template. |
| 10 Publish + audit | `sentinel.py` | Bus only on material change; every decision audited. |

Every weight, threshold, keyword, place alias, option family and domestic default is in
`config/sentinel.yaml`. The reference network (corridors, carriers, facilities, markets) is in
`data/network.json`.

### India-inside rule table (v2)

| Feed (source) | Becomes | Location | Affects | Delay (default when no end is given) |
|---|---|---|---|---|
| `road_authority` closed | ROAD_CLOSURE | road segment (KASARA_GHAT) | loads whose road ahead uses the segment or its corridor | until reopen, else 1 d |
| `transport_union` strike | TRUCK_STRIKE | region (NASHIK); scope = union | loads moved by the union's carriers (network) | announced length, else 3 d |
| `reefer_telemetry` ≥ +3 °C for ≥ 30 min, or breakdown | REEFER_FAILURE | truck | the load on that truck | 0.5 d swap + shelf life burnt (0.03 d per °C·h) |
| `facility_status` power_failure / full | FACILITY_OUTAGE | facility (CS-NSK) | power: loads held there; full: loads waiting at pack houses in that region | until restore, else 0.5 d; full: 1 d |
| `compliance_feed` delayed / held / failed | COMPLIANCE_HOLD | facility or port; scope = market | export loads at or passing it, for that market (or any) | until cleared, else 3 d; failed → market unreachable |
| `mandi_prices` ≥ 20% under baseline | MARKET_PRICE_CRASH | market (MKT-VASHI) | loads bound for that market | 0 d; cost impact = % drop |
| `weather` orange / red over a region | WEATHER_ALERT | region | loads at its facilities or on its corridors | 1 d / 3 d |

The reopen / call-off / back-in-range / "ok" / cleared / recovered messages from the same feeds
resolve these disruptions.

**Vessel cut-off.** For road closures, strikes, reefer failures, compliance holds and port
congestion, a load whose delay pushes it past its port gate-in cut-off (`ContainerRef.cutoff_at`)
waits for the next sailing (+7 d, `next_sailing_gap`). That is why a 1-day landslide costs 8 days.
Weather, facility outages and the global types don't add it.

### Decisions where the spec was silent (assumptions to tune with the team)

- **Carrier advisory `action=delay`** becomes a `VESSEL_DELAY` on the lane, matched to the listed
  vessels (the "lane delay" of the Gulf timeline).
- **Merging beyond the exact fingerprint.** Within 72 h a signal joins an open disruption when
  (strongest first):
  - it has the same fingerprint;
  - it is a port closure, congestion or weather alert at the same port, or at the destination of a
    delayed lane;
  - it is a vessel delay on a ship the disruption already covers;
  - it is a news report of the same kind of event nearby;
  - it is a structured signal inside the place of a news-only candidate.

  "Same kind" means the same `families` entry in the config, so a rain alert never corroborates a
  strike rumour. The highest-ranked structured type wins (`type_rank`).
- **News never confirms on its own.** Repeats from one outlet count once.
- **Confirmed vs active.**
  - Port authority, tariff, road authority and union notices are active from their stated start.
  - Weather is active from `valid_from`.
  - Telemetry, facility, compliance and price feeds are active from arrival.
- **Candidates show severity 1** and no containers until confirmed. A rumour the authority
  contradicts is dropped as "refuted".
- **Supporting signals don't widen the affected set.**
- **Material change** for republishing: status, severity, affected set, delay moved by at least a
  day, plus type and location.
- **Rounding** is half up.
- **Data choices behind the acceptance numbers:**
  - Gulf severity 5 needs a median `days_left` under about 14.7 (the Round 1 deck's 16–24 days
    would give 4).
  - A tariff can't exceed 3 with the spec's weights.
  - The India-inside EU boxes are seeded at 10–14 days left (late-season boxes held for a booking).
    The domestic truckloads have 6–9 days.
- **Domestic severity depends on the cut-off rule.** Without it the strike drops from 4/urgent to
  2/normal (a test checks this). Validate `next_sailing_gap` and real cut-offs with an exporter.

---

## Optional LLM

Set `SENTINEL_LLM`, or pass `--llm` to either command. The chain is SAP Generative AI Hub →
Anthropic → OpenAI → none; the value picks where it starts, and a provider whose SDK or
credentials are missing is skipped (printed at start-up).

| `SENTINEL_LLM` | Provider | Model (override) | Credentials |
|---|---|---|---|
| `off` (default) | none | n/a | n/a |
| `genai_hub` | SAP Generative AI Hub, native OpenAI proxy (`chat.completions.create(model_name=...)`) | `gpt-4o-mini` (`SENTINEL_GENAI_HUB_MODEL`) | SAP AI Core service key (`AICORE_CLIENT_ID`, `AICORE_CLIENT_SECRET`, ...) or `~/.aicore/config.json` |
| `anthropic` | Claude via the Anthropic SDK | `claude-opus-5-5` (`SENTINEL_ANTHROPIC_MODEL`) | `ANTHROPIC_API_KEY` or an `ant auth login` profile |
| `openai` | OpenAI SDK | `gpt-4o-mini` (`SENTINEL_OPENAI_MODEL`) | `OPENAI_API_KEY` |

`SENTINEL_LLM_TIMEOUT` (seconds, default 8) bounds every call. Install providers with
`python -m pip install -r agents/sentinel/requirements-llm.txt`.

- **SAP package.** `generative-ai-hub-sdk` is deprecated and archived on PyPI. Its successor
  `sap-ai-sdk-gen` keeps the `gen_ai_hub` import the code uses. The Sentinel only checks that AICORE
  credentials or `~/.aicore/config.json` exist; follow SAP's setup guide (help.sap.com, "generative
  AI hub SDK") for the full list.
- **Requests.**
  - Claude: JSON through structured outputs, at `effort: low`, with server-side refusal fallbacks
    (`fallbacks: "default"`) on models that accept them.
  - OpenAI: a strict JSON schema.
  - Gen AI Hub: JSON mode.
- **Safety.**
  - Article text is untrusted data; every classification must pass the `NewsClassification`
    schema and resolve to a known place.
  - News can only create a candidate.
  - Summaries are written only for confirmed disruptions, from a fixed set of fields: one line, at
    most 25 words, no numbers that aren't in those fields.
  - Any error, timeout, refusal or failed check falls back to keyword rules or the template; a
    test replays every scenario with a broken LLM and still matches `data/expected/`.
- **The expected files assume the LLM is off.**

---

## Integration with teammates (spec section 9)

**Contract.** `models.py` is the single source of truth: `Disruption`, `BusMessage`,
`ContainerRef`, `RawEvent`, the network entities (`Corridor`, `Carrier`, `Facility`, `Market`) and
the interfaces. The orchestrator imports it.

**Contract changes in v2 (India-inside).** All of them are additive; nothing existing was renamed or
removed:

- six new `DisruptionType` values: ROAD_CLOSURE, TRUCK_STRIKE, REEFER_FAILURE, FACILITY_OUTAGE,
  COMPLIANCE_HOLD, MARKET_PRICE_CRASH;
- six new feed sources with their payloads;
- optional `ContainerRef` fields `facility_id`, `carrier_id`, `truck_id`, `market_id`, `cutoff_at`
  and the domestic legs as `status` values (`LEGS`);
- the `NetworkRepository` interface.

**The orchestrator must be told** that `type` can now take the six new values.

**Bus messages.** `{"topic": ..., "disruption": Disruption, "changed_fields": [...]}` on
`disruption.detected`, `disruption.updated` and `disruption.resolved`.

- Candidates are never published.
- `detected` lists every field as changed.
- `resolved` keeps the affected list, so freed containers are known.

**For the Scenario Planner: option families, not decisions.** The Sentinel does not choose or
execute actions. It names which families apply to each type (`options_by_type` in the config;
`SentinelConfig.options_for(type)`): alternate port (JNPT ↔ Mundra), alternate carrier or rail,
hold in cold store (cost per tonne-day in `data/network.json`), split across domestic markets,
processor salvage, re-sequence packing, redo documents, switch destination, reroute vessel. The
planner ranks them against the shelf-life clock, saturation and compliance. A human approves.

**Swap the fakes for real implementations.** Everything is injected:

```python
from datetime import UTC, datetime
from agents.sentinel.llm_client import build_llm_client
from agents.sentinel.sentinel import DisruptionSentinel


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


sentinel = DisruptionSentinel(
    containers=cap_container_repo,  # ContainerRepository: list_active() -> list[ContainerRef]
    bus=team_bus,  # EventBus: publish(topic, message_dict)
    audit=hash_chained_audit,  # AuditSink: log(actor, action, details, rationale)
    clock=SystemClock(),
    llm=build_llm_client().client,  # None unless SENTINEL_LLM is set
    network=master_data_network,  # NetworkRepository: network() -> Network (default: data/network.json)
)
sentinel.ingest(raw_event_dict)  # per feed message (RawEvent fields, received_at included)
sentinel.tick()  # on a schedule (e.g. every minute): expiry, start times, resolution
sentinel.active()  # what to act on now
```

- **ContainerRepository.**
  - `route_legs` are the road segments, corridors, ports and chokepoints still ahead.
  - `status` is the leg the load is on.
  - `days_left` comes from the Shelf-Life Tracker (null → 30, flagged as estimated).
  - Fill `cutoff_at`, `carrier_id`, `truck_id` and `facility_id` for loads still in India; leave
    them null and those rules simply don't fire.
- **AuditSink.** The actor is `disruption_sentinel`. Actions are `create`, `merge`, `escalate`,
  `ignore`, `resolve`, plus `reject`. `details` is JSON-serialisable with `simulated: true`.
- **Feature flag:** `SENTINEL_LLM=off|genai_hub|anthropic|openai`, default `off` for the demo
  laptop.

---

## Layout

```
agents/sentinel/
  models.py        contract: RawEvent, payloads, ContainerRef, network entities, Disruption, BusMessage, interfaces
  settings.py      typed loader for config/sentinel.yaml
  normalise.py     payload validation, place canonicalisation (Gazetteer)
  classify.py      rule tables + news classifier (LLM, keyword fallback)
  merge.py         fingerprints, merge rules, lifecycle (DisruptionTracker)
  scoring.py       confidence, expected delay, vessel cut-offs, severity
  matching.py      affected containers
  summary.py       LLM summary + template fallback
  sentinel.py      DisruptionSentinel: wiring, publish, audit, layer / options / links, event stream
  fakes.py         InMemoryBus, InMemoryAudit, JsonContainerRepo, JsonNetworkRepo, SimClock, ScriptedLLM
  llm_client.py    SAP Gen AI Hub -> Anthropic -> OpenAI -> None
  replay.py        scenario loader, replayer, expected-file checker, CLI
  api.py           FastAPI: REST + SSE + demo entry point
  static/index.html  prototype UI (no build step, works offline)
  config/sentinel.yaml
  data/network.json              corridors, carriers, facilities, markets (illustrative)
  data/containers.json           global book (v1 scenarios)
  data/containers_india.json     India-inside book (v2 scenarios)
  data/scenarios/*.json  data/expected/*.json
  tests/           test_classify, test_merge, test_scoring, test_matching, test_replay,
                   test_contract, test_api, test_llm, test_domestic
```

## Before handing over (spec checklist)

- [x] All tests green (`make sentinel-test`) and lint clean
- [ ] `make sentinel-demo` with Wi-Fi off, on the demo laptop
- [ ] `models.py` shared with the team as the contract, including the v2 additions
- [ ] India-inside numbers (transit times, cut-offs, sailing gap, cold-store costs, shelf-loss rate)
      checked with a real exporter
- [ ] 60-second backup clip of the India-inside demo recorded

All weights, defaults and reliability scores are starting assumptions for the prototype. Tune
them with the team, and say so if judges ask.
