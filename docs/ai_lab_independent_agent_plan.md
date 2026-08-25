# Independent AI Research Agent (`ai_research_agent`)

## Goal

Run autonomous research loops (hypothesis → training → OOT → decision →
SHADOW/PAPER overlay) inside a dedicated container that:

- talks to the platform **only** through the AI Lab HTTP API;
- uses any OpenCode model selected in the dashboard (dynamic catalog);
- has **no** access to Docker, shell or the database;
- persists every proposal, result reference and decision in the existing
  AI Lab tables so history stays auditable;
- never triggers LIVE execution; PAPER overlays are the maximum write power,
  LIVE approval remains an explicit operator action.

## Current components

| Component | Role today |
| --- | --- |
| `polyflip/ai_lab/llm.py` | LLM provider abstraction. `HypothesisProposal` / `AgentDecision` pydantic contracts, `MockLLMProvider`, `OpenAIResponsesProvider` (Responses **and** Chat Completions transports, `route_opencode_models`). `get_llm_model_catalog()` is currently a static list (`DEFAULT_OPENCODE_MODELS`) — replaced by the dynamic catalog in T02. |
| `polyflip/ai_lab/agent.py` | `AILabAgent.execute_iteration()` — the in-process research iteration (context build, LLM calls, experiment creation, evaluation). Reused by the external runner through the API instead of direct import. |
| `polyflip/ai_lab/agent_runner.py` | Legacy worker: claims `QUEUED/RUNNING/EVALUATING` runs directly from the DB, leases via `AIWorkerLease`, heartbeats, recovers stale jobs. Will be superseded by the external agent after e2e validation. |
| `polyflip/api/ai_lab.py` | Existing safe REST surface: run CRUD, `/runs/{id}/iterate` queueing, step claim, results recording, evaluate / finalize / shadow promotion, run transitions (pause/resume/cancel), overlay list/rollback, `GET /api/ai-lab/llm/models`. All routes are behind `verify_api_key`. |
| `polyflip/templates/optimizer.html`, `polyflip/static/js/optimizer.js` | Dashboard. Today model selects are populated from the static catalog; T04 switches them to the dynamic catalog API. |
| `docker-compose.yml` | Defines `api`, `db`, `scheduler`, `lgbm_training_worker`, legacy `ai_lab_agent`, execution workers. A new `ai_research_agent` service is added in T06. |

## Target architecture

```
+---------------------------+
| dashboard (optimizer)     |
|  dynamic model catalog    |
|  run creation + control   |
+------------+--------------+
             | HTTP
+------------v--------------+
| FastAPI api container     |
|  /api/ai-lab/agent/*      |
|  /api/ai-lab/llm/models   |
|  PAPER overlay resolver   |
+------------+--------------+
             | HTTP (typed JSON, agent token)
+------------v--------------+
| ai_research_agent         |
|  poll -> claim -> context |
|  LLM (OpenCode)           |
|  submit proposal          |
|  wait experiment result   |
|  decide -> complete       |
+---------------------------+
no DB / no Docker / no shell
```

### Run lifecycle (existing statuses reused)

```
QUEUED -> RUNNING -> EVALUATING -> COMPLETED
                  \-> PAUSED / CANCELLED / FAILED
winner -> SHADOW assignment -> (operator approval) -> deployment
PAPER tuning -> AIConfigOverlay (expires automatically)
```

The external agent claims a queued run through the API (T07), reads an
aggregated context snapshot, submits one validated `HypothesisProposal`,
waits for TRAIN/OOT/POLYMARKET_OOT results produced by the existing
executor/scheduler pipeline, then submits a validated `AgentDecision`.
Every artifact is referenced by id (`run_id`, `config_id`, `result_id`,
`assignment_id`, `overlay_id`).

### Security envelope

- Agent container receives only: `AI_LAB_API_BASE_URL`, `AI_LAB_AGENT_TOKEN`,
  OpenCode credentials/provider settings, poll interval.
- New `/api/ai-lab/agent/*` routes authenticate with the dedicated agent
  token (falls back to the global API key while bootstrapping).
- Context payloads are aggregates only (active models summary, recent trade
  statistics, prior experiments, feature sets, quality gates). Raw orders,
  secrets, raw tables and shell commands are never returned.
- LIVE activation is untouched: `approve_and_activate_deployment` remains an
  operator-only endpoint; agents can at most create SHADOW assignments and
  PAPER-scoped `AIConfigOverlay` rows.

## Delivery plan

| Step | Content |
| --- | --- |
| T01 | API contract table for `/api/ai-lab/agent/*` |
| T02 | Dynamic OpenCode model catalog + `ai_llm_model_catalog` table |
| T03 | Model availability probe endpoint |
| T04 | Dashboard dynamic model selection |
| T05 | Immutable LLM selection snapshot on run creation |
| T06 | `services/ai_research_agent` container + compose wiring |
| T07 | Typed agent API (claim/heartbeat/context/proposal/decision/complete) |
| T08 | Autonomous loop implementation in the external runner |
| T09 | Applicable PAPER overlays resolver + decision/trade trace |
| T10 | SHADOW gate explanations surfaced in dashboard |
| T11 | Agent control block on `/optimizer` |
| T12 | Test suite + switch from legacy runner |

Each step lands as its own commit with targeted tests and `git diff --check`.

## T01 — External agent API contract

All endpoints live under `/api/ai-lab/agent/*`, authenticate with
`Authorization: Bearer <AI_LAB_AGENT_TOKEN>` (fallback: global API key while
bootstrapping) and exchange JSON only. The external agent **never** opens a
DB connection, never imports SQLAlchemy models, never executes shell or docker
commands: every write happens inside the API process.

| Endpoint | Purpose |
| --- | --- |
| `POST /api/ai-lab/agent/claim` | Atomically claim one queued research run (idempotent via `AIWorkerLease`). |
| `POST /api/ai-lab/agent/heartbeat` | Prove liveness for the leased run and renew the lease. |
| `GET /api/ai-lab/agent/runs/{id}/context` | Fetch the aggregated, safe context snapshot for the leased run. |
| `POST /api/ai-lab/agent/runs/{id}/proposal` | Submit one validated `HypothesisProposal` (creates config + step). |
| `POST /api/ai-lab/agent/runs/{id}/decision` | Submit one validated `AgentDecision` after OOT results exist. |
| `POST /api/ai-lab/agent/runs/{id}/complete` | Finish the run (`COMPLETED`/`FAILED`) or return it to the queue. |

### State machine

```
QUEUED --claim--> RUNNING --proposal--> RUNNING(step TRAIN/OOT/POLYMARKET_OOT)
RUNNING --results ready--> EVALUATING --decision--> RUNNING | COMPLETED | FAILED
RUNNING --complete{action:"requeue"}--> QUEUED   (lease released)
heartbeat failure / TTL expiry ------------------> lease reclaimable by any agent
```

### Payload sketches

```jsonc
// POST /claim  {}
// -> {"run": null} or:
{"run": {"id": 41, "status": "RUNNING", "objective": "...", "scope": {"asset": "BTC"},
          "autonomy_level": "EXPERIMENT", "budget_experiments": 3, "budget_seconds": 3600,
          "lease_token": "<opaque>", "llm_provider": "opencode",
          "llm_research_model": "...", "llm_summary_model": "..."}}

// POST /heartbeat {"run_id": 41, "lease_token": "<opaque>"}
// -> {"run_id": 41, "leased_until": "2026-08-19T12:00:00Z"}

// GET /runs/41/context ->
{"run": {"id": 41, "iteration": 1, "budget_remaining_steps": 2},
 "active_models": [{"asset": "BTC", "version": 51, "model_type": "logreg",
                     "accuracy": 0.64, "ece": 0.03, "quality_gate_passed": true}],
 "recent_trade_statistics": {"trades_24h": 120, "win_rate": 0.53, "net_pnl_24h": 4.2},
 "prior_experiments": [{"config_id": 77, "median_oot_pnl": -1.2,
                         "verdict": "REJECTED", "reason": "negative pnl"}],
 "available_feature_sets": ["FS_D0", "FS_D1", "FS_D2"],
 "quality_gate": {"min_trades": 30, "max_ece": 0.15,
                   "min_positive_oot_windows": 2}}

// POST /runs/41/proposal  -> HypothesisProposal (validated pydantic schema)
// -> {"config_id": 91, "step_id": 501}

// POST /runs/41/decision   -> AgentDecision (validated pydantic schema)
// -> {"accepted": true, "assignment_id": null, "overlay_id": null}

// POST /runs/41/complete {"action": "COMPLETED"|"FAILED"|"REQUEUE",
//                          "reason": "..."}
// -> {"run_id": 41, "status": "COMPLETED"}
```

Error envelope: `{"detail": "<machine-readable reason>"}` with HTTP 401/404/
409/422. A lost lease yields `409 LEASE_LOST` and the agent must drop the run.
