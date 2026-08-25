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
