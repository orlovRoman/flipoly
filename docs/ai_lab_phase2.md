# AI Lab phase 2: safe experiment lifecycle

The second phase adds a database-backed control surface for experiments. It is
deliberately isolated from model activation, RuntimeSettings and order
submission.

## API

All endpoints require the normal API key.

- `POST /api/ai-lab/permissions` creates an immutable permission profile
  version. The previous version remains in the database, while the new version
  becomes current.
- `GET /api/ai-lab/permissions` lists all permission versions.
- `POST /api/ai-lab/runs` creates a run in `DRAFT`.
- `GET /api/ai-lab/runs` and `GET /api/ai-lab/runs/{id}` expose the run,
  audit steps and stored results. The list endpoint supports
  `status`, `created_by`, `limit` and cursor `before_id`; it returns
  `next_before_id` for the next page.
- `POST /api/ai-lab/runs/{id}/steps` appends a human-readable and structured
  step record.
- `POST /api/ai-lab/runs/{id}/transition` enforces the lifecycle; terminal
  outcomes (`FAILED`, `CANCELLED`, `REJECTED`, `ROLLED_BACK`) are system
  outcomes and do not require an agent permission. It also checks
  the permission action required for each transition. The public API cannot
  transition a run to `ACTIVE`; that state requires a future explicit human
  approval handler.
- `POST /api/ai-lab/runs/{id}/actions/check` checks an action against the
  permission snapshot without executing it.
- `POST /api/ai-lab/configs` stores an immutable, content-hashed experiment
  configuration.
- `POST /api/ai-lab/runs/{id}/approval` creates an explicit human approval
  request for activation or rollback.

## Safety boundary

The service rejects unknown actions and never includes `ACTIVATE_LIVE` or
`CHANGE_LIVE_POLICY` in the autonomous action set. A run without a permission
snapshot cannot authorize any action. No endpoint in this phase changes active
models, live budgets, execution modes, or open positions.

## Example

1. Create a permission profile with `CREATE_EXPERIMENT`,
   `TRAIN_MODEL`, `RUN_OOT_BACKTEST`, `PROMOTE_TO_SHADOW`,
   `REQUEST_ACTIVATION` and `STOP_EXPERIMENT`.
2. Create a run with that profile and an explicit experiment budget; the
   run stores the exact permission version as its immutable snapshot.
3. Append each hypothesis, training attempt and backtest result as a step.
4. Transition the run to `SHADOW` only after the recorded evaluation.
5. Create an approval request; activation remains a separate human-controlled
   phase.
