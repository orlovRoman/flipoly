# AI Lab Phase 8 — Automatic Run Finalization & Strict Finalization Gate

Phase 8 closes the autonomous experiment loop with a strict, verifiable evaluation gate before granting candidate entry into passive SHADOW observation.

## Strict Finalization Gate Contract

Before any candidate is assigned to SHADOW, `evaluate_finalization_gate` validates the aggregated `POLYMARKET_OOT` results against non-negotiable quantitative criteria:

1. **Independent OOT Windows**:
   - Must contain at least **3 distinct, non-empty OOT windows** (`MIN_WINDOWS = 3`), defined by unique `(oot_window_start, oot_window_end)` pairs.
2. **Total Trade Volume**:
   - Must have at least **50 total trades** across the evaluated OOT windows (`MIN_TOTAL_TRADES = 50`). `run.scope["min_trades"]` can only increase this threshold, never decrease it below the system minimum.
3. **Strictly Positive Median PnL**:
   - The median net PnL across evaluated OOT samples must be strictly positive (`median_net_pnl > 0.0`).
4. **Finite Values**:
   - PnL, drawdown, and trade counts must be valid finite numbers (excluding `NaN`, `Inf`, and `None`).

### Gate Rejection Reasons
When a candidate fails the gate, it receives a descriptive rejection code:
- `NO_RESULTS`: No evaluation results recorded for the run.
- `NO_PNL_SAMPLE`: No successful `POLYMARKET_OOT` evaluation rows exist for the candidate.
- `INSUFFICIENT_TRADES`: Total trade count across OOT windows is below the minimum threshold (< 50).
- `INSUFFICIENT_WINDOWS`: Total unique OOT windows is below the minimum threshold (< 3).
- `NON_POSITIVE_PNL`: Median net PnL across windows is `<= 0.0`.
- `INVALID_RESULT`: Non-finite, NaN, or corrupted metrics detected.

## Finalize an Experiment

After the scheduler/worker has processed all planned steps, call:

```http
POST /api/ai-lab/runs/{run_id}/finalize
Content-Type: application/json
X-API-Key: <api-key>

{
  "auto_shadow": true,
  "asset": "BTCUSDT",
  "regime": "mid_vol"
}
```

The endpoint:
1. Evaluates all persisted experiment results through `evaluate_finalization_gate`.
2. Returns structured recommendation output:
   ```json
   {
     "recommendation_status": "READY_FOR_SHADOW",
     "rejection_reasons": [],
     "window_count": 3,
     "total_trades": 74,
     "median_pnl": 1.24
   }
   ```
3. When `READY_FOR_SHADOW` and `auto_shadow=true`, creates an `AIShadowAssignment` in passive SHADOW.
4. If an assignment already exists for the same scope and candidate, returns it idempotently without error.
5. On rejection, writes the detailed report to `run.summary` and records an immutable audit entry in `ai_step_audit_logs`.

## Safety Boundary

This endpoint cannot activate a model into `ACTIVE`, modify `RuntimeSettings`, change trading filters, or open orders. The promotion to LIVE requires human approval through `AIApprovalRequest` with an immutable `DeploymentRevision`.
