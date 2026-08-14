# AI Lab Phase 8 — automatic run finalization

Phase 8 closes the autonomous experiment loop without granting the laboratory
permission to change live execution.

## Finalize an experiment

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

1. Evaluates the persisted experiment results using the same median
   Polymarket-OOT PnL/trades/drawdown report used by the manual evaluator.
2. Returns `NO_PNL_SAMPLE` or `INSUFFICIENT_DATA` without selecting a model when
   there is no eligible Polymarket sample.
3. When the report is `READY_FOR_SHADOW`, selects its recommended config and
   creates an `AIShadowAssignment` in passive SHADOW.
4. Stores the report and assignment provenance in the run summary so later
   audits can explain what was selected and why.

The candidate artifact ID can be supplied explicitly with
`candidate_artifact_id`. Otherwise the first artifact attached to the
recommended result is used. `baseline_artifact_id` is optional.

Set `auto_shadow` to `false` to produce the report only. This is useful for
reviewing a run without creating an assignment.

## Safety boundary

This endpoint cannot activate a model, modify RuntimeSettings, change trading
filters, or open orders. The next step after SHADOW remains an explicit,
permission-checked human action. A failed or incomplete run is not promoted.
