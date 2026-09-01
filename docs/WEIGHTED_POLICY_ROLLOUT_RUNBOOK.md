# Weighted trading policy rollout runbook

This runbook is for the isolated branch codex/weighted-trading-policy-rollout.
The production checkout is not used for development. The safe default remains
TRADING_POLICY_MODE=LEGACY and LIVE_TRADING_ENABLED=false.

## 1. Capture the runtime

Run from the repository root:

~~~text
python scripts/weighted_policy_runtime_snapshot.py \
  --output artifacts/weighted_policy/runtime_snapshot.json \
  --assert-live-disabled
~~~

The snapshot contains the commit, policy mode, policy ID, costs, MRF mode and
execution role. It intentionally omits database URLs, private keys and tokens.

## 2. Capture baseline and observations

Create a read-only 7/14/30 day report:

~~~text
python scripts/weighted_policy_baseline.py \
  --output artifacts/weighted_policy/baseline_7_14_30.json
~~~

Create a repeatable resolved fixture after exporting observations:

~~~text
python scripts/weighted_policy_benchmark.py \
  --days 30 \
  --output artifacts/weighted_policy/benchmark_30d.json \
  --export-observations artifacts/weighted_policy/observations_30d.json

python scripts/weighted_policy_fixture.py \
  --input artifacts/weighted_policy/observations_30d.json \
  --output artifacts/weighted_policy/fixture_30.json \
  --size 30
~~~

The exporter uses the latest quote at or before the decision timestamp and
never uses a future quote for an entry decision. It uses observed filled cost
when it is present in an input row; otherwise it uses the shared fee/slippage
estimate.

## 3. Offline acceptance

The benchmark compares MARKET_ONLY, LEGACY, MARKET_LOGREG, MARKET_LGBM,
FULL_WEIGHTED_MRF, OUTSIDER_AGREE and STACKER. It reports Brier score,
log-loss, realized net PnL, win rate, cluster-bootstrap PnL intervals and
threshold sensitivity. Stackers use chronological purged walk-forward folds,
bounded ridge-logistic coefficients, role/agreement features and hierarchical
segment shrinkage.

Run the required offline suite:

~~~text
pytest -q -m "not live"
~~~

A fold is not an activation proof. The report must be saved with its source
observations and runtime snapshot.

## 4. Policy artifact

Create an immutable artifact only from a reviewed benchmark report:

~~~text
python scripts/weighted_policy_benchmark.py \
  --input artifacts/weighted_policy/fixture_30.json \
  --output artifacts/weighted_policy/fixture_report.json \
  --artifact artifacts/weighted_policy/policy_v1.json \
  --policy-version weighted-policy-v1
~~~

An existing artifact cannot be overwritten with a different hash. The runtime
policy ID should be set to that artifact ID or another reviewed immutable
identifier.

## 5. Shadow and activation gate

Use WEIGHTED_SHADOW for parallel telemetry. It computes the new score and EV
using the same market quote, fee curve, spread and slippage inputs, but does
not replace the LEGACY action.

Before any fixed-bet activation, check all plan evidence:

~~~text
python scripts/weighted_policy_activation_check.py \
  --shadow-days 14 \
  --shadow-resolved-markets 1000 \
  --shadow-candidate-trades 300 \
  --repeat-oot-reports 1 \
  --live-fills 300 \
  --pnl-ci-lower 0.01 \
  --artifact artifacts/weighted_policy/policy_v1.json
~~~

The command exits with status 2 until every minimum is met. A positive point
estimate is not enough when the bootstrap lower bound is non-positive.

## 6. Rollout rules

1. LEGACY is the rollback value.
2. WEIGHTED_SHADOW is the default experiment mode.
3. WEIGHTED_ACTIVE is allowed only with a reviewed immutable artifact and a
   passing activation check.
4. The first active sizing mode is fixed USDC=1.00. Kelly sizing stays offline
   until uncertainty estimates and live-fill evidence are available.
5. Taker and maker are separate execution roles. Their fee/slippage inputs and
   telemetry must not be mixed.
6. Any change to coefficients, costs, thresholds, MRF veto or role requires a
   new policy artifact and a new out-of-time report.
7. LIVE_TRADING_ENABLED must remain false unless the operator explicitly
   authorizes real-money execution.

## 7. Rollback

Set TRADING_POLICY_MODE=LEGACY, restore the previous policy ID, restart the
affected services, and preserve the failed report and runtime snapshot for
audit. Do not rewrite an immutable artifact or delete the evidence.
