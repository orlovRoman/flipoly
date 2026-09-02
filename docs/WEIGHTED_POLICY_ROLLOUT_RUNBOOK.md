# Weighted trading policy rollout runbook

This runbook is for the isolated branch codex/weighted-trading-policy.
The production checkout is not used for development. The safe default remains
TRADING_POLICY_MODE=LEGACY and LIVE_TRADING_ENABLED=false.

## 1. Capture the runtime

Run from the repository root:

~~~text
python scripts/weighted_policy_runtime_snapshot.py \
  --output artifacts/weighted_policy/runtime_snapshot.json \
  --assert-live-disabled
~~~

The snapshot contains the commit, complete weighted controls, costs, sizing
parameters and active ModelRegistry model/version metadata. It intentionally
omits database URLs, private keys, model weights and tokens. If the database is
unavailable, active_models_source records that fact instead of inventing
versions.

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
FULL_WEIGHTED, FULL_WEIGHTED_MRF, OUTSIDER_AGREE_ONLY and STACKER. It reports
Brier score, log-loss, ROI, realized net PnL, win rate, cluster-bootstrap PnL
intervals, threshold/price/time/beta-MRF tuning, 2.5%/5%/10% lower-bound Kelly,
and fixed $1/$1.5/$2/$3 sizing steps. Stackers use chronological purged
walk-forward folds, bounded ridge-logistic coefficients, role/agreement
features and hierarchical segment shrinkage.
The benchmark CLI defaults to a one-market-group purge gap. Pass `--purge-gap 0`
only for a documented sensitivity run; it is not the rollout default.

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
POLICY_ID=$(python -c 'import json; print(json.load(open("artifacts/weighted_policy/policy_v1.json"))["artifact_id"])')
python scripts/weighted_policy_shadow_evidence.py \
  --days 30 \
  --policy-id "$POLICY_ID" \
  --repeat-oot-reports 1 \
  --output artifacts/weighted_policy/shadow_evidence.json

python scripts/weighted_policy_activation_check.py \
  --evidence artifacts/weighted_policy/shadow_evidence.json \
  --artifact artifacts/weighted_policy/policy_v1.json
~~~

The evidence collector is read-only. It derives counts, all-arm telemetry,
Brier/PnL/cluster-CI, calibration error and LIVE expected-vs-realized price
drag from the database. Raw arm decisions remain in each benchmark summary,
while `policy_*` fields replay the ACTIVE role, spread, time, price, agreement
and net-EV gates; `shadow_candidate_trades` counts only policy-eligible
candidates and `shadow_raw_candidate_trades` exposes the unfiltered raw count.
Pass the artifact ID so shadow and ACTIVE rows are filtered to one immutable
policy; the activation check rejects missing, mixed, or mismatched policy IDs.
The pre-live command exits with status 2 until the SHADOW/OOT quality conditions
are met; live fills are intentionally not required yet. A positive point
estimate is not enough when the bootstrap lower bound is non-positive.
repeat_oot_reports must be provided for each independently saved OOT report.

For a final pre-live review, add `--require-rollout-quality` and provide
`--stability-ok --sensitivity-ok --sizing-mode FIXED
--sizing-base-bet-usdc 1`. These flags require explicit reviewed evidence for
stability and parameter sensitivity and reject any first ACTIVE sizing other
than a fixed one-dollar bet.

After the fixed $1 ACTIVE period has accumulated at least 300 fills, run the
same check with --require-live-validation. That second gate enforces the T57
fill count, execution-drag limit and calibration-error limit.

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
