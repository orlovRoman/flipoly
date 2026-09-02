# Weighted trading policy

## Purpose

The weighted policy makes the market quote the primary prior and turns LogReg,
LightGBM, and the optional market-regime evidence into calibrated inputs. It is
implemented as a pure function so the same arithmetic can be used by live
decisions, PAPER execution, shadow analysis, and future backtests.

All probabilities use the YES axis. The market prior is normalized from both
executable asks to remove the book overround:

```text
p_market_yes = yes_ask / (yes_ask + no_ask)
p_logreg_yes = P(flip) converted to P(YES wins) using the current favorite
p_lgbm_yes   = LightGBM P(UP)

logit(p_final_yes) = logit(p_market_yes)
  + beta_lr   * (logit(p_logreg_yes) - logit(p_market_yes))
  + beta_lgbm * (logit(p_lgbm_yes)   - logit(p_market_yes))
  + beta_mrf  * regime_evidence
  + intercept
```

The initial configured weights are `0.90 / 0.05 / 0.05` for market / LogReg /
LightGBM. The market is the prior rather than another independent vote. If a
model input is missing, its residual is zero and its weight is absorbed by the
market; it is never reassigned to another model.

These are rollout defaults, not learned optimal weights. The market quote is a
strong prior and is intentionally given most of the mass until the two model
probabilities have passed the same OOF calibration and economic replay. The
weights are configurable per experiment; they must not be increased because a
model has a high raw AUC or because it produces more trades.

The roles are deliberately different:

- LogReg estimates the conditional probability of a flip given the Polymarket
  phase and market features.
- LightGBM estimates the canonical YES/NO outcome from the underlying market
  features and acts as an independent directional signal when its model is
  valid.
- The market quote is a prior and execution reference, not an independent
  vote. A large model weight should therefore be justified by incremental OOF
  log-loss/Brier improvement over the quote prior, not by agreement with it.

## Expected value and execution costs

The side is selected by the highest cost-aware expected value, not by a hard
model vote:

```text
net_ev_per_share = P(side wins) - best_ask - estimated_cost_per_share
```

For a taker, the default crypto fee estimate is price-dependent:

```text
fee_per_share = fee_rate * (price * (1 - price)) ** fee_exponent
```

The PAPER gateway rounds the aggregate fee to five decimal places, matching
the venue's fee precision.

The live decision path reads the official CLOB V2 market-info endpoint by
condition ID. It uses `fd.r`, `fd.e`, and `fd.to` for the fee curve and
`mos` for the market-reported minimum order size, and records the source.
If metadata is unavailable, the configured fallback is recorded explicitly.
The execution gateway remains authoritative for the actual fee on a fill.

PAPER's `POLYMARKET_PRICE_DEPENDENT` fee model uses the same formula. The
legacy `FLAT_NOTIONAL` model remains available for focused unit tests and
explicit compatibility callers.

## Rollout modes

`TRADING_POLICY_MODE` is deliberately independent from
`LIGHTGBM_DECISION_MODE`:

- `LEGACY`: unchanged hard-vote policy.
- `WEIGHTED_SHADOW`: calculate and persist weighted telemetry, but keep the
  legacy action. This is the default validation stage.
- `WEIGHTED_ACTIVE`: use the weighted side and per-share net EV for the action.
  Legacy predictive gates (`MIN_WIN_PROB`, `FLIP_THRESHOLD`, consensus and
  favorite/outsider edge thresholds) do not participate. Time/price bounds,
  spread guard, funding/risk veto, operator switches, and execution validation
  remain hard safety limits. The rollout stake is fixed by
  `WEIGHTED_FIXED_BET_USDC` (default $1) until sizing is separately validated.

The optional `WEIGHTED_MRF_BETA` applies signed MRF evidence as a bounded
log-odds adjustment. A positive evidence value supports YES/UP; a negative
value supports NO/DOWN. It is zero by default. Evidence is still recorded at
beta zero, while the legacy post-decision MRF veto/stake multiplier is not
applied to `WEIGHTED_ACTIVE`.

## Key telemetry

Each combined decision records the component probabilities, effective weights,
missing components, selected side, YES/NO net EV, selected cost, fee source,
additive log-odds contributions, models_agree, all benchmark-arm summaries,
and policy mode in decision_details / lgbm_meta. This makes it possible to
compare the shadow counterfactual with the executed legacy result before
switching to WEIGHTED_ACTIVE. The offline report also records role-specific
residuals, tuning candidates, lower-bound Kelly fractions and fixed sizing
steps, all tied to a dataset fingerprint.

## Validation checklist

1. Run unit tests for `polyflip/trading/weighted_policy.py` and the combined
   integration tests.
2. Run WEIGHTED_SHADOW over at least 14 days and 1,000 resolved markets,
   then run the read-only weighted_policy_shadow_evidence.py collector.
   Compare all arms' action agreement, net EV, realized PnL, fees, fill rate,
   and calibration by asset/price bucket.
3. Check that missing LightGBM signals are visible in telemetry and do not
   silently receive the LightGBM weight.
4. Only then enable `WEIGHTED_ACTIVE` in a separately reviewed deployment.

For fitting weights, use a time-ordered nested OOF sweep over
`market_weight`, `logreg_weight`, and `lgbm_weight` with a simplex
constraint. Select on net Polymarket PnL after fees/slippage and on calibration
(Brier, log-loss, reliability), with a minimum trade count and a holdout
window. If the best holdout weight for LightGBM is near zero, treat that as
evidence that the model adds no incremental signal in that slice—not as a
reason to lower the trading threshold.
