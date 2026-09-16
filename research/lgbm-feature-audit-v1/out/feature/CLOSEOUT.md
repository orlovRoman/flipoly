# Feature Audit CLOSEOUT — lgbm-feature-audit-v1

**verdict: STOP_NO_SUPPORTED_FEATURES**
Gate 21 (feature audit): `STOP_NO_SUPPORTED_FEATURES` — 0/79 features clear the supported grid.
Gate 22 (closeout): `STOP_NO_SUPPORTED_FEATURES`.
Binding spec SHA: `ca4dd2fdb7ea6b65e084cb4c59941bcb65741519b385b1fc36e4c5a971147295` (lgbm_feature_audit_v1.yaml v1.0.0); direction-spec `c36f0de5…` is deliberately NOT used for binding.

## Setup
- Cohort: shared_ds 1,497,818 decisions; folds trainpool,F1..F6 + gap (leakage buffer). Labels exist only from F2 (`contract_target`, fallback `flip_native`); eval cohort = F2–F6.
- Feature matrix: 79 features, as-of via searchsorted, snapshot features ~51% present (lag 29s); fold F1 predates snapshot coverage (~0% there).
- 37 active models (17 lgbm + 20 logreg) from frozen blobs. 8 eval lgbm with deployment rows (ids 804,805,812,815,816,817,1019,1024) = 22 fold-units, 34,142 rows.
- Eval folds for ablation/policy: F3–F6 (F2 has no train labels for the model chain).

## F2 — data quality
- **5 features are 100% missing in every fold**: funding_rate, funding_rate_ma3, funding_extreme, strike_gap_pct, log_moneyness → `DATA_QUALITY_BLOCKED`.
- 26 snapshot-derived features hit hard `BLOCK`/`MISSINGNESS_DRIFT` verdicts in f2 → **31 features total are `DATA_QUALITY_BLOCKED`** (artifact of F1/trainpool having ~0 snapshot coverage; honest result).

## F2b — timestamp/leakage provenance (SAMPLED scope, added per review)
- `f2b_leakage.py`: **deterministic 60,000-row sample** (12k/fold F2–F6) of the 1,497,818-row matrix (4.0% coverage). 220 candle rows (44 BASE features × 5 assets), 26 snapshot features, 4 time features, 5 structurally always-NaN.
- Sampled result: `any_future_timestamp_row_in_sample = false`, `leakage_risk_block_in_sample = false`; all real features `value_match ≥ 0.999` and `origin_le_decision = 1.0`. The 5 flagged rows are the structurally-always-NaN features.
- **Honesty scoping**: the check is a 60k-row sample, NOT a full-matrix scan. Full-matrix timestamp provenance is asserted *by construction* in `f1_build_matrix` (as-of searchsorted + 29s lag), and the artifact/claims are labelled `SAMPLED_*` accordingly (denominator `total_rows` is recorded; nothing is quoted as full-matrix). Artifacts: `SAMPLED_LEAKAGE_CHECK.csv`, `SAMPLED_LEAKAGE_CHECK_SUMMARY.json`.

## F3 — cross-feature correlations (added per review)
- Pearson + Spearman + Jaccard-missing over train_part_each_fold; 15,405 pairs (3,081 pairs × 5 folds). **Pearson/Spearman are computed on each pair's COMMON mask** (means/std on the pair's shared observed rows only; Spearman ranks recomputed on the shared rows) per review. 20 pairs near-dup in ≥1 fold; **19 stable pairs ≥4 folds** → 21 distinct member features, 13 of which are already data-quality blocked; the 8 non-blocked members are all `REDUNDANT` (cvd_1/day_of_week/dow/hour_of_day/hour_utc/ret_1/signed_body_pct/taker_buy_ratio). Artifacts: `FEATURE_CORRELATIONS*.parquet`, `FEATURE_CORRELATIONS_AGGREGATE.csv`.

## F4 — model importance
- SHAP headline: range_1 (0.0247, sign +), range_avg_24, dist_to_low_24 (sign-stable 1.0), ema_ratio_9_21, ret_1. Split/gain leaders: ret_1, ret_3, range_1, dist_to_low_24, vol_z_1.
- Permutation (fold-unit bootstrap, strata day/asset/regime): **only cvd_6 and vol_6 have feature-level CI excluding 0** — and neither clears the full supported grid. Everything else is statistically indistinguishable from 0 at unit level.

## F5 — univariate OOF (L2-LogReg, train_k_to_validation_k, Holm α=0.05)
- Top AUC: mid_price 0.8107, pm_best_ask/bid/quote_pressure ≈0.807, price_distance_from_max 0.74.
- **INCONCLUSIVE**: min p_holm = 0.1741 at 200-permutation resolution; no feature survives Holm (0/79). Resolution limit documented in `F5_VERDICT.json` and in gate `f5_resolution`.

## F6 — group ablation
- 20 variants (ALL / DROP / ONLY × 9 catalog groups / MINIMAL_CONTROL=[ret_1, mid_price, spread, time_left_min]) on F3–F6.
- **Day-block paired CI (the canonical main_ci)**: all 9 groups contribute positively when removed; **cross_asset_breadth (ΔLL 0.01338, CI [0.00070, 0.03362]) and volume_taker_flow (0.00641, CI [0.00009, 0.01579]) are robustly helping**. regime has the largest point estimate but CI crosses 0.
- MINIMAL_VS_ALL day-block ΔLL = +0.00912 (CI crosses 0, 2/4 folds positive): the 79-feature set is *not* reliably better than the 4-feature control at day-block granularity.
- EVERYTHING feature-level fails: the SINGLE-feature contribution is sub-resolution; only group-level removal hurts.

## F7 — policy impact (canonical economics, rerun per review)
- **Canonical single-side $1-budget economics** (replaces the buggy select_side): `exec_cost(ask) = ask + 0.07·ask·(1−ask) + 0.005·ask`; one side chosen by max positive edge `e_yes = p − cost_yes`, `e_no = (1−p) − cost_no`; win net `q−1`, loss `−1`, skip `0`.
- All group `DROP_GROUP` deltas on canonical net are now negative across folds (removal never helps the full model); **removing sequence / time / polymarket_state robustly *hurts* economics** (day-block net CI entirely above 0). Predictions remain uncalibrated; policy results are diagnostic, not actionable.
- `GROUP_TRADING_DAYBLOCK_CI.csv` holds the per-day `d_net = net_ALL − net_DROP` CI.

## Verdicts
- SUPPORTED: 0/79. DATA_QUALITY_BLOCKED: 31. REDUNDANT: 8. UNSTABLE: 5 (cvd_6, dist_to_high_24/96, dist_to_low_24/96). NO_POLICY_IMPACT: 20. UNSUPPORTED: 15.
- The full supported grid (`permutation_hurts_validation` AND `drop_group_hurts_oof` AND `effect_in_ge_folds>=4` AND `improvement_ci_not_crossing_zero` AND `stable_effect_direction` AND `holds_after_correlations` AND `no_leakage_no_missingness_proxy` AND `paired_trading_economics_impact`) is not cleared by any single feature. `no_leakage_no_missingness_proxy` is all-Y: missingness-proxy is now computed on the **real FEATURE_MATRIX** (395 fold-feature rows = 79 features × folds F2–F6, point-biserial of isna mask vs contract_target; 0 features have max|r| > 0.2 → `MISSINGNESS_PROXY.csv`), and the sampled leakage check is clean under its honest sampled label.
- Artifacts: `FEATURE_VERDICTS.json`, `FEATURE_VERDICTS_EVIDENCE.csv`, `GATE_FEATURE.json` (status COMPLETED, verdict STOP_NO_SUPPORTED_FEATURES), `MISSINGNESS_PROXY.csv`.

## Proxy threshold — late operationalization

- Threshold `|r_missing_target| > 0.2` for the missingness-proxy is a **late
  operationalization**: it was NOT numerically fixed in the frozen spec
  (`lgbm_feature_audit_v1.yaml` v1.0.0). It is not called preregistered or frozen.
- Maximum observed |r_missing_target| across all 395 fold-feature rows ≈ **0.058**,
  far below the 0.2 cutoff.
- The STOP verdict does **not** depend on this condition: the nearest supported-grid
  candidates fail at least two other gate conditions (e.g. robust permutation /
  improvement CI / economics), so removing the proxy condition would not change the
  outcome.

## Non-blocking limitation — leakage coverage
- Leakage was checked row-by-row on only 4% of the matrix (60,000 of 1,497,818 rows).
  Claims are labelled SAMPLED everywhere; full-matrix causality is asserted by
  as-of construction (searchsorted + 29s lag), not by a full scan. These phrasings
  must not later be shortened to "no leakage in the whole matrix".

## Independent verification
- `verify_final.py` (rewritten as an independent verifier): **71 checks PASS** — matrix shape, always-missing set, day-block LL CI recomputed from `_abl_preds` (10 groups), canonical-net trading day-block CI recomputed with an *independent* economics implementation incl. BUY_NO-regression guard + row-level agreement with `canonical_side_net`, spec-SHA binding (= ca4dd2fd…, ≠ c36f0de5…), no constant gate flags, SAMPLED leakage checks (checked_rows == declared 60,000 denominator, coverage_fraction == checked/total, per-fold 12k), real-matrix missingness-proxy checks (all 79 features, folds F2–F6, 395 rows, finite r, gate discloses 395), F3 near-dup↔REDUNDANT alignment (≥19 stable pairs), robust perm set {cvd_6, vol_6}, F5 Holm empty + INCONCLUSIVE, verdict counts + status-priority consistency, gate-count agreement, and no-required-gate-input-empty.

## Recommendation
- Do NOT promote any of the 79 features as individually-supported.
- Group-level signals exist (cross_asset_breadth, volume_taker_flow for logloss; sequence/time/polymarket_state for trading net) as *groups*, not features. For any future pipeline: drop the 5 always-missing features, treat snapshot-cohort features as unavailable pre-08-10, and rely on control-vs-ALL comparisons rather than single-feature flagging.
- Frozen spec + all artifacts retained under `research/lgbm-feature-audit-v1`.