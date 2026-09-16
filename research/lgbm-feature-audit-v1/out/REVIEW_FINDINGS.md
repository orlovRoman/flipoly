# REVIEW_FINDINGS — 2026-09-15 rev2 + 2026-09-16 resolution

- direction INVALIDATED_PENDING_RERUN (E4-E6 stub)
- feature ~~INVALIDATED_PENDING_RERUN~~ → **COMPLETED** (independent rerun; see log below)
- shared CONDITIONAL_PASS (parquet 48a239b6, 1497818 rows, 21757 markets)
- E1 repro_status now in e1_summary.json: 3/9/45/16
- opportunity_id to be defined as market_id+policy_id+window, clustering by market/day

## Feature-audit review items (rev2) — resolution 2026-09-16

1. **Wrong binding SHA** (c36f0de5 = direction spec) → all artifacts rebound to
   `ca4dd2fdb7ea6b65e084cb4c59941bcb65741519b385b1fc36e4c5a971147295`
   (lgbm_feature_audit_v1.yaml v1.0.0); `_update_manifest.py` fixed so a re-run
   cannot revert the binding.
2. **canonical-PnL bug** (select_side BC/NC overwrite) → replaced by `canonical_side_net`
   single-side $1-budget economics; F7 rerun; NO-side regression guard in verify_final.
3. **F2 without timestamp/leakage check** → new `f2b_leakage.py`; now
   **SAMPLED scope**: `SAMPLED_LEAKAGE_CHECK.csv + SAMPLED_LEAKAGE_CHECK_SUMMARY.json`
   (60,000 rows / 12k per fold F2..F6 = 4.0% of 1,497,818; denominator + coverage
   recorded; full-matrix provenance credited to as-of construction in f1, NOT to a
   row-by-row scan). any_future_timestamp_row_in_sample=false, leakage_risk_block_in_sample=false.
4. **F3 not reproducible (single matrix-wide row count)** → new `f3_correlations.py`:
   15,405 pairs via train_part_each_fold; **Pearson/Spearman recomputed on each pair's
   COMMON missing-mask** (means/std + ranks on the pair's shared observed rows only);
   20 near-dup pairs ≥1 fold, **19 stable pairs ≥4 folds** → in f8 those members are
   REDUNDANT (8 non-blocked).
5. **verify_final weak (34 hardcoded-ish PASS)** → rewritten independent verifier,
   **71 checks PASS** (independent economics, SHA binding, constant-flag scans,
   SAMPLED leakage with denominator equality, real-matrix missingness-proxy 79×5=395
   + finite r + gate disclosure, correlations↔verdicts, counts, no empty gate inputs).
6. **Group vs MINIMAL_CONTROL significance** → documented sensitivity to aggregation;
   day-block CI is primary; MINIMAL_VS_ALL day-block ΔLL +0.00912 CI crosses 0.
7. **Group significance preliminary** → day-block paired CI + trading net CI are now
   independent artifacts (GROUP_DAYBLOCK_CI.csv, GROUP_TRADING_DAYBLOCK_CI.csv).
8. **Missingness-proxy was not actually computed** (old f8 read `_abl_preds`, which
   have no source features) → new f8 reads `FEATURE_MATRIX.parquet` (all 79 features,
   per labeled fold F2..F6 vs `contract_target`); artifact `MISSINGNESS_PROXY.csv`
   (395 rows); 0 features with max|r| > 0.2 (max observed ≈ 0.058). Verifier checks
   all 79 features + folds F2..F6 + finite r.
9. **Leakage check presented as full-matrix** (only 60k of 1,497,818 rows scanned)
   → renamed to SAMPLED_LEAKAGE_CHECK* with honest scope fields; claims reworded;
   verifier asserts checked_rows == declared 60,000 denominator.

## Late operationalization (accepted, non-blocking)

- Proxy threshold `|r_missing_target| > 0.2` was NOT numerically fixed in the original
  spec; recorded here as a late operationalization. Max observed |r| ≈ 0.058. STOP
  holds without this condition: nearest candidates fail at least two other gate
  conditions.

## Verdict (final, post-rerun)

- GATE_FEATURE.json: `STOP_NO_SUPPORTED_FEATURES`, status COMPLETED, gate_version v1.1.0.
- SUPPORTED 0 / DATA_QUALITY_BLOCKED 31 / REDUNDANT 8 / UNSTABLE 5 /
  NO_POLICY_IMPACT 20 / UNSUPPORTED 15 (n=79).
- F5 is explicitly INCONCLUSIVE (200-perm resolution, min p_holm 0.1741);
  F6 day-block LL robustly helps cross_asset_breadth + volume_taker_flow;
  trading net robustly hurt by removing sequence/time/polymarket_state.
- verify_final.py: PASS, **71 checks**. Git-часть: PENDING (push в research/lgbm-feature-audit-v1; ветка сейчас на f4befc8d).
- Исследовательский расчёт: PASS / COMPLETED. Решение: STOP_NO_SUPPORTED_FEATURES.
  Активация новых моделей/фич по этому аудиту запрещена; работающие production-модели,
  CT, LogReg и текущие торговые режимы не затрагиваются.