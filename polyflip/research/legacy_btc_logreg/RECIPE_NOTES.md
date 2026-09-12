# Recipe table: v11 vs successor/peers/current (from artifacts + registry)

## v11 (id 827, 2026-08-16) — recorded, not inferred
- Data: training window NULL in registry (unknown — recorded as UNKNOWN).
- Target: `POLYMARKET_FLIP_VS_FINAL_OUTCOME`, flip vs favourite, parity excluded.
- Features: [mid_price, spread, time_left_min] (registry + artifact agree).
- Weights: class_weight=null, sample_weight_mode=uniform, tau=0.0.
- Regularization: C=0.1, lbfgs (penalty flag reads "deprecated" = sklearn
  version drift between train-time and now; effective L2 assumed, flagged).
- Calibration: method=PLATT in params, BUT the shipped blob is a BARE
  LogisticRegression (no calibrator object). Registry ECE=0.0038. Whether
  Platt was applied at train and dropped, or recorded-but-unused, is UNKNOWN.
- Validation: GROUPED_WALK_FORWARD + OOT windows T1/T2/T3 (time-grouped,
  market counts ~1278 each) + canonical recompute (logreg-polymarket-v1:
  fee 0.02, min_edge 0.03, outsider_max 0.45 -> +16.34/294).
- Gate: recipe's own gate REJECTED it (`deployable: false`, COMBINED PnL<=0),
  yet it was activated (is_active=true, activated_at NULL) and traded PAPER
  +15.92. Activation bypassed the gate — mechanism unknown, recorded.
- Metadata: model_metadata NULL.

## Successor v12 (id 828, same minute) and peers
- Same 3 features (registry strings identical); blobs 854 B like v11.
- Coefficients/ECE differ slightly (v12 ECE 0.0024). Variant triplets
  (e.g. ETH 14/15/16 same minute) look like threshold/seed variants —
  exact variant axis UNKNOWN (training_params diff pending).

## Current v22 (id 1043, 2026-08-30)
- 26 features (list in registry), blob 2865 B = CalibratedClassifierCV
  (sigmoid) over Pipeline. Full pipeline inspection pending (step 24).

## Leakage audit (best effort; unknowns marked UNKNOWN)
- No scaler/imputer in v11 blob -> no scaler-fit leakage possible for v11.
- No calibrator shipped -> no calibration-on-train leakage IN THE ARTIFACT
  (but Platt-on-what remains unknown).
- Market-in-train-and-validation: GROUPED_WALK_FORWARD suggests market-aware
  splits, but the exact grouping key is UNKNOWN (code version at train time
  unknown; current trainer.py must NOT be assumed to be the old recipe).
- Snapshot multiplicity (many rows/market), lag causality, weight alignment:
  open items for the retrain comparison (step 26 controls them by
  construction: one row per market, causal features, fixed target).
- If the old success depended on leakage, it is not kept as a virtue
  (minimal reproducible example required per defect found).
