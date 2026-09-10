# Legacy BTC LogReg: final report (development; new period pending)

Reference: BTC_leaning v11 (registry id 827), PINNED per user (PAPER basis).
`reference_model.json` / `CANDIDATES.json` / `stage4/` / `stage56/` hold the
evidence. Every number below links to a table, artifact or test.

## 1. Did the old BTC model's positive result reproduce?
- Accounting YES: trade_history rows for (BTC_leaning, v11) sum to
  realized_pnl +15.916 over 1190 PAPER rows (1001 SUCCESS) — recomputed
  row-by-row (`stage4/legacy_trades_v11.csv.gz`). The "1001 trades" figure
  is the SUCCESS subset.
- As model skill NO: on common rows (785) and operation rows (3080), all
  14 group models produce BIT-IDENTICAL policy entries and PnL under common
  gates (operation +17.74 / common −3.16 for every model). Forecast gaps are
  <=0.006 Brier. Nothing in the result requires v11's coefficients.
- Saved-prediction replay: 60/60 funnel BUYs recomputed, median |diff|
  0.0039 (input staleness, not pipeline mismatch) — the pipeline is faithful.

## 2. Model vs policy vs period vs errors
- Model ≈ 0 (above).
- Policy decisive for selectivity: the binding gates are price (ask<=0.40),
  spread/mid<=0.08, time window, MRF vetoes. The p_flip gate (observed
  global 0.2 + 0.05 band -> 0.25) is NON-BINDING: all models output ~0.3 and
  fire always. Selectivity lives outside the model.
- Period dominates: identical models+policy give +17.74 (08-20..09-09) vs
  −3.16 (08-30..09-09). The old PAPER window coincides with the good half.
- Concentration: top-3 PAPER wins = +9.1 of +15.9 (314W/687L). Removing the
  largest win is sensitivity, not refutation — recorded, not used to dismiss.
- Errors: none found that inflate the result. Leakage audit: bare-LogReg
  blob (no scaler to leak), GROUPED_WALK_FORWARD + OOT T1-T3, unknowns marked
  (train code version, Platt-on-what, activation bypass of a REJECTED gate).
- Split signal: same v11 lost LIVE (−11.73/43) — execution/period, mechanism
  open (not attributed without fill-role analysis).

## 3. Properties worth keeping
- Minimal 3-feature recipe (mid_price, spread, time_left_min), C=0.1, flip
  target with parity exclusion, grouped walk-forward + OOT validation.
- Documented gate structure (0.2 global + band) as OBSERVED policy, not tuned.
- No rarity value (fires always); no shipped calibrator (registry ECE 0.0038,
  raw kept + monitored). Ablation: dropping any one feature moves Brier only
  in the 4th decimal — no removal justified; logit shares mid~60%/tlm~35%.

## 4. What transfers to ETH/SOL/XRP/DOGE
- FROZEN BTC weights: NO — worst Brier on all four assets.
- OLD_RECIPE_LOCAL (same recipe, per-asset fit on pre-08-30 data): consistent
  gains — BTC .165->.148, ETH .152->.125, SOL .169->.155, XRP .174->.165,
  DOGE .183->.180 (`stage56/legacy_transfer.json`).
- BTC_OFFSET_LOCAL (intercept-only): ~= FULL everywhere — base-rate
  adaptation dominates; full refit adds ~nothing. Recipe transfers; a cheap
  offset suffices. Portability holds: all three features are dimensionless
  (mid/spread) or identical-unit (minutes); no dollar-price leakage.
- Status: development-only. New-period confirmation pending (window
  09-10T13:00Z..09-17T13:00Z server UTC).

## 5. Is there a PAPER candidate?
- BTC model challenger: NO (NO_IMPROVEMENT_FOUND — nothing justified).
- Transfer rule OLD_RECIPE_LOCAL: FIXED, not yet evaluated on new data.
  PAPER connection is a separate change after the new-period eval.
  No auto-retrain wiring (all retrains candidate-only, distinctly named).

## Verdict
`TRANSFER_RECIPE_ONLY` (primary) + `KEEP_OLD_BTC` (BTC control stays).
Scenario PnL is never presented as actual. Clock note: local machine runs
+7h vs server UTC; all windows are server UTC; commit times carry the skew.
