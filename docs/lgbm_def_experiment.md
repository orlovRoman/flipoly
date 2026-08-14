# LightGBM D/E/F experiment

This experiment compares two assets first: `XRPUSDT_high_vol` and `DOGEUSDT_high_vol`.

## Variants

- **D** — audit-derived stable features only.
- **E** — D plus canonical Polymarket strike features (`strike_gap_pct`, `log_moneyness`).
- **F** — E plus decision-time Polymarket context: `pm_momentum_5m`, `pm_volume_5m`, `pm_spread_pct`, and `pm_quote_pressure`.

The current database does not persist order-book depth, so F uses only persisted top-of-book/context fields. It does not fabricate a depth imbalance. All context rows are joined at or before `market_start`.

## Selection

When `CRYPTO_LGBM_HYPERPARAM_SEARCH_TRIALS` is greater than one, controlled trials are ranked by median OOT Polymarket PnL across the three chronological OOT windows, with a drawdown penalty. Candidates with fewer than 50 OOT trades receive a penalty and remain diagnostic-only. AUC, ECE and Brier remain diagnostic metrics.

The final LightGBM model first uses a chronological tail only to determine `best_iteration`; it is then refit on the complete fitting portion before calibration. Row bagging is enabled with `subsample_freq=1`.

## Safe rollout

Train D, E and F with activation disabled. Compare each saved candidate in `OUTSIDER_ONLY`, `FAVORITE_ONLY` and `COMBINED`. Activate only a winner that has at least 50 OOT trades, positive median PnL, acceptable drawdown and consistent sign across all three OOT windows. F candidates should remain inactive if the live context coverage is missing or incomplete.
