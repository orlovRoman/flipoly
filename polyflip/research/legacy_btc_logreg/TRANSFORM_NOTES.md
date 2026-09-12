# Transform chain: raw LogReg -> trade (verified in code 7b7823ed)

1. `p_flip_raw` = pinned model `predict_proba[:,1]` on
   `[mid_price, spread, time_left_min]` (target=flip).
2. `p_flip_effective = apply_ece_correction(p, ece)` (position_sizing.py):
   `0.5 + (p-0.5) * max(0, 1 - ece/0.10)`. v11 registry ECE 0.0038 ->
   shrink 0.962 (near-identity; recorded `p_logreg_win`/`p_candidate_win`
   match raw to 4dp on the 60-row replay sample).
3. Side/veto: `flip_threshold` + `abstain_band` (|p-thr| < band -> abstain),
   `mrf_extreme_veto`, NO/YES roles (combined_voting.py).
4. `logreg_flip_to_yes_probability(p_flip, fresh_yes_price)`
   (weighted_policy.py:821): YES-favourite defined as market >= 0.50 ->
   `1-p_flip`, else `p_flip`. NOTE: runtime maps parity 0.5 to the YES side
   while training EXCLUDED parity rows. Research keeps parity -> SKIP and
   records the difference.
5. Price/edge gates (`OUTSIDER_MAX_PRICE`, `MIN_EDGE`/net-EV) -> sizing
   (position_sizing.py; fee renderer `apply_polymarket_fee` = gross*(1-0.002)
   is backtest-only and is NOT the confirmed venue fee).

## Role of v11 in decisions (server logs, step 9)
- `trade_history` entry_model BTC_leaning v11: 1233 rows (PAPER 1190 @
  +15.916; LIVE 43 @ -11.725; SUCCESS 1014 / FAILED 208 / SKIPPED 11),
  period 2026-08-20..09-09. Same-model LIVE loss is a first
  model-vs-policy/period split signal (step 23 candidate).
- `decision_funnel_log` entry v11: 10,708 decisions (1203 BUY: 652 NO +
  551 YES; 9505 SKIP), no `fallback_reason` rows -> v11 evaluated each time
  (chose or vetoed), never silently bypassed. UI-display-only attribution
  excluded by funnel presence (step 9 self-check).

## Saved-prediction replay (step 10)
60 funnel BUYs recomputed locally (pinned v11 + causal snapshot features):
60/60 rows matched, median |diff| 0.0039, 58 within 1e-2, 11 within 1e-3,
max 0.011. Residual = input staleness (nearest snapshot <= decision, ~=45 s
cadence; end_time_est-based time_left), NOT pipeline mismatch. Verdict:
pipeline reproduced; inputs documented with quantified staleness.
