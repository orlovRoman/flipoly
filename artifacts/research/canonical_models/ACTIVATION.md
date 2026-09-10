# Activation package: minimal collection for the frozen CT-final protocol

Target: server repo `/home/orlovrp/flipoly`, branch `research/ct-execution-comparison`
@ `2b531a2` (verified deployed state; CT files present; loop running).
Principle: NO decision-logic changes (spec/classifier/window/thresholds frozen).
Only record what the loop already computes.

## Gap (why regime_match is 0.64–0.67, not 1.0)
`decision_details` in `ct_decision_reservations` carries spec, timing, features
and quote *times* — but NOT the input series/IDs, NOT book prices, NOT the code
sha. `up_snapshot_id`/`down_snapshot_id` are null. Exact CT reproduction from
storage is impossible today. Live history also mixes snapshot mids
(`mid_price`, `poly_*` NULL) with depth fallback, while reconstruction used
depth-only mids.

## Minimal diff (server branch, on top of 2b531a2)
1. `polyflip/trading/decision_runners.py`, `decide_ct_outsider_mode`, `details`
   dict (~line 1872, shared by SKIP reservation and BUY TradeDecision): add
   `input_bundle` with
   - `up_history` / `down_history`: the EXACT series passed to evaluation
     (list of {t, price, source, row_id or null}), plus `history_sources`;
   - `quotes`: full SideQuote contents for BOTH legs
     (bid/ask/mid, event_at, received_at, snapshot/depth row ids);
   - `code_version`: git sha of the running tree (captured once at startup);
   - `execution`: effective settings {role: post-only-maker, limit, budget,
     fee config reference}.
2. Paper-trade recording: populate `trade_history.trade_role` (MAKER/TAKER;
   currently NULL). Status/error already recorded — keep.
3. Nothing else. No spec, classifier, window, cadence, or threshold changes.

## Execution-model note (frozen comparison vs live loop)
- Live loop: post-only MAKER at limit. Evidence 2026-09-09..09-10: 8 BUYs →
  7 FAILED (6 POST_ONLY_REJECTED crossing book, 2 edge-policy unavailable),
  1 SUCCESS settled. A registered BUY is NOT a fill.
- Frozen scenario comparison: taker ladder-walk on the SAME observed ladder,
  identically for control and CT (fully observable, fair). Live PAPER fills
  are reconciled as a separate stream, never merged into policy PnL.

## Freshness by construction
Bundle saved AT the decision carries ages ~0.1 s, satisfying the frozen
15 s gate on both legs. Periodic snapshots (~45 s cadence) remain what they
are: insufficient alone (30/48 STALE on history), kept for history windows.

## Readiness procedure (operator, after deploy)
1. Restart the PAPER loop on the patched tree; confirm new reservations carry
   `input_bundle` + `code_version`.
2. Run `scripts/research/canonical_models/check_record_cycle.py` against fresh
   decisions (extend with a bundle-presence check at activation time).
3. On PASS: record `final_start` UTC = next 15m-contract start boundary;
   `final_end = final_start + 14x24h`. Intermediate review changes neither
   rules nor window. Heavy calculations stay local.
