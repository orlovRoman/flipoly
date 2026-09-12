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

## Deploy record (executed 2026-09-10)
- Server tree verified clean, HEAD == `2b531a2` (patch base). Patch scp'd
  (ASCII staging path; Cyrillic home path breaks Win32 scp), `apply --check`
  clean on server, applied (62 insertions, 1 file), `py_compile` clean.
- Rebuilt images: `flipoly-execution_worker_paper` (manifest
  `sha256:d43907261f3462dbc15bda29f8a29bf7298ca83e00df8984360a0cd77c1c73dd`),
  `flipoly-scheduler` (manifest
  `sha256:0bbe9e61d7c6a77d8804736672daaf641539ee29881620b053cb3a19c48ef7f1`).
- Restarted ONLY `execution_worker_paper` + `scheduler` (~08:47 UTC); all other
  services untouched and Running; both new containers Up/healthy.
- First post-restart decisions (08:55, 09:10 UTC) carry `input_bundle`
  (19/19 and 20/20 obs). `code_version` = "unknown" (no .git inside runtime
  image — expected; logic identity = top-level `spec_hash` + image digests).
- Bundle content verified: full quotes (bid/ask/mid, event/received, ages
  ~0.02–0.07 s), history sources, execution role post-only-maker.
- Reproduction from saved bundle (4402907, DOWN): ER 0.4831 / sc 0.8462 /
  ac −0.4935 / 19 obs — BIT-IDENTICAL to registered `ct_features`;
  regime UNCERTAIN reproduced. (4403224: PRICE_FILTER short-circuit live;
  bundle recomputes UNCERTAIN/VALID deterministically.)
- Readiness: PASS on post-restart decisions (bundle + depth≤15 s +
  snapshot≤60 s). Formal `check_record_cycle.py` runs pre-final_start on
  complete chunks.

## Deploy procedure (reference; PAPER loop only)
Patch: `activation_ct_bundle.patch` (this dir; verified `git apply --check`
clean against `2b531a2` + `py_compile` clean). Additive only, decision logic
untouched, bundle wrapped in try/except.
```sh
cd /home/orlovrp/flipoly
git status --short  # must be clean
git checkout research/ct-execution-comparison
git apply /tmp/activation_ct_bundle.patch
git diff --stat
python3 -m py_compile polyflip/trading/decision_runners.py
docker compose build execution_worker_paper scheduler
docker compose up -d execution_worker_paper scheduler
docker compose ps execution_worker_paper scheduler
```
Verify (read-only): new reservations carry the bundle —
```sql
SET statement_timeout TO '30s';
SELECT decision_at, action,
       (decision_details->'input_bundle') IS NOT NULL AS has_bundle,
       decision_details->'input_bundle'->>'code_version' AS code
  FROM ct_decision_reservations ORDER BY decision_at DESC LIMIT 5;
```

## Readiness procedure (operator, after deploy)
1. Confirm new reservations carry `input_bundle` + `code_version` (query above).
2. Run `scripts/research/canonical_models/check_record_cycle.py` against fresh
   decisions (extend with a bundle-presence check at activation time).
3. On PASS: record `final_start` UTC = next 15m-contract start boundary;
   `final_end = final_start + 14x24h`. Intermediate review changes neither
   rules nor window. Heavy calculations stay local.
