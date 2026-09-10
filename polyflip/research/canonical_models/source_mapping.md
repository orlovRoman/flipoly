# Source mapping (step 5) — every required field links to a REAL column/API or is marked MISSING

> Rule: field names are never guessed. Fill from the actual schema; keep MISSING rows visible.

| Required field | Real table/column or API | Status | Notes |
|---|---|---|---|
| market_id | `live_markets.market_id` (PK) + `market_snapshots.market_id`; Gamma `markets.id` at collection | MAPPED | 1 row/market enforced in `market_outcome_dataset` (`assert market_id.is_unique`) |
| asset | `live_markets.asset` / `market_snapshots.asset`; Binance symbol via `constants.ASSET_TO_BINANCE_SYMBOL`; supported set `COMBINED_MODE_SUPPORTED_ASSETS` = BTC/ETH/DOGE/XRP/SOL | MAPPED | collector maps Gamma title/tags → asset (`collector/client.py`) |
| start_at / end_at | `live_markets.end_time_est` (NOT NULL) + `end_date` (nullable); start DERIVED as `end_time_est − 15m` (`market_start` in dataset) | MAPPED_DERIVED | no explicit `start_at` column; 15m duration confirmed by `trading/feature_builder.py: market_duration_min=15.0` |
| up_token_id / down_token_id | `live_markets.yes_token_id` / `no_token_id` (NOT NULL); legacy `clob_token_id` / `clob_token_down_id` (nullable); Gamma `clobTokenIds[0..1]` at collection | MAPPED | YES=UP leg, NO=DOWN leg |
| strike_value | DB: `market_snapshots.strike_value` + `live_markets.strike_value` exist but contain 0 rows (verified 2026-09-10); Gamma market object has NO strike/openingPrice/priceToBeat field | MISSING_IN_DB | retrospective path: Chainlink TWAP-60s stream value at window start (see resolution rule); availability AT DECISION TIME still to prove (step 8). Binance fallback FORBIDDEN in canonical sample |
| strike_source | funnel `decision_funnel_log.strike_source` (nullable); otherwise NEW registry field | PARTIAL | must normalise to canonical_confirmed \| retrospective \| binance_proxy \| unknown |
| strike_available_at | MISSING (no column) | MISSING | must prove availability before decision (step 8); retrospective needs separate proof |
| resolution_rule | 15m crypto family (`cryptoMarketConfig.id=btc-15m-twap-60`, twapLookbackSeconds=60): UP iff Chainlink TWAP over window ≥ price at window start, else DOWN (equality → UP). Source: Gamma `description` + `resolutionSource` (verified on live market 4402621) | MAPPED_15M_CRYPTO | other families (daily etc.) need own rule rows; equality rule pinned: price==strike → UP |
| resolution_source | `live_markets.resolution_source` + `settlement_price_source` (515 recent markets → Chainlink `...-usdtwap-60s-streams` URLs); `resolution_status/resolved_at/resolution_checked_at`; Gamma `resolutionSource` | MAPPED | Chainlink TWAP-60s streams per asset |
| actual_outcome | `market_snapshots.final_outcome`: YES 4.18M / NO 4.20M / PENDING 7.5k rows (2026-06-25→now); `live_markets.final_outcome` mostly NULL (mirror staleness — use snapshot-derived labels); resolver `extract_final_outcome` | MAPPED | real resolution only; own Binance-vs-strike NEVER substitutes; INVALID/PENDING excluded from main sample |
| outcome_available_at | MISSING as explicit column; proxy = `live_markets.resolved_at` / `resolution_checked_at` / first YES/NO `market_snapshots.recorded_at` | PARTIAL | needed for split causality; proxy must be validated |
| quotes both sides + depth | UP leg top-of-book: `market_snapshots.mid_price/best_bid/best_ask/spread` (CLOB `/book` on yes_token_id; `poly_*` columns 100% NULL = legacy, do not use). DOWN leg + ladders: `orderbook_depth_snapshots` (149k rows; 106k/day, YES/NO balanced, `is_truncated=false`): full `bids/asks` jsonb per token + `event_at/received_at/best_*_price+size/depth_usdc_*`, FK → snapshots.id, idx (market_id,token_id,received_at). Timing: `market_timestamp/received_timestamp/recorded_at`; `time_left_seconds` 100% NULL → derive `end_time_est − recorded_at` | MAPPED (join required) | DOWN leg from depth (cf. 2b531a2 + `decide_ct_outsider_mode`); NO never via 1−YES |
| underlying price (contract-consistent, at decision) | `underlying_observations` (1.1M rows): (instrument, source, event_at, price, received_at) — but source=BINANCE only (all 5 assets, ~130-170k/day); canonical (Chainlink TWAP stream) series NOT collected | MISSING_CANONICAL | Binance kept as separate proxy/additional column only; Chainlink retrospective path shared with strike (step 8) |
| fee (applicable commission) | CONFIRMED formula (crypto v2, taker-only): fee = C × feeRate × p × (1−p), feeRate=0.07 (docs.polymarket.com + polymarket.com/fees; verified against official table: 100 sh @0.50 → $1.75). Maker fee = 0. Per-market `feeType/feeSchedule/takerBaseFee/makerBaseFee` on Gamma object (verified live). Repo flat 0.002 (`settings_registry`, `backtester`) is STALE for crypto — do not use as confirmed. | FORMULA_CONFIRMED_HISTORY_TBD | per-market scheme+period to confirm from Gamma objects (incl. historical); maker/taker ROLE decisive (maker=$0); takerBaseFee=1000 semantics TO_VERIFY (minimum?); study: fee UNKNOWN until per-market confirmation, scenarios 0/0.001/0.002 + breakeven; engine = vendored `orderbook_execution.py` |
| CT filter realization | `polyflip/trading/ct_policy.py` (BTC_CT_T5_V1, commit 2b531a2, vendored byte-identical) + `polyflip/research/regime_features.py::classify_local_regime`; wrapper `compute_token_ct_regime`; PAPER hook `decision_runners.decide_ct_outsider_mode` | MAPPED | spec: BTC 15m, window 210–300 s, ask [0.01,0.40], history 900 s, min_obs 3 (not classifier default 4), age ≤15 s, REVERSION only, 1 decision/market; other assets out of scope |
| trade-economics engine | `polyflip/research/orderbook_execution.py` (`simulate_orderbook_execution` + `calculate_trade_payout_and_pnl`, 2b531a2 vendored) | MAPPED | budget = up to $1 purchase cost, fee on top; unspent never loss; truncated depth → UNKNOWN; study wraps it (no second calculator) |
| time availability (event/received) | snapshots `market_timestamp` (event) / `received_timestamp` (received) / `recorded_at`; candles `open_time/close_time/is_closed` | MAPPED | decision uses only snapshots with event ≤ decision; unknown time never → zero age |

## Strike credibility classes (step 8)
- `canonical_confirmed`: confirmed source, provably available at decision.
- `retrospective`: canonically restored, availability separately proven.
- `binance_proxy`: NEVER in the main canonical sample.
- `unknown`: excluded from main sample, counted in coverage.

## Resolution families (step 9)
- 15m crypto (`btc-15m-twap-60` et al.): Chainlink TWAP-60s over window vs window-start price; equality → UP. PINNED.
- Other families: TODO (own rule rows required before use).

## Coverage snapshot 2026-09-10 (server, read-only audit)
- `market_snapshots` ≈ 6.9M rows, 2026-06-25 → now; outcomes YES/NO backfilled (PENDING only 7.5k).
- 2026-09-09: 525 markets (105/asset), 84.5k snapshots, 106k depth rows; strike_value rows = 0.
- `live_markets` ≈ 35.6k rows (35,592 PENDING / 11 RESOLVED mirror state — NOT the label source).
- `underlying_observations` ≈ 1.1M rows, BINANCE only.
- Gate implication: CT-vs-price-control comparison is feasible now (history+quotes+depth+outcomes present); M0–M4 need canonical strike+underlying (retrospective Chainlink path unproven → model track toward DATA_BLOCKED unless restored).

## First export slice 2026-09-09 (verified SHA-256, local = server)
- `canonical_live_markets.csv.gz` 35,603 rows — `43b54cab…d0441`
- `canonical_snap_20260909.csv.gz` 84,510 rows — `edcdc8fc…0d8c8e`
- `canonical_depth_20260909.csv.gz` 106,660 rows — `21ce24f3…7082f`
- SQL: `export_live.sql`, `export_snap.sql`, `export_depth.sql` (COPY needed-columns, day-chunked, read-only + statement_timeout; gzip+compose-cp+scp, no PS binary redirect).
- Slice registry: 525 markets, all resolved (253 YES / 272 NO); token map 100% OK; durations MIXED (15m + multi-hour — duration per market from `end_time_est`); derived decision window [285,300]s hits 121 markets; depth history ≥3 obs on both legs in 92 markets.
