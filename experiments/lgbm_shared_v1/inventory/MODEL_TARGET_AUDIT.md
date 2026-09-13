# MODEL_TARGET_AUDIT.md — Этап 2 (lgbm_shared_v1)

## Registry scope
- 1069 rows, blobs on all (143.5 MB lgbm + 4.9 MB logreg), 37 active.
- model_weights fallback: entirely NULL (primary model_blob only).
- Namespaces: `{SYM}USDT_{high,mid,low}_vol` slots (per-(symbol,regime) version
  sequences) + plain/phase assets (BTC/SOL/XRP/BTC_decided/…).

## Target taxonomy (truth-priority: manifest > commit+code > description)
- CANONICAL_POLYMARKET_DIRECTION: 141 (all lgbm, manifest POLYMARKET_FINAL_OUTCOME,
  trained 08-09..09-09). Horizon: contract outcome.
- POLYMARKET_FLIP_VS_FINAL_OUTCOME: 201 (all logreg, manifest, trained 08-10..09-08).
  Horizon: market end, favorite-loses.
- LEGACY_BINANCE_DIRECTION: 298 (all lgbm 15m, manifest NULL, ret_*-only features,
  trained 07-04..08-06). Proof: birth code 482a6914
  (`target = (ret_1.shift(-1) > 0)`, "next candle Up?", epsilon-filtered) +
  guard hint in 0968d065 ("Binance synthetic target ret_1.shift(-1) deprecated").
  Horizon: next 15m candle. Caveats: epsilon filter dropped small-move candles
  (native eval on ALL next candles is the honest comparison); pre-08-09
  shift(-1) training had lookahead (ecfded5a removed it) so legacy backtest
  metrics are suspect.
- UNKNOWN_TARGET: 429 (408 mid-state logreg 06-26..08-19 no manifest/proof;
  16 primordial logreg ids 1-16 empty schema; 5 lgbm ids 1025-1029 features='F').
- TARGET_CONFLICT: 0.

## Blob verification (server, repo context; local-only loads fail on custom class)
- 1069/1069 deserialize. Types: CalibratedClassifierCV 804, Pipeline 142,
  LogisticRegression 88, CalibratedLightGBMModel 30, dict 5.
- Schema match for all non-empty schemas; zeros-predict OK everywhere.
- NaN-predict fails for all 625 logreg-family (era imputation required at replay;
  E1 must replicate era preprocessing or use complete vectors only).
- ids 1025-1029 blobs are DICTS, not models (incl. 2 ACTIVE: SOL v46, XRP v46).
  E1 status for them: NON_REPRODUCIBLE. Active dict-blobs = production-hygiene
  violation (recorded, not fixed here).

## Usage timeline (funnel direction slots, 08-03..09-13)
- 75 mapped key×version (97.3%) + 2 unmapped (ETHUSDT_low_vol v26: 4687 rows;
  SOLUSDT_mid_vol v16: 9978 rows) = 14,665 rows MISSING_MODEL_ARTIFACT.
- Mapped usage: LEGACY 173,431 + CANONICAL 197,204 rows.
- Pre-guard era used NULL-manifest (legacy-era) versions; post-08-12 usage is
  100% CANONICAL. 381 transition rows (08-09 commit..08-12 deploy) carry
  probability on legacy versions — pre-deploy, NOT guard violations.
- Zero post-deploy TARGET_GUARD_VIOLATIONS. Guard-effective date 08-12
  (first FINAL usage; deployment proxy, commit was 08-09).
- 33,568 rows carry version+probability but NULL model key (fallback path?);
  asset+version resolution is Этап-4 builder work, else UNKNOWN_MODEL.
- F1 validation (07-27..08-03) has ~no direction data (usage starts 08-03):
  paired replay covers 08-03+ only.

## Active-set anomalies (no action taken here)
- 15 CANONICAL lgbm + 15 FLIP logreg + 7 NULL-manifest actives.
- Gate-failed actives: BTC_decided v19 (override=t), XRP v44 (override=f!).
- Legacy-era actives since July (SOL v15, ETH_contested v1, SOL_decided v2).
- F-schema dict actives SOL v46 / XRP v46 (cannot predict).
- "Active" != "loaded by direction predictor" (separate consumers exist);
  predictor-path verification above is scoped to funnel direction slots.

## Open unknowns for later phases
- 408 midfam logreg target (UNKNOWN; logreg-improvement experiment ids give
  era, not target proof).
- Era imputer for logreg NaN replay.
- Plain/phase-asset actives' consumers (combined path, out of direction scope
  except as fixed policy context).
