# MRF Runbook — Shadow → Active Rollout

## 1. Modes

| Mode | Env Var | Behavior |
|------|---------|----------|
| `OFF` | `MARKET_REGIME_FILTER_MODE=OFF` | MRF完全 disabled. No logging, no blocking. Default. |
| `SHADOW` | `MARKET_REGIME_FILTER_MODE=SHADOW` | Classifies regime, logs to audit, but does NOT modify actions or bet sizes. |
| `ACTIVE` | `MARKET_REGIME_FILTER_MODE=ACTIVE` | Classifies regime and applies multiplier to bet_size_usdc. Blocked trades get action=SKIP. |

**Switching modes**: Change `MARKET_REGIME_FILTER_MODE` in `.env` and restart `execution_worker_paper` container. No code rebuild needed.

## 2. Candle Validity

MRF requires **97 closed 15m candles** per asset (BTC, ETH, SOL, XRP) from Binance.

- `MIN_HISTORY_CANDLES = 97` (24h + 1 candle)
- Candles must be **closed** (no forming bars)
- `open_time <= decision_time - 15min` (lookahead guard)
- `build_snapshot_from_candles()` auto-filters future candles
- `validate_candle_continuity()` checks: duplicates, sorting, gaps > 30min, span < 25h

**Stale data indicators**:
- `history_ready=False` in snapshot → insufficient candles
- `reason_codes` containing `insufficient_history:` → specific asset missing data
- `reason_codes` containing `candle_continuity:` → gaps or duplicates detected

## 3. Regime Classification

| Regime | Condition | Typical % (45d) |
|--------|-----------|------------------|
| `MIXED` | Contradictory signals | 64% |
| `SIDEWAYS` | Low return + low efficiency | 27% |
| `HIGH_VOL_CHOP` | High vol ratio + low efficiency | 5-7% |
| `TREND_UP` | Positive return + efficiency + bullish breadth | 2-3% |
| `TREND_DOWN` | Negative return + efficiency + bearish breadth | 1-2% |
| `UNKNOWN` | Insufficient history | 0% |

**Classifier thresholds** (default, tuned for 15m crypto):
- `trend_ret_threshold=0.02` (2% over 24h)
- `sideways_ret_max=0.005`
- `trend_efficiency_min=0.4`
- `breadth_strong_threshold=0.65`
- `high_vol_ratio_threshold=1.5`

## 4. Policy Rules (OUTSIDER strategy)

| Regime | Outsider Trade | Multiplier |
|--------|---------------|------------|
| `SIDEWAYS` | Allow | 1.0 |
| `HIGH_VOL_CHOP` | Block | 0.0 |
| `TREND_UP` | Block (outsider fades trend) | 0.0 |
| `TREND_DOWN` | Block (outsider fades trend) | 0.0 |
| `MIXED` | Allow with reduction | 0.5 |
| `UNKNOWN` | Allow with reduction | 0.8 |

## 5. Reason Codes

Every regime classification produces reason codes for audit:
- `ret_24h:0.0123` — 24h return
- `efficiency:0.42` — trend efficiency ratio
- `up_ratio:0.58` — fraction of bullish candles
- `high_vol_ratio:1.8` — vol_4h / vol_24h
- `insufficient_history` — not enough candles
- `contradictory_signals` — MIXED classification
- `candle_continuity:gap_at_index_N:Xmin` — candle gap detected

## 6. Telemetry

MRF writes `mrf_applied` log entry per decision:
```json
{
  "event": "mrf_applied",
  "asset": "BTC",
  "mode": "SHADOW",
  "regime": "SIDEWAYS",
  "original_action": "BUY_YES",
  "adjusted_action": "BUY_YES",
  "original_bet": 10.0,
  "adjusted_bet": 10.0,
  "applied": false
}
```

In SHADOW mode, `applied=false` always. In ACTIVE mode, `applied=true` when regime blocks/modifies trade.

## 7. Rollback

To disable MRF instantly:
1. Set `MARKET_REGIME_FILTER_MODE=OFF` in `.env`
2. Restart: `docker compose up -d execution_worker_paper`
3. No code changes needed — MRF checks mode at decision time

## 8. Criteria Before ACTIVE

Before switching from SHADOW to ACTIVE, verify:
1. **Shadow mode stable**: 7+ days of continuous SHADOW logging without errors
2. **Regime distribution reasonable**: Not 100% UNKNOWN or MIXED
3. **No false positives**: Regime classifications match manual chart review
4. **Backtest positive delta**: ACTIVE PnL > BASELINE PnL over sufficient sample
5. **Block rate acceptable**: 5-25% of trades blocked (not 0%, not 90%)

**DO NOT** switch to ACTIVE based on 2 days of data. Minimum 2 weeks of shadow logging recommended.

## 9. Key Files

| File | Purpose |
|------|---------|
| `polyflip/crypto/market_regime.py` | Feature builder (returns, vol, efficiency) |
| `polyflip/crypto/market_regime_classifier.py` | Regime classification (6 regimes) |
| `polyflip/crypto/market_regime_policy.py` | Strategic policy (allow/block/multiplier) |
| `polyflip/crypto/market_regime_apply.py` | Bridge to decision (applies policy to trade) |
| `polyflip/crypto/market_regime_integration.py` | Snapshot builder from ORM candles |
| `polyflip/crypto/market_regime_audit.py` | Telemetry serialization |
| `polyflip/trading/decision_runners.py:203` | `_apply_mrf_filter()` — integration point |
| `polyflip/trading/trading_config.py:73-79` | MRF config fields |

## 10. Known Limitations

- **Trend detection rare**: Only 2-4% of evaluations classify as TREND on 15m data
- **MIXED dominates**: ~64% of regimes are MIXED (low signal-to-noise ratio)
- **No profitability guarantee**: MRF filters bad regimes but doesn't create edge
- **Crypto_candles table may be empty**: MRF reads from Binance via `get_recent_candles()`, requires network access
