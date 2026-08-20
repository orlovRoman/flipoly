# T03 — Контракт исторических свечей

## CryptoCandle ORM (db/models.py:27)

| Column | Type | Nullable | Notes |
|---|---|---|---|
| id | Integer PK | no | autoincrement |
| symbol | String(32) | no | e.g. BTC, ETH |
| interval | String(8) | no | e.g. 15m |
| open_time | DateTime(tz=True) | no | **Partition key** — UTC |
| close_time | DateTime(tz=True) | yes | NULL if is_closed=false |
| is_closed | Boolean | yes | NULL possible in old rows |
| open/high/low/close | Float | no | OHLC |
| volume | Float | no | |
| taker_buy_volume | Float | yes | |
| source | String(16) | no | default='binance' |

**Unique**: `(symbol, interval, open_time)` — uix_crypto_candle
**Check**: `(is_closed = false) OR (is_closed IS NULL) OR (close_time IS NOT NULL)`
**Indexes**: `(symbol, interval)`, `(open_time)`

## Repository (candle_repository.py)

### get_recent_candles(symbol, interval, limit=200)
- **Filter**: `is_closed = True` ONLY — no incomplete candles returned
- **Order**: `open_time DESC` → reversed to ASC (oldest first)
- **Return**: `Sequence[CryptoCandle]`
- **Boundary safety**: If `is_closed` is NULL (legacy rows), candle is excluded

### get_latest_open_time(symbol, interval)
- Returns `MAX(open_time)` across ALL candles (including open)
- Used for incremental fetch, not for feature building

### has_incomplete_candles(symbol, interval)
- Checks for `is_closed IS NULL OR close_time IS NULL`
- Diagnostic only

### upsert_candles(symbol, interval, candles)
- ON CONFLICT DO NOTHING (PG) or DO UPDATE (SQLite)
- Never transitions `is_closed=True` back to False (uses CASE guard)

## Decision Boundary Rule

For feature building at decision time `T`:
1. Query `get_recent_candles(symbol, '15m', limit=200)`
2. All returned candles have `is_closed=True` → `open_time + 15m ≤ T`
3. No lookahead: last candle's `close_time ≤ T`
4. Feature builder uses `candles[:-1]` for historical, `candles[-1]` as current

## Pruning

- `prune_old_candles()`: deletes `open_time < (now - 90 days)`
- Runs daily from scheduler, non-blocking with collector

## Known Limitations

1. `is_closed` can be NULL in legacy rows — `get_recent_candles` excludes them
2. `close_time` is nullable — closed candle check relies on `is_closed` flag
3. No explicit `timezone` column — all times assumed UTC (Binance native)
4. `taker_buy_volume` nullable — feature builder must handle NULL→0.0
5. Retention 90 days — MRF needs 24h only, well within limit

## Conclusion

The candle contract is **sound** for MRF:
- `get_recent_candles()` guarantees only closed candles
- `is_closed=True` implies `close_time IS NOT NULL` (check constraint)
- 90-day retention covers MRF's 24h horizon
- No changes needed to candle infrastructure for T04-T08
