# Market Regime Filter — Architecture Reconnaissance (T01)

**Branch:** `codex/market-regime-filter-implementation`  
**Commit:** `72d3d63`  
**Date:** 2026-08-20  
**Scope:** Code read-only, no modifications.

---

## 1. Data Pipeline: Candle Ingestion → Feature Building → ML Inference → Trade Decision

### 1.1 Candle Storage

**Table:** `crypto_candles` — `polyflip/db/models.py:27-53`

| Column | Type | Notes |
|--------|------|-------|
| `symbol` | `String(32)` | e.g. `BTCUSDT`, `ETHUSDT`, `SOLUSDT` |
| `interval` | `String(8)` | e.g. `15m` |
| `open_time` | `DateTime(tz)` | UTC candle open time |
| `is_closed` | `Boolean` | True when candle is finalized |
| `open/high/low/close/volume` | `Float` | OHLCV |
| `taker_buy_volume` | `Float` | Taker-buy side volume |

**Unique constraint:** `(symbol, interval, open_time)` — `uix_crypto_candle`

**Secondary table:** `binance_15m_bars` — `models.py:742-784` — separate historical bars with `asset` (`BTCUSDT`), `open_time`, `close_time`, OHLCV, `quote_asset_volume`, `number_of_trades`, `taker_buy_base_volume`. Used by `Binance15mFeature` pre-computed features and training pipeline.

**Pre-computed features:** `binance_15m_features` — `models.py:861-920` — per-bar pre-computed FS_D0 features (`ret_1..ret_24`, `volatility_6/24`, `rsi_14`, `atr_14`, `volume_ratio_6/24`) plus `extra_features` JSONB. Indexed by `(asset, open_time, feature_set)`.

### 1.2 Feature Builder

**File:** `polyflip/crypto/feature_builder.py`

**Entry:** `build_features(candles: list[dict], symbol: str) -> dict` (line ~36)  
**Input:** list of OHLCV dicts (min 100 candles recommended, max 200)  
**Output:** dict of ~18 features used by LightGBM

Key features produced:
- `ret_1`, `ret_3`, `ret_6`, `ret_12`, `ret_24` — multi-horizon returns
- `volatility_6`, `volatility_24` — rolling std of returns
- `rsi_14`, `atr_14` — standard technical indicators
- `volume_ratio_6`, `volume_ratio_24` — volume vs moving average
- `vol_trend` — volatility trend signal (used for regime classification)

**Validator:** `CryptoFeaturesValidator` — `predictor.py:55-75` — checks required features exist and are finite.

### 1.3 LightGBM Predictor

**File:** `polyflip/crypto/predictor.py`

**Class:** `CryptoPredictor` (line ~100)

**Key method:** `predict(symbol, features) -> CryptoSignal`  
**Returns:** `CryptoSignal` dataclass (line 77-94):
```python
@dataclass
class CryptoSignal:
    regime: str = ""           # "low_vol" / "mid_vol" / "high_vol"
    status: str = "MODEL_NOT_LOADED"
    direction: str = "NONE"    # "UP" / "DOWN" / "NONE"
    p_up: float = 0.0
    p_down: float = 0.0
    signal_strength: float = 0.0
    strike: float | None = None
    threshold_up: float = 0.55
    threshold_down: float = 0.45
    model_key: str = ""
    model_version: int = -1
    features_ok: bool = False
    risk_vetoed: bool = False
    risk_reason: str | None = None
    stake_multiplier: float = 1.0
    funding_rate: float = 0.0
    ece: float = 0.0
    inverted: bool = False
    p_up_raw: float = 0.0
    p_down_raw: float = 0.0
```

**Model loading:** `_load_models(symbol)` (line ~220) — loads per-regime models from `model_registry` where `asset = f"{symbol}_{regime}"` and `is_active = True`. Regimes: `low_vol`, `mid_vol`, `high_vol`.

**Regime classification:** Uses `VolatilityRegimePolicy.classify(vol_trend)` — defined in `polyflip/crypto/backtester.py` or imported. The `vol_trend` feature is computed from candle data.

**Threshold management:** Per-regime thresholds stored in `model_registry.decision_threshold` (UP) and `model_registry.decision_threshold_down` (DOWN). Loaded into `self._thresholds[symbol][regime]`.

**Models are cached:** `self._models: dict[str, dict[str, Any]]` — symbol → regime → LGBM model.

### 1.4 Market Direction Service

**File:** `polyflip/crypto/market_direction_service.py`

**Key function:** `get_or_create_market_direction_signal(market_id, asset, symbol, ...) -> MarketDirectionSignal` (line 50)

- Creates or retrieves a `MarketDirectionSignal` record per `market_id` (unique constraint)
- Calls `CryptoPredictor.predict()` internally
- Persists signal once per market lifetime (no recalculation on repeat calls)
- Returns signal with `regime`, `direction`, `p_up`, `p_down`, `status`, etc.

**DB Model:** `MarketDirectionSignal` — `models.py:1539-1575`
- `market_id` (unique), `asset`, `symbol`, `regime`, `direction`, `p_up`, `p_down`, `signal_strength`, `strike`, `threshold_up/down`, `model_key`, `model_version`, `features_ok`, `risk_vetoed`, `status`, `inverted`, `p_up_raw`, `p_down_raw`, `created_at`

### 1.5 Decision Runners (Trade Decision Entry)

**File:** `polyflip/trading/decision_runners.py`

**Key function:** `_resolve_lgbm_attribution(...)` (line ~177) — calls `get_or_create_market_direction_signal()`

**LightGBM mode control:** `lightgbm_decision_mode` setting (`settings_registry.py:180`):
- `"OFF"` — LightGBM completely disabled
- `"SHADOW"` — model runs, results logged, but NOT applied to trades
- `"ACTIVE"` — model runs AND results affect trade decisions

**Mode check pattern** (used repeatedly):
```python
lgbm_mode = getattr(cfg, "lightgbm_decision_mode", "SHADOW")
is_active = lgbm_mode == "ACTIVE"
is_shadow = lgbm_mode == "SHADOW"
logreg_only = lgbm_mode in {"SHADOW", "OFF"}
```

**Direction status propagation:**
- `"SHADOW_NOT_APPLIED"` when shadow mode
- `"DISABLED_BY_OPERATOR"` when OFF
- Actual status (`READY`, `DIRECTION_NONE_FALLBACK_LR`, `FUNDING_VETOED`, etc.) when ACTIVE

### 1.6 Combined Voting

**File:** `polyflip/trading/combined_voting.py`

**Key classes:**
- `CryptoSignalProxy` (line 40) — wraps `CryptoSignal` for voting
- `DirectionConsensus` (line 49) — result of direction voting
- `resolve_direction_consensus(...)` (line 71) — orchestrates direction decision

**DirectionConsensus fields:**
- `direction_value`: `"UP"` / `"DOWN"` / `"NONE"`
- `direction_status`: `"READY"` / `"FUNDING_VETOED"` / `"LOW_DIRECTION_PROB"` / etc.
- `direction_regime`: from `CryptoSignal.regime`

**Voting paths:**
- `OUTSIDER` strategy — `OUTSIDER_MAX_PRICE = 0.45` (config.py:47), outsider-specific discount `OUTSIDER_PWIN_DISCOUNT = 0.65` (settings_registry.py:138)
- `COMBINED` strategy — combines multiple signals
- `FAVORITE` / `SKIP` — not used actively

### 1.7 Decision Funnel Logging

**File:** `polyflip/trading/funnel_logger.py`

**Key function:** `log_decision_funnel(...)` (line ~55)

**DB Model:** `DecisionFunnelLog` — `models.py:640-740`

Relevant columns for regime filter:
- `direction_regime` (line 707) — existing field, already populated
- `direction_status` (line 708) — existing field
- `direction_probability`, `direction_p_up/down`, `direction_threshold_up/down`, `direction_value`, `direction_raw_opinion`, `direction_p_up_raw/down_raw`
- `g3_dead_zone`, `g4_no_flip`, `g5_min_edge`, `g6_price_range`, `g7_crypto_confirm`, `g8_combined_vote` — funnel gate booleans

---

## 2. Existing Regime Architecture

### 2.1 Volatility Regime (LightGBM)

The existing system already classifies candles into **volatility regimes** (`low_vol`, `mid_vol`, `high_vol`) used for per-regime LightGBM models. This is a **model selection** regime, not a **directional/market state** regime.

- Regime classification happens in `CryptoPredictor.predict()` via `VolatilityRegimePolicy.classify(vol_trend)`
- Each `{symbol}_{regime}` gets its own LightGBM model in `model_registry`
- This is an **internal model detail** — it does NOT gate trading decisions

### 2.2 LightGBM Operational Mode

The `lightgbm_decision_mode` setting controls whether LightGBM predictions are:
- **OFF** — completely disabled, LogReg-only
- **SHADOW** — model runs, logged in funnel, but not applied to trade decisions
- **ACTIVE** — model runs AND its direction vote counts in final decision

This is the **existing mode switch** for directional prediction. MRF will add a new layer ON TOP of this.

### 2.3 What Does NOT Exist Yet

- No `MarketRegimeFilter` class
- No per-market "regime gate" that blocks trades based on market regime
- No market regime stored in `DecisionFunnelLog` or `MarketDirectionSignal` (direction_regime is volatility regime, not market regime)
- No backtest infrastructure for regime-gated trading
- No shadow-rollout mechanism for regime filter

---

## 3. Test Infrastructure

### 3.1 Unit Test Framework

**Config:** `pyproject.toml:46-54`
- `asyncio_mode = "strict"`
- `testpaths = ["tests"]`
- Markers: `live`, `postgres`

**Fixtures:** `tests/conftest.py`
- `engine` — in-memory SQLite per test (full isolation)
- `db_session` — clean SQLAlchemy async session per test
- `pg_session_factory` — PostgreSQL integration (opt-in via `POSTGRES_INTEGRATION_TESTS=1`)
- `clean_models_cache_fixture` — clears LightGBM model cache between tests

### 3.2 Test Commands

```bash
# Unit tests (default, in-memory SQLite)
pytest tests/ -v

# Specific test file
pytest tests/test_<name>.py -v

# With coverage
pytest tests/ --cov=polyflip --cov-report=term-missing

# PostgreSQL integration tests (requires DB)
POSTGRES_INTEGRATION_TESTS=1 TEST_DATABASE_URL="..." pytest tests/ -v -m postgres
```

### 3.3 Test Data Patterns

- `make_dummy_execution()` in conftest.py — creates mock `TradeExecution` objects
- Tests use SQLite in-memory, so all DB tests are fast and isolated
- No factory library used; tests create model instances directly

---

## 4. Key Integration Points for Market Regime Filter

### 4.1 Where MRF Should Intercept

The regime filter should be evaluated **after** candle/feature loading but **before** the direction consensus and trade decision. The natural insertion point is:

1. **Inside `decision_runners.py`** — after `_resolve_lgbm_attribution()` gets the `CryptoSignal`, before the voting/edge calculation
2. **OR** as a separate gate in the funnel, logged via `funnel_logger.py`

### 4.2 Existing Funnel Gate Pattern

The funnel already has gates `g3`–`g8` as boolean flags. MRF would add a **new gate** (e.g., `g_market_regime`) that:
- Receives candle history
- Computes regime features (momentum, volatility, ADX-like signals)
- Classifies regime as `TREND_UP` / `TREND_DOWN` / `SIDEWAYS` / `HIGH_VOL_CHOP` / `MIXED`
- Returns PASS/REJECT + reason codes
- Is logged in `DecisionFunnelLog` with new columns

### 4.3 Configuration Surface

Existing settings pattern (`settings_registry.py`):
- `lightgbm_decision_mode` — `SHADOW` / `ACTIVE` / `OFF`
- `OUTSIDER_MAX_PRICE`, `OUTSIDER_PWIN_DISCOUNT` — strategy params

MRF should follow this pattern with settings like:
- `MARKET_REGIME_MODE` — `SHADOW` / `ACTIVE` / `OFF`
- `MARKET_REGIME_MIN_HISTORY` — minimum candles required
- Per-regime trade permission flags

### 4.4 DB Schema

`DecisionFunnelLog` already has `direction_regime` (volatility regime) and `direction_status`. MRF will need:
- New columns: `market_regime`, `market_regime_status`, `market_regime_reason`
- OR a new table `market_regime_signals` (similar to `market_direction_signals`)

---

## 5. Discovered Constraints

1. **No future data in candles** — `open_time` is candle open, candles are 15m bars from Binance. `is_closed=True` means candle is finalized. MRF must only use closed candles.

2. **UTC timestamps** — all candle times are UTC (timezone-aware). MRF must work in UTC.

3. **15-minute granularity** — trades happen within 15m windows. Regime features must be computed from 15m bars.

4. **Minimum history** — `feature_builder.py` requires ~100 candles minimum. MRF will likely need fewer (e.g., 24–48 candles = 6–12 hours) but should validate.

5. **SQLite test limitation** — unit tests use SQLite in-memory. Any JSONB-specific features won't work in unit tests; need PostgreSQL integration tests for JSONB columns.

6. **Existing regime = volatility regime** — the `regime` field in `CryptoSignal` and `MarketDirectionSignal` refers to volatility regime for model selection. MRF introduces a **separate** market regime concept. Naming must be clear to avoid collision.

7. **LightGBM model_key pattern** — models are keyed as `{SYMBOL}_{regime}` (e.g., `BTCUSDT_low_vol`). MRF regime is a different concept.

8. **Funnel logging is append-only** — `DecisionFunnelLog` rows are never updated after creation. New MRF columns must be populated at creation time.

9. **`lightgbm_decision_mode`** is the existing operational mode switch. MRF adds a NEW mode switch (`market_regime_mode`) that operates orthogonally.

10. **No migration dependency** — the existing branch has migration `20260819_ai_artifact_contract` already on staging. MRF migrations must not conflict.

---

## 6. Files Read (with line references)

| File | Key Lines | Purpose |
|------|-----------|---------|
| `polyflip/db/models.py` | 27-53, 56-111, 640-740, 742-920, 1539-1575 | CryptoCandle, MarketSnapshot, DecisionFunnelLog, Binance15mBar/Feature, MarketDirectionSignal |
| `polyflip/crypto/feature_builder.py` | 36+ | `build_features()` — OHLCV → 18 features |
| `polyflip/crypto/predictor.py` | 77-94, 100+, 220+ | `CryptoSignal` dataclass, `CryptoPredictor`, model loading & regime classification |
| `polyflip/crypto/market_direction_service.py` | 50+ | `get_or_create_market_direction_signal()` |
| `polyflip/trading/decision_runners.py` | 36+, 51+, 177+, 314+, 477+ | LightGBM mode gating, direction resolution, funnel logging |
| `polyflip/trading/combined_voting.py` | 40+, 49+, 71+, 300+ | `CryptoSignalProxy`, `DirectionConsensus`, `resolve_direction_consensus()` |
| `polyflip/trading/funnel_logger.py` | 55+ | `log_decision_funnel()` — funnel audit logging |
| `polyflip/config.py` | 24, 47 | `ACTIVE_FEATURES`, `OUTSIDER_MAX_PRICE` |
| `polyflip/settings_registry.py` | 62+, 130+, 138+, 180+, 219+, 333+ | Strategy settings including `lightgbm_decision_mode` |
| `polyflip/crypto/backtester.py` | 17+, 41+, 111+, 222+ | `SUPPORTED_BACKTEST_MODES`, backtest regime logic |
| `polyflip/trade_recorder.py` | 64+, 267+ | Trade recording with direction_regime |
| `tests/conftest.py` | 1-132 | Test fixtures, SQLite engine, `make_dummy_execution()` |
| `pyproject.toml` | 1-54 | Dependencies, test config |
