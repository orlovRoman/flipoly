# Market Regime Filter — Design Specification (T02)

**Branch:** `codex/market-regime-filter-implementation`  
**Date:** 2026-08-20  
**Status:** DRAFT — pending orchestrator review  
**Depends on:** T01 recon (`market-regime-filter-recon.md`)

---

## 1. Purpose

A rule-based market regime filter that evaluates the current state of 15-minute Polymarket crypto markets and gates trade decisions. The filter operates **orthogonally** to the existing LightGBM direction model and volatility regime system. It adds no new ML models.

---

## 2. Core Data Contract

### 2.1 `MarketRegimeSnapshot`

```python
@dataclass(frozen=True)
class MarketRegimeSnapshot:
    """Immutable snapshot of market regime at a point-in-time boundary."""

    # Identity
    as_of: datetime               # UTC, candle boundary used for computation
    assets: tuple[str, ...]       # ("BTCUSDT", "ETHUSDT", "SOLUSDT")

    # Per-asset features (keyed by symbol)
    asset_returns: dict[str, AssetReturns]
    # {  "BTCUSDT": AssetReturns(ret_4h=0.012, ret_12h=-0.003, ret_24h=0.021, ...), ... }

    # Global basket features
    basket: BasketFeatures

    # Classification result
    global_regime: Regime          # TREND_UP | TREND_DOWN | SIDEWAYS | HIGH_VOL_CHOP | MIXED
    asset_regimes: dict[str, Regime]  # per-asset classification
    confidence: float              # 0.0–1.0, aggregate confidence

    # Metadata
    reason_codes: tuple[str, ...]  # e.g. ("BREADTH_HIGH", "EFFICIENCY_LOW", ...)
    history_ready: bool            # True if minimum data requirements met
    history_candles: int           # number of candles used
    history_hours: float           # wall-clock hours covered


@dataclass(frozen=True)
class AssetReturns:
    """Per-asset return and volatility features at multiple horizons."""
    ret_4h: float       # return over last 4 hours (16 candles)
    ret_12h: float      # return over last 12 hours (48 candles)
    ret_24h: float      # return over last 24 hours (96 candles)
    volatility: float   # annualized vol of 15m returns over 24h
    efficiency: float   # trend efficiency ratio: |net move| / sum(|bar returns|)
    volume_trend: float # volume ratio vs 24h average


@dataclass(frozen=True)
class BasketFeatures:
    """Cross-asset aggregate features."""
    median_ret_4h: float
    median_ret_12h: float
    median_ret_24h: float
    breadth_up: float       # fraction of assets with ret_4h > threshold
    breadth_down: float     # fraction of assets with ret_4h < -threshold
    dispersion: float       # cross-asset return dispersion (std of ret_4h)
    avg_efficiency: float   # mean efficiency across assets
    avg_volatility: float   # mean volatility across assets
    coverage: int           # number of assets with sufficient data
```

### 2.2 `Regime` Enum

```python
class Regime(str, Enum):
    TREND_UP = "TREND_UP"         # directional uptrend across horizons
    TREND_DOWN = "TREND_DOWN"     # directional downtrend across horizons
    SIDEWAYS = "SIDEWAYS"         # no clear direction, moderate vol
    HIGH_VOL_CHOP = "HIGH_VOL_CHOP"  # high volatility, no direction
    MIXED = "MIXED"               # contradictory signals
```

### 2.3 `RegimePolicyResult`

```python
@dataclass(frozen=True)
class RegimePolicyResult:
    """Output of the regime policy for a specific strategy/trade."""
    allow: bool                    # whether trade is permitted
    stake_multiplier: float        # 0.0–1.0, scaling factor for position size
    reason: str                    # human-readable reason code
    regime: Regime                 # the regime that produced this decision
    mode: str                      # "OFF" | "SHADOW" | "ACTIVE"
    applied: bool                  # True only if mode=ACTIVE and allow was modified
```

---

## 3. Temporal Model

### 3.1 As-Of Boundary

All regime computation uses **closed candles only** with `open_time <= as_of`. The `as_of` timestamp is:
- The market's `recorded_at` or opening time (from `market_direction_service.get_or_create_market_direction_signal()`)
- Frozen for the market's lifetime (no recalculation from future candles)

### 3.2 Horizons

| Horizon | Candles (15m) | Use |
|---------|---------------|-----|
| 4h | 16 | Short-term momentum |
| 12h | 48 | Medium-term trend |
| 24h | 96 | Long-term direction, volatility estimation |

### 3.3 Minimum History

```python
MIN_HISTORY_CANDLES = 96   # 24 hours — needed for 24h horizon + vol estimation
```

If `history_candles < MIN_HISTORY_CANDLES`:
- `history_ready = False`
- `global_regime = MIXED`
- `reason_codes` includes `"INSUFFICIENT_HISTORY"`
- Policy returns `allow=True, stake_multiplier=1.0` (fail-open)

---

## 4. Feature Computation

### 4.1 Per-Asset Features (`AssetReturns`)

Computed from closed candles for each symbol:

```python
def compute_asset_returns(
    candles: list[Candle],  # sorted by open_time ASC, closed only
    as_of: datetime,
) -> AssetReturns | None:
```

- `ret_4h` = `(close[-1] - close[-17]) / close[-17]` if 17+ candles available
- `ret_12h` = `(close[-1] - close[-49]) / close[-49]` if 49+ candles available
- `ret_24h` = `(close[-1] - close[-97]) / close[-97]` if 97+ candles available
- `volatility` = `std(bar_returns[-96:]) * sqrt(96 * 365.25 * 24)` (annualized)
- `efficiency` = `abs(close[-1] - close[-97]) / sum(abs(close[i] - close[i-1]) for i in range(-96, 0))`
- `volume_trend` = `mean(volume[-16:]) / mean(volume[-96:])` (4h avg vs 24h avg)

### 4.2 Global Basket Features (`BasketFeatures`)

```python
def compute_basket_features(
    asset_returns: dict[str, AssetReturns],
) -> BasketFeatures:
```

- `median_ret_4h/12h/24h` = median across available assets
- `breadth_up` = count(ret_4h > BREADTH_THRESHOLD) / coverage
- `breadth_down` = count(ret_4h < -BREADTH_THRESHOLD) / coverage
- `dispersion` = std(ret_4h across assets)
- `avg_efficiency` = mean(efficiency across assets)
- `avg_volatility` = mean(volatility across assets)
- `coverage` = number of assets with valid `AssetReturns`

Default `BREADTH_THRESHOLD = 0.005` (0.5% over 4h).

---

## 5. Classification Rules

### 5.1 Global Regime

```python
def classify_global_regime(basket: BasketFeatures) -> tuple[Regime, list[str]]:
```

| Condition | Regime | Reason Codes |
|-----------|--------|-------------|
| `breadth_up >= 0.67` AND `median_ret_4h > 0` AND `median_ret_12h > 0` AND `avg_efficiency > EFFICIENCY_THRESHOLD` | `TREND_UP` | `BREADTH_UP`, `RET_POSITIVE`, `EFFICIENCY_HIGH` |
| `breadth_down >= 0.67` AND `median_ret_4h < 0` AND `median_ret_12h < 0` AND `avg_efficiency > EFFICIENCY_THRESHOLD` | `TREND_DOWN` | `BREADTH_DOWN`, `RET_NEGATIVE`, `EFFICIENCY_HIGH` |
| `avg_volatility > VOL_THRESHOLD` AND `avg_efficiency < EFFICIENCY_THRESHOLD` | `HIGH_VOL_CHOP` | `VOLATILITY_HIGH`, `EFFICIENCY_LOW` |
| `abs(median_ret_4h) < SIDEWAYS_THRESHOLD` AND `abs(breadth_up - breadth_down) < 0.34` | `SIDEWAYS` | `LOW_RET`, `MIXED_BREADTH` |
| else | `MIXED` | `CONFLICTING_SIGNALS` |

Default thresholds:
```python
EFFICIENCY_THRESHOLD = 0.3   # trend efficiency ratio
VOL_THRESHOLD = 0.8          # annualized vol (high = choppy)
SIDEWAYS_THRESHOLD = 0.003   # 0.3% median 4h return
```

### 5.2 Per-Asset Regime

```python
def classify_asset_regime(ret: AssetReturns) -> tuple[Regime, list[str]]:
```

Per-asset uses the same rules but on single-asset features:
- `TREND_UP`: `ret_4h > 0` AND `ret_12h > 0` AND `efficiency > EFFICIENCY_THRESHOLD`
- `TREND_DOWN`: `ret_4h < 0` AND `ret_12h < 0` AND `efficiency > EFFICIENCY_THRESHOLD`
- `HIGH_VOL_CHOP`: `volatility > VOL_THRESHOLD` AND `efficiency < EFFICIENCY_THRESHOLD`
- `SIDEWAYS`: `abs(ret_4h) < SIDEWAYS_THRESHOLD` AND `abs(ret_12h) < SIDEWAYS_THRESHOLD`
- `MIXED`: everything else

---

## 6. Strategy Policy

### 6.1 Policy Function

```python
def regime_policy(
    snapshot: MarketRegimeSnapshot,
    strategy: str,          # "OUTSIDER" | "ML_TREND"
    asset: str | None,      # specific asset, or None for global
    mode: str,              # "OFF" | "SHADOW" | "ACTIVE"
) -> RegimePolicyResult:
```

### 6.2 Policy Matrix

| Regime | OUTSIDER | ML_TREND |
|--------|----------|----------|
| `TREND_UP` | **BLOCK** (stake_mult=0.0) — outsider bets against strong trend | **ALLOW direction=UP** (mult=1.0), **BLOCK direction=DOWN** (mult=0.0) |
| `TREND_DOWN` | **BLOCK** (stake_mult=0.0) — outsider bets against strong trend | **ALLOW direction=DOWN** (mult=1.0), **BLOCK direction=UP** (mult=0.0) |
| `SIDEWAYS` | **ALLOW** (mult=1.0) — ideal for outsider contrarian | **ALLOW** (mult=0.7) — reduced confidence |
| `HIGH_VOL_CHOP` | **REDUCE** (mult=0.3) — too noisy | **REDUCE** (mult=0.3) — too noisy |
| `MIXED` | **ALLOW** (mult=1.0) — default, no regime signal | **ALLOW** (mult=0.8) — slight caution |

### 6.3 Mode Behavior

| Mode | Effect |
|------|--------|
| `OFF` | Policy not evaluated. `allow=True, applied=False`. Trade proceeds unchanged. |
| `SHADOW` | Policy evaluated and logged in funnel. `allow`/`stake_multiplier` computed but NOT applied to actual trade decision. `applied=False`. |
| `ACTIVE` | Policy evaluated, logged, AND applied. `applied=True`. Trade decision uses `stake_multiplier` and `allow` flag. |

---

## 7. Fail-Safe Behavior

1. **Missing candles**: If `< MIN_HISTORY_CANDLES` available → `history_ready=False`, regime=`MIXED`, policy returns `allow=True, mult=1.0` (fail-open)
2. **Missing asset**: If one of BTC/ETH/SOL has no candles → exclude from basket, compute with remaining assets. If 0 assets available → `MIXED`, fail-open.
3. **NaN/Inf in features**: Treat asset as missing, log warning
4. **DB read failure**: Fail-open, `allow=True, mult=1.0`, log error
5. **Classification ambiguity**: Default to `MIXED` (fail-open)
6. **Never block on unknown errors**: All failures result in `allow=True`

---

## 8. Naming Distinction

| Concept | Existing System | MRF New |
|---------|----------------|---------|
| Volatility regime | `CryptoSignal.regime` = `"low_vol"` / `"mid_vol"` / `"high_vol"` — used for model selection | **NOT affected** |
| LightGBM mode | `lightgbm_decision_mode` = `OFF/SHADOW/ACTIVE` — controls direction model application | **NOT affected** |
| Market regime | — | `MarketRegimeSnapshot.global_regime` = `TREND_UP/DOWN/SIDEWAYS/HIGH_VOL_CHOP/MIXED` — gates trade decisions |
| MRF mode | — | `market_regime_filter_mode` = `OFF/SHADOW/ACTIVE` — controls MRF application |

---

## 9. Integration Point

### 9.1 Where MRF Intercepts

In `decision_runners.py`, after `_resolve_lgbm_attribution()` returns the `CryptoSignal` and before the edge/voting calculation:

```
1. Load candles for asset (closed, up to as_of)
2. compute_asset_returns() for each asset
3. compute_basket_features()
4. classify_global_regime() + classify_asset_regime()
5. Build MarketRegimeSnapshot
6. Call regime_policy() for the current strategy
7. If mode=ACTIVE and policy.allow=False → SKIP trade
8. If mode=ACTIVE and policy.stake_multiplier < 1.0 → scale position
9. Log everything in DecisionFunnelLog
```

### 9.2 Funnel Logging Extensions

New columns in `DecisionFunnelLog`:
- `market_regime` (String(32)) — global regime classification
- `market_regime_asset` (String(32)) — per-asset regime
- `market_regime_mode` (String(8)) — OFF/SHADOW/ACTIVE
- `market_regime_allow` (Boolean) — policy allow/deny
- `market_regime_multiplier` (Float) — stake multiplier
- `market_regime_reasons` (String(512)) — comma-separated reason codes
- `market_regime_applied` (Boolean) — whether policy was actually applied

---

## 10. What This Does NOT Do

- Does NOT add a new LightGBM model
- Does NOT change volatility regime classification
- Does NOT modify LightGBM mode (`lightgbm_decision_mode`)
- Does NOT affect entry-model selection, funding veto, stop-loss, settlement, or daily limits
- Does NOT change production database schema beyond new nullable columns
- Does NOT automatically enable `ACTIVE` mode
- Does NOT claim profitability based on limited backtest data

---

## 11. Configuration Defaults

```python
# settings_registry.py additions
MARKET_REGIME_FILTER_MODE = "OFF"           # OFF by default, never ACTIVE
MARKET_REGIME_FILTER_VERSION = "1.0.0"
MARKET_REGIME_MIN_HISTORY = 96              # candles (24h)
MARKET_REGIME_OUTSIDER_TREND_MULTIPLIER = 0.0
MARKET_REGIME_UNKNOWN_MULTIPLIER = 1.0
MARKET_REGIME_BREADTH_THRESHOLD = 0.005
MARKET_REGIME_EFFICIENCY_THRESHOLD = 0.3
```

---

## 12. Verification Checklist

- [x] No future features used (only closed candles with `open_time <= as_of`)
- [x] Volatility regime (`low_vol/mid_vol/high_vol`) NOT mixed with market regime (`TREND_UP/DOWN/...`)
- [x] No new LightGBM model added
- [x] All `MarketRegimeSnapshot` fields are frozen (immutable dataclass)
- [x] Fail-safe: all error paths → `allow=True`
- [x] `SHADOW` mode never modifies trade decisions
- [x] `ACTIVE` not set as default
- [x] UTC timestamps throughout
