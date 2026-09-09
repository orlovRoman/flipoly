"""
polyflip/trading/ct_policy.py

Core implementation of the Counter-Trend (CT) Outsider Strategy:
- Immutable specification BTC_CT_T5_V1 with deterministic cryptographic hashing.
- Unified CT calculation function with causal sorting and NaN filtering.
- Pure decision function evaluate_ct_policy (no DB, models, or execution side effects).
- Explicit skip reasons: INSUFFICIENT_HISTORY, INVALID_SERIES, CALCULATION_ERROR, PARITY, etc.
- Symmetric outsider selection (UP vs DOWN) using causal mid prices.
- Diagnostic evaluation of all 3 variants on each market.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np
import structlog

from polyflip.research.regime_features import classify_local_regime

logger = structlog.get_logger(__name__)


# ==============================================================================
# 1. Specification BTC_CT_T5_V1
# ==============================================================================

@dataclass(frozen=True)
class CTSpecification:
    """
    Versioned specification for the CT Outsider profile.
    Changing any parameter deterministically changes the specification hash.
    """
    spec_id: str = "BTC_CT_T5_V1"
    version: int = 1
    execution_contour: str = "PAPER"
    asset: str = "BTC"
    contract_horizon_min: int = 15
    # Decision window: 210 to 300 seconds (3.5 to 5.0 minutes) before expiration
    decision_window_min_sec: float = 210.0
    decision_window_max_sec: float = 300.0
    # Price boundaries: ask of chosen outsider in [0.01, 0.40] inclusive
    price_min: float = 0.01
    price_max: float = 0.40
    # Lookback window for CT history: 15 minutes prior to decision
    ct_history_window_sec: float = 900.0
    # Minimum valid observations in the causal window
    min_observations: int = 3
    # Required signal state from classifier
    required_signal: str = "REVERSION"
    # Order sizing: up to $1 purchase cost, taker fee accounted separately
    max_purchase_cost_usdc: float = 1.00
    taker_fee_rate: float = 0.002
    # Exit policy: hold until market settlement
    exit_policy: str = "HOLD_TO_SETTLEMENT"
    # One decision per market and profile version
    max_decisions_per_market: int = 1
    # Symmetric outsider selection tolerance for parity
    parity_tolerance: float = 1e-4
    # Maximum acceptable quote age / staleness
    max_staleness_sec: float = 15.0
    # Classifier name and pinned hyperparameters
    classifier_name: str = "classify_local_regime"
    classifier_params: dict[str, float] = field(default_factory=lambda: {
        "er_trend_thresh": 0.60,
        "er_rev_thresh": 0.40,
        "sign_change_thresh": 0.45,
        "autocorr_thresh": -0.05,
        "quiet_vol_thresh": 1e-4,
        "quiet_path_thresh": 1e-4,
    })

    def canonical_dict(self) -> dict[str, Any]:
        """Returns a canonical dictionary of all parameters for hashing and audit."""
        data = asdict(self)
        # Ensure nested dict is sorted
        if "classifier_params" in data:
            data["classifier_params"] = dict(sorted(data["classifier_params"].items()))
        return dict(sorted(data.items()))

    @property
    def spec_hash(self) -> str:
        """
        Deterministic SHA-256 hash of the canonical specification.
        Any change to any parameter alters this hash.
        """
        serialized = json.dumps(self.canonical_dict(), sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


BTC_CT_T5_V1_SPEC = CTSpecification()


def get_btc_ct_t5_v1_spec() -> CTSpecification:
    """Returns the canonical BTC_CT_T5_V1 specification."""
    return BTC_CT_T5_V1_SPEC


# ==============================================================================
# 2. Unified CT Calculation Function
# ==============================================================================

@dataclass(frozen=True)
class CTRegimeResult:
    state: str  # "REVERSION", "TREND", "QUIET", "UNCERTAIN"
    status: str  # "VALID", "INSUFFICIENT_HISTORY", "INVALID_SERIES", "CALCULATION_ERROR"
    valid: bool
    efficiency_ratio: float | None
    sign_change_freq: float | None
    autocorr_lag1: float | None
    local_mean: float | None
    local_std: float | None
    slope_norm: float | None
    observations_count: int
    first_obs_at: datetime | None
    last_obs_at: datetime | None
    raw_prices: tuple[float, ...]
    error_detail: str | None = None


def compute_token_ct_regime(
    observations: Sequence[Mapping[str, Any] | tuple[datetime, float] | float],
    decision_at: datetime,
    window_sec: float = 900.0,
    min_observations: int = 3,
    classifier_params: dict[str, float] | None = None,
) -> CTRegimeResult:
    """
    Unified pure function for preparing and classifying a token price series for CT.
    Shared by research replay, historical audit, and the live/paper trading cycle.

    Enforces:
    1. Causal time boundary: strictly observations with timestamp <= decision_at
       and timestamp >= decision_at - window_sec.
    2. Chronological sorting by timestamp.
    3. Filtering of NaN, non-finite, and non-positive prices.
    4. Explicit classification of INSUFFICIENT_HISTORY (< min_observations) and
       INVALID_SERIES (all-NaN or invalid).
    5. Exception isolation (returns CALCULATION_ERROR on any internal failure).
    """
    if decision_at.tzinfo is None:
        decision_at = decision_at.replace(tzinfo=timezone.utc)

    window_start = decision_at.timestamp() - window_sec
    decision_ts = decision_at.timestamp()

    parsed_obs: list[tuple[float, datetime, float]] = []
    had_raw_items = False

    for item in observations:
        had_raw_items = True
        ts_val: float | None = None
        dt_val: datetime | None = None
        p_val: float | None = None

        if isinstance(item, Mapping):
            # Check received timestamp if available: must not be in the future relative to decision_at
            raw_rec = item.get("received_at") or item.get("received_timestamp")
            if raw_rec is not None:
                dt_rec = None
                if isinstance(raw_rec, datetime):
                    dt_rec = raw_rec if raw_rec.tzinfo is not None else raw_rec.replace(tzinfo=timezone.utc)
                else:
                    try:
                        dt_rec = datetime.fromisoformat(str(raw_rec).replace("Z", "+00:00"))
                    except Exception:
                        dt_rec = None
                if dt_rec is not None and dt_rec.timestamp() > decision_ts:
                    continue

            # Extract timestamp
            raw_t = item.get("recorded_at") or item.get("market_timestamp") or item.get("timestamp") or item.get("event_at")
            if raw_t is not None:
                if isinstance(raw_t, datetime):
                    dt_val = raw_t if raw_t.tzinfo is not None else raw_t.replace(tzinfo=timezone.utc)
                else:
                    try:
                        dt_val = datetime.fromisoformat(str(raw_t).replace("Z", "+00:00"))
                    except Exception:
                        dt_val = None
                if dt_val is not None:
                    ts_val = dt_val.timestamp()

            # Extract price
            raw_p = item.get("mid_price")
            if raw_p is None:
                raw_p = item.get("price")
            if raw_p is None:
                raw_p = item.get("poly_up_mid")
            if raw_p is not None:
                try:
                    p_val = float(raw_p)
                except (ValueError, TypeError):
                    p_val = None

        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            raw_t, raw_p = item[0], item[1]
            if isinstance(raw_t, datetime):
                dt_val = raw_t if raw_t.tzinfo is not None else raw_t.replace(tzinfo=timezone.utc)
                ts_val = dt_val.timestamp()
            elif isinstance(raw_t, (int, float)):
                ts_val = float(raw_t)
                dt_val = datetime.fromtimestamp(ts_val, tz=timezone.utc)
            try:
                p_val = float(raw_p)
            except (ValueError, TypeError):
                p_val = None
        elif isinstance(item, (int, float)):
            try:
                p_val = float(item)
            except (ValueError, TypeError):
                p_val = None
            # If no timestamp provided, treat as relative order within causal window
            ts_val = decision_ts
            dt_val = decision_at

        if ts_val is not None and dt_val is not None and p_val is not None:
            # Causal filtering
            if window_start <= ts_val <= decision_ts:
                if math.isfinite(p_val) and p_val > 0.0:
                    parsed_obs.append((ts_val, dt_val, p_val))

    # Sort strictly by timestamp ascending
    parsed_obs.sort(key=lambda x: x[0])

    if not parsed_obs:
        status = "INVALID_SERIES" if had_raw_items else "INSUFFICIENT_HISTORY"
        return CTRegimeResult(
            state="UNCERTAIN",
            status=status,
            valid=False,
            efficiency_ratio=None,
            sign_change_freq=None,
            autocorr_lag1=None,
            local_mean=None,
            local_std=None,
            slope_norm=None,
            observations_count=0,
            first_obs_at=None,
            last_obs_at=None,
            raw_prices=(),
            error_detail="No valid finite observations found within causal window",
        )

    if len(parsed_obs) < min_observations:
        return CTRegimeResult(
            state="UNCERTAIN",
            status="INSUFFICIENT_HISTORY",
            valid=False,
            efficiency_ratio=None,
            sign_change_freq=None,
            autocorr_lag1=None,
            local_mean=None,
            local_std=None,
            slope_norm=None,
            observations_count=len(parsed_obs),
            first_obs_at=parsed_obs[0][1],
            last_obs_at=parsed_obs[-1][1],
            raw_prices=tuple(x[2] for x in parsed_obs),
            error_detail=f"Observations count {len(parsed_obs)} < required {min_observations}",
        )

    prices_arr = np.array([x[2] for x in parsed_obs], dtype=float)
    params = classifier_params or {}

    try:
        clf = classify_local_regime(
            prices=prices_arr,
            min_observations=min_observations,
            er_trend_thresh=params.get("er_trend_thresh", 0.60),
            er_rev_thresh=params.get("er_rev_thresh", 0.40),
            sign_change_thresh=params.get("sign_change_thresh", 0.45),
            autocorr_thresh=params.get("autocorr_thresh", -0.05),
            quiet_vol_thresh=params.get("quiet_vol_thresh", 1e-4),
            quiet_path_thresh=params.get("quiet_path_thresh", 1e-4),
        )
    except Exception as exc:
        logger.exception("ct_regime_classification_error", error=str(exc))
        return CTRegimeResult(
            state="UNCERTAIN",
            status="CALCULATION_ERROR",
            valid=False,
            efficiency_ratio=None,
            sign_change_freq=None,
            autocorr_lag1=None,
            local_mean=None,
            local_std=None,
            slope_norm=None,
            observations_count=len(parsed_obs),
            first_obs_at=parsed_obs[0][1],
            last_obs_at=parsed_obs[-1][1],
            raw_prices=tuple(prices_arr),
            error_detail=f"Exception during classify_local_regime: {exc}",
        )

    er = float(clf["efficiency_ratio"]) if clf.get("efficiency_ratio") is not None and math.isfinite(clf["efficiency_ratio"]) else None
    sc = float(clf["sign_change_freq"]) if clf.get("sign_change_freq") is not None and math.isfinite(clf["sign_change_freq"]) else None
    ac = float(clf["autocorr_lag1"]) if clf.get("autocorr_lag1") is not None and math.isfinite(clf["autocorr_lag1"]) else None
    mu = float(clf["local_mean"]) if clf.get("local_mean") is not None and math.isfinite(clf["local_mean"]) else None
    sigma = float(clf["local_std"]) if clf.get("local_std") is not None and math.isfinite(clf["local_std"]) else None
    sl = float(clf["slope_norm"]) if clf.get("slope_norm") is not None and math.isfinite(clf["slope_norm"]) else None

    return CTRegimeResult(
        state=str(clf.get("state", "UNCERTAIN")),
        status="VALID",
        valid=True,
        efficiency_ratio=er,
        sign_change_freq=sc,
        autocorr_lag1=ac,
        local_mean=mu,
        local_std=sigma,
        slope_norm=sl,
        observations_count=len(parsed_obs),
        first_obs_at=parsed_obs[0][1],
        last_obs_at=parsed_obs[-1][1],
        raw_prices=tuple(prices_arr),
    )


# ==============================================================================
# 3. Market Token Mapping & Quotes Dataclasses
# ==============================================================================

@dataclass(frozen=True)
class MarketTokenMapping:
    market_id: str
    asset: str
    expiration: datetime
    up_token_id: str
    down_token_id: str

    def validate(self) -> tuple[bool, str | None]:
        """Validates explicit, unambiguous mapping."""
        if not self.market_id or str(self.market_id).strip() == "":
            return False, "MISSING_MARKET_ID"
        if not self.up_token_id or str(self.up_token_id).strip() in {"", "N/A", "None"}:
            return False, "MISSING_UP_TOKEN_ID"
        if not self.down_token_id or str(self.down_token_id).strip() in {"", "N/A", "None"}:
            return False, "MISSING_DOWN_TOKEN_ID"
        if self.up_token_id == self.down_token_id:
            return False, "AMBIGUOUS_TOKEN_MAPPING"
        return True, None


@dataclass(frozen=True)
class SideQuote:
    side: str  # "UP" or "DOWN"
    token_id: str
    best_bid: float | None
    best_ask: float | None
    mid_price: float | None
    snapshot_id: Any | None = None
    event_at: datetime | None = None
    received_at: datetime | None = None

    def compute_age_sec(self, decision_at: datetime) -> float | None:
        """
        Computes quote age in seconds relative to decision_at.
        Uses event_at if available; falls back to received_at with explicit designation.
        Does NOT clamp or zero negative ages.
        """
        ref = self.event_at if self.event_at is not None else self.received_at
        if ref is None:
            return None
        dec = decision_at if decision_at.tzinfo is not None else decision_at.replace(tzinfo=timezone.utc)
        ref_dt = ref if ref.tzinfo is not None else ref.replace(tzinfo=timezone.utc)
        return (dec - ref_dt).total_seconds()

    def is_valid_for_decision(self, decision_at: datetime, max_staleness_sec: float = 15.0) -> tuple[bool, str | None]:
        if self.mid_price is None or not math.isfinite(self.mid_price) or self.mid_price <= 0.0 or self.mid_price >= 1.0:
            return False, "INVALID_MID_QUOTE"
        if self.best_ask is None or not math.isfinite(self.best_ask) or self.best_ask <= 0.0 or self.best_ask >= 1.0:
            return False, "INVALID_ASK_QUOTE"
        if self.best_bid is not None:
            if not math.isfinite(self.best_bid) or self.best_bid <= 0.0 or self.best_bid >= 1.0:
                return False, "INVALID_BID_QUOTE"
            if self.best_bid > self.best_ask:
                return False, "CROSSED_BOOK_QUOTE"

        # Explicit timestamp contract: at least one timing mark (event_at or received_at) is required.
        if self.event_at is None and self.received_at is None:
            return False, "MISSING_QUOTE_TIMESTAMP"

        dec = decision_at if decision_at.tzinfo is not None else decision_at.replace(tzinfo=timezone.utc)

        # 1. Received timestamp must not be later than decision_at
        if self.received_at is not None:
            rec = self.received_at if self.received_at.tzinfo is not None else self.received_at.replace(tzinfo=timezone.utc)
            if rec > dec:
                return False, "FUTURE_QUOTE_DETECTED: FUTURE_QUOTE_RECEIVED: RECEIVED_IN_FUTURE"

        # 2. Event timestamp must not be later than decision_at
        if self.event_at is not None:
            evt = self.event_at if self.event_at.tzinfo is not None else self.event_at.replace(tzinfo=timezone.utc)
            if evt > dec:
                return False, "FUTURE_QUOTE_DETECTED: EVENT_IN_FUTURE"

        # 3. Assess quote age
        age_sec = self.compute_age_sec(dec)
        if age_sec is None:
            return False, "MISSING_QUOTE_TIMESTAMP"

        # Negative age must NOT be zeroed; negative age violates causality.
        if age_sec < 0.0:
            return False, "FUTURE_QUOTE_DETECTED: NEGATIVE_AGE"

        # Staleness boundary: age_sec <= max_staleness_sec is accepted, > max_staleness_sec is stale.
        if age_sec > max_staleness_sec:
            return False, "STALE_QUOTE"
        return True, None



@dataclass(frozen=True)
class CTDecision:
    action: str  # "BUY" or "SKIP"
    side: str | None  # "UP", "DOWN", or None
    token_id: str | None
    reason: str
    spec_id: str
    spec_hash: str
    decision_at: datetime
    time_left_sec: float
    market_id: str
    asset: str
    limit_price: float | None
    budget_usdc: float
    selected_ask: float | None
    selected_mid: float | None
    selected_bid: float | None
    other_mid: float | None
    outsider_margin: float | None
    ct_regime: str
    ct_features: dict[str, Any]
    data_ids: dict[str, Any]
    is_executable: bool


# ==============================================================================
# 4. Pure Decision Function
# ==============================================================================

def evaluate_ct_policy(
    spec: CTSpecification,
    decision_at: datetime,
    market_mapping: MarketTokenMapping,
    up_quote: SideQuote,
    down_quote: SideQuote,
    chosen_token_history: Sequence[Mapping[str, Any] | tuple[datetime, float] | float] | None,
) -> CTDecision:
    """
    Pure decision function for the CT Outsider strategy.

    Inputs:
    - spec: CTSpecification
    - decision_at: decision timestamp (UTC)
    - market_mapping: verified UP/DOWN token IDs and expiration
    - up_quote: causally available quotes for UP token
    - down_quote: causally available quotes for DOWN token
    - chosen_token_history: historical mid price snapshots strictly of the chosen token

    Output:
    - CTDecision with action BUY or SKIP, reason, parameters, and audit data.

    Properties:
    - No database, ML model, LLM, or execution dependencies.
    - Deterministic: identical inputs produce identical outputs.
    - Symmetric: identical evaluation rules for UP and DOWN.
    """
    if decision_at.tzinfo is None:
        decision_at = decision_at.replace(tzinfo=timezone.utc)

    # 1. Validate token mapping and asset alignment
    map_ok, map_err = market_mapping.validate()
    if not map_ok:
        return _make_skip(
            spec=spec,
            decision_at=decision_at,
            market_mapping=market_mapping,
            reason=f"MAPPING_ERROR: {map_err}",
            time_left_sec=0.0,
        )

    if market_mapping.asset.strip().upper() != spec.asset.strip().upper():
        return _make_skip(
            spec=spec,
            decision_at=decision_at,
            market_mapping=market_mapping,
            reason=f"ASSET_MISMATCH: {market_mapping.asset} != {spec.asset}",
            time_left_sec=0.0,
        )

    # 2. Compute time left to expiration
    exp = market_mapping.expiration
    if exp.tzinfo is None:
        exp = exp.replace(tzinfo=timezone.utc)
    time_left_sec = (exp - decision_at).total_seconds()

    up_age = up_quote.compute_age_sec(decision_at)
    down_age = down_quote.compute_age_sec(decision_at)
    timing_diag: dict[str, Any] = {
        "decision_at": decision_at.isoformat(),
        "time_left_sec": round(time_left_sec, 3),
        "up_quote_event_at": up_quote.event_at.isoformat() if up_quote.event_at else None,
        "up_quote_received_at": up_quote.received_at.isoformat() if up_quote.received_at else None,
        "up_quote_age_sec": up_age,
        "down_quote_event_at": down_quote.event_at.isoformat() if down_quote.event_at else None,
        "down_quote_received_at": down_quote.received_at.isoformat() if down_quote.received_at else None,
        "down_quote_age_sec": down_age,
    }

    # 3. Decision window check: [210, 300] seconds
    if not (spec.decision_window_min_sec <= time_left_sec <= spec.decision_window_max_sec):
        return _make_skip(
            spec=spec,
            decision_at=decision_at,
            market_mapping=market_mapping,
            reason="OUTSIDE_WINDOW",
            time_left_sec=time_left_sec,
            data_ids={"timing_diagnostics": {**timing_diag, "skip_reason": "OUTSIDE_WINDOW"}},
        )

    # 4. Validate quotes from both sides
    up_ok, up_err = up_quote.is_valid_for_decision(decision_at, spec.max_staleness_sec)
    down_ok, down_err = down_quote.is_valid_for_decision(decision_at, spec.max_staleness_sec)
    if not up_ok or not down_ok:
        err = up_err or down_err or "MISSING_QUOTE"
        return _make_skip(
            spec=spec,
            decision_at=decision_at,
            market_mapping=market_mapping,
            reason=f"QUOTE_ERROR: {err}",
            time_left_sec=time_left_sec,
            data_ids={"timing_diagnostics": {**timing_diag, "skip_reason": f"QUOTE_ERROR: {err}"}},
        )

    up_mid = float(up_quote.mid_price)  # type: ignore[arg-type]
    down_mid = float(down_quote.mid_price)  # type: ignore[arg-type]

    # 5. Outsider selection by comparing mid prices
    diff = up_mid - down_mid
    if abs(diff) <= spec.parity_tolerance:
        return _make_skip(
            spec=spec,
            decision_at=decision_at,
            market_mapping=market_mapping,
            reason="PARITY",
            time_left_sec=time_left_sec,
            selected_mid=up_mid,
            other_mid=down_mid,
            outsider_margin=0.0,
        )

    if diff < -spec.parity_tolerance:
        chosen_side = "UP"
        chosen_quote = up_quote
        other_mid = down_mid
    else:
        chosen_side = "DOWN"
        chosen_quote = down_quote
        other_mid = up_mid

    chosen_mid = float(chosen_quote.mid_price)  # type: ignore[arg-type]
    chosen_ask = float(chosen_quote.best_ask)  # type: ignore[arg-type]
    chosen_bid = float(chosen_quote.best_bid) if chosen_quote.best_bid is not None else None
    outsider_margin = other_mid - chosen_mid

    # 6. Price filter on chosen outsider's ask: [0.01, 0.40] inclusive
    if not (spec.price_min <= chosen_ask <= spec.price_max):
        return _make_skip(
            spec=spec,
            decision_at=decision_at,
            market_mapping=market_mapping,
            reason="PRICE_FILTER",
            time_left_sec=time_left_sec,
            side=chosen_side,
            token_id=chosen_quote.token_id,
            selected_ask=chosen_ask,
            selected_mid=chosen_mid,
            selected_bid=chosen_bid,
            other_mid=other_mid,
            outsider_margin=outsider_margin,
        )

    # 7. Check history of the chosen token
    if chosen_token_history is None or len(chosen_token_history) == 0:
        return _make_skip(
            spec=spec,
            decision_at=decision_at,
            market_mapping=market_mapping,
            reason="MISSING_TOKEN_HISTORY",
            time_left_sec=time_left_sec,
            side=chosen_side,
            token_id=chosen_quote.token_id,
            selected_ask=chosen_ask,
            selected_mid=chosen_mid,
            selected_bid=chosen_bid,
            other_mid=other_mid,
            outsider_margin=outsider_margin,
        )

    ct_res = compute_token_ct_regime(
        observations=chosen_token_history,
        decision_at=decision_at,
        window_sec=spec.ct_history_window_sec,
        min_observations=spec.min_observations,
        classifier_params=spec.classifier_params,
    )

    if ct_res.status == "INSUFFICIENT_HISTORY":
        return _make_skip(
            spec=spec,
            decision_at=decision_at,
            market_mapping=market_mapping,
            reason="INSUFFICIENT_HISTORY",
            time_left_sec=time_left_sec,
            side=chosen_side,
            token_id=chosen_quote.token_id,
            selected_ask=chosen_ask,
            selected_mid=chosen_mid,
            selected_bid=chosen_bid,
            other_mid=other_mid,
            outsider_margin=outsider_margin,
            ct_regime="UNCERTAIN",
            ct_features=_pack_ct_features(ct_res),
        )

    if ct_res.status == "INVALID_SERIES":
        return _make_skip(
            spec=spec,
            decision_at=decision_at,
            market_mapping=market_mapping,
            reason="INVALID_SERIES",
            time_left_sec=time_left_sec,
            side=chosen_side,
            token_id=chosen_quote.token_id,
            selected_ask=chosen_ask,
            selected_mid=chosen_mid,
            selected_bid=chosen_bid,
            other_mid=other_mid,
            outsider_margin=outsider_margin,
            ct_regime="UNCERTAIN",
            ct_features=_pack_ct_features(ct_res),
        )

    if ct_res.status == "CALCULATION_ERROR":
        return _make_skip(
            spec=spec,
            decision_at=decision_at,
            market_mapping=market_mapping,
            reason="CALCULATION_ERROR",
            time_left_sec=time_left_sec,
            side=chosen_side,
            token_id=chosen_quote.token_id,
            selected_ask=chosen_ask,
            selected_mid=chosen_mid,
            selected_bid=chosen_bid,
            other_mid=other_mid,
            outsider_margin=outsider_margin,
            ct_regime="UNCERTAIN",
            ct_features=_pack_ct_features(ct_res),
        )

    # 8. Check CT regime signal: strictly REVERSION
    if ct_res.state != spec.required_signal:
        return _make_skip(
            spec=spec,
            decision_at=decision_at,
            market_mapping=market_mapping,
            reason=f"REGIME_NOT_REVERSION: {ct_res.state}",
            time_left_sec=time_left_sec,
            side=chosen_side,
            token_id=chosen_quote.token_id,
            selected_ask=chosen_ask,
            selected_mid=chosen_mid,
            selected_bid=chosen_bid,
            other_mid=other_mid,
            outsider_margin=outsider_margin,
            ct_regime=ct_res.state,
            ct_features=_pack_ct_features(ct_res),
        )

    # 9. All criteria satisfied: BUY decision
    dec_dt = decision_at if decision_at.tzinfo is not None else decision_at.replace(tzinfo=timezone.utc)
    up_evt = up_quote.event_at if (up_quote.event_at and up_quote.event_at.tzinfo is not None) else (up_quote.event_at.replace(tzinfo=timezone.utc) if up_quote.event_at else None)
    down_evt = down_quote.event_at if (down_quote.event_at and down_quote.event_at.tzinfo is not None) else (down_quote.event_at.replace(tzinfo=timezone.utc) if down_quote.event_at else None)
    up_rcv = up_quote.received_at if (up_quote.received_at and up_quote.received_at.tzinfo is not None) else (up_quote.received_at.replace(tzinfo=timezone.utc) if up_quote.received_at else None)
    down_rcv = down_quote.received_at if (down_quote.received_at and down_quote.received_at.tzinfo is not None) else (down_quote.received_at.replace(tzinfo=timezone.utc) if down_quote.received_at else None)

    up_age_sec = (dec_dt - up_evt).total_seconds() if up_evt else None
    down_age_sec = (dec_dt - down_evt).total_seconds() if down_evt else None

    data_ids = {
        "opportunity_id": f"{market_mapping.market_id}_{dec_dt.isoformat()}",
        "decision_id": f"CT:{spec.spec_id}:{market_mapping.market_id}",
        "up_snapshot_id": up_quote.snapshot_id,
        "down_snapshot_id": down_quote.snapshot_id,
        "up_quote_at": up_evt.isoformat() if up_evt else None,
        "down_quote_at": down_evt.isoformat() if down_evt else None,
        "up_received_at": up_rcv.isoformat() if up_rcv else None,
        "down_received_at": down_rcv.isoformat() if down_rcv else None,
        "up_quote_age_sec": round(up_age_sec, 3) if up_age_sec is not None else None,
        "down_quote_age_sec": round(down_age_sec, 3) if down_age_sec is not None else None,
        "history_count": ct_res.observations_count,
        "first_obs_at": ct_res.first_obs_at.isoformat() if ct_res.first_obs_at else None,
        "last_obs_at": ct_res.last_obs_at.isoformat() if ct_res.last_obs_at else None,
        "timing_diagnostics": {**timing_diag, "skip_reason": None},
    }

    return CTDecision(
        action="BUY",
        side=chosen_side,
        token_id=chosen_quote.token_id,
        reason="CT_SIGNAL_REVERSION",
        spec_id=spec.spec_id,
        spec_hash=spec.spec_hash,
        decision_at=decision_at,
        time_left_sec=round(time_left_sec, 2),
        market_id=market_mapping.market_id,
        asset=market_mapping.asset,
        limit_price=chosen_ask,
        budget_usdc=spec.max_purchase_cost_usdc,
        selected_ask=chosen_ask,
        selected_mid=chosen_mid,
        selected_bid=chosen_bid,
        other_mid=other_mid,
        outsider_margin=outsider_margin,
        ct_regime=ct_res.state,
        ct_features=_pack_ct_features(ct_res),
        data_ids=data_ids,
        is_executable=True,
    )


# ==============================================================================
# 5. Diagnostic Controls (Requirement 13)
# ==============================================================================

@dataclass(frozen=True)
class CTMarketDiagnostics:
    market_id: str
    decision_at: datetime
    # 1. Historical YES-only control (how the original research model would have decided)
    historical_yes_only: CTDecision
    # 2. Symmetric price control without CT requirement
    symmetric_price_control: CTDecision
    # 3. Symmetric CT policy (the only one that generates an executable PAPER order)
    symmetric_ct_policy: CTDecision


def evaluate_all_ct_diagnostics(
    spec: CTSpecification,
    decision_at: datetime,
    market_mapping: MarketTokenMapping,
    up_quote: SideQuote,
    down_quote: SideQuote,
    up_history: Sequence[Any] | None,
    down_history: Sequence[Any] | None,
) -> CTMarketDiagnostics:
    """
    Evaluates 3 distinct diagnostic results on a single market:
    1. historical_yes_only (is_executable=False)
    2. symmetric_price_control (is_executable=False)
    3. symmetric_ct_policy (is_executable=True if BUY)

    Ensures control calculations NEVER create executable orders or consume book depth.
    """
    if decision_at.tzinfo is None:
        decision_at = decision_at.replace(tzinfo=timezone.utc)

    # 1. Historical YES-only control
    exp = market_mapping.expiration
    if exp.tzinfo is None:
        exp = exp.replace(tzinfo=timezone.utc)
    time_left_sec = (exp - decision_at).total_seconds()

    yes_only_dec: CTDecision
    if not (spec.decision_window_min_sec <= time_left_sec <= spec.decision_window_max_sec):
        yes_only_dec = _make_skip(spec, decision_at, market_mapping, "OUTSIDE_WINDOW", time_left_sec)
    elif up_quote.mid_price is None or up_quote.best_ask is None:
        yes_only_dec = _make_skip(spec, decision_at, market_mapping, "MISSING_QUOTE", time_left_sec)
    elif up_quote.mid_price > 0.5:
        yes_only_dec = _make_skip(
            spec, decision_at, market_mapping, "PRICE_FILTER", time_left_sec,
            side="UP", token_id=up_quote.token_id, selected_mid=up_quote.mid_price, selected_ask=up_quote.best_ask,
        )
    elif not (spec.price_min <= up_quote.best_ask <= spec.price_max):
        yes_only_dec = _make_skip(
            spec, decision_at, market_mapping, "PRICE_FILTER", time_left_sec,
            side="UP", token_id=up_quote.token_id, selected_mid=up_quote.mid_price, selected_ask=up_quote.best_ask,
        )
    elif up_history is None or len(up_history) == 0:
        yes_only_dec = _make_skip(
            spec, decision_at, market_mapping, "MISSING_TOKEN_HISTORY", time_left_sec,
            side="UP", token_id=up_quote.token_id, selected_mid=up_quote.mid_price, selected_ask=up_quote.best_ask,
        )
    else:
        up_ct = compute_token_ct_regime(up_history, decision_at, spec.ct_history_window_sec, spec.min_observations, spec.classifier_params)
        if not up_ct.valid or up_ct.status != "VALID":
            yes_only_dec = _make_skip(
                spec, decision_at, market_mapping, up_ct.status, time_left_sec,
                side="UP", token_id=up_quote.token_id, selected_mid=up_quote.mid_price, selected_ask=up_quote.best_ask,
                ct_regime=up_ct.state, ct_features=_pack_ct_features(up_ct),
            )
        elif up_ct.state != spec.required_signal:
            yes_only_dec = _make_skip(
                spec, decision_at, market_mapping, f"REGIME_NOT_REVERSION: {up_ct.state}", time_left_sec,
                side="UP", token_id=up_quote.token_id, selected_mid=up_quote.mid_price, selected_ask=up_quote.best_ask,
                ct_regime=up_ct.state, ct_features=_pack_ct_features(up_ct),
            )
        else:
            yes_only_dec = CTDecision(
                action="BUY",
                side="UP",
                token_id=up_quote.token_id,
                reason="HISTORICAL_YES_ONLY_CONTROL",
                spec_id=spec.spec_id,
                spec_hash=spec.spec_hash,
                decision_at=decision_at,
                time_left_sec=round(time_left_sec, 2),
                market_id=market_mapping.market_id,
                asset=market_mapping.asset,
                limit_price=up_quote.best_ask,
                budget_usdc=spec.max_purchase_cost_usdc,
                selected_ask=up_quote.best_ask,
                selected_mid=up_quote.mid_price,
                selected_bid=up_quote.best_bid,
                other_mid=down_quote.mid_price,
                outsider_margin=(down_quote.mid_price - up_quote.mid_price) if down_quote.mid_price else None,
                ct_regime=up_ct.state,
                ct_features=_pack_ct_features(up_ct),
                data_ids={"mode": "HISTORICAL_YES_ONLY"},
                is_executable=False,  # DIAGNOSTIC ONLY
            )

    # 2. Symmetric price control without CT regime filter
    diff = (up_quote.mid_price or 0.5) - (down_quote.mid_price or 0.5)
    symm_price_dec: CTDecision
    if not (spec.decision_window_min_sec <= time_left_sec <= spec.decision_window_max_sec):
        symm_price_dec = _make_skip(spec, decision_at, market_mapping, "OUTSIDE_WINDOW", time_left_sec)
    elif up_quote.mid_price is None or down_quote.mid_price is None or up_quote.best_ask is None or down_quote.best_ask is None:
        symm_price_dec = _make_skip(spec, decision_at, market_mapping, "MISSING_QUOTE", time_left_sec)
    elif abs(diff) <= spec.parity_tolerance:
        symm_price_dec = _make_skip(spec, decision_at, market_mapping, "PARITY", time_left_sec)
    else:
        if diff < 0:
            c_side = "UP"
            c_q = up_quote
            o_mid = down_quote.mid_price
        else:
            c_side = "DOWN"
            c_q = down_quote
            o_mid = up_quote.mid_price

        if not (spec.price_min <= c_q.best_ask <= spec.price_max):  # type: ignore[operator]
            symm_price_dec = _make_skip(
                spec, decision_at, market_mapping, "PRICE_FILTER", time_left_sec,
                side=c_side, token_id=c_q.token_id, selected_ask=c_q.best_ask, selected_mid=c_q.mid_price, other_mid=o_mid,
            )
        else:
            symm_price_dec = CTDecision(
                action="BUY",
                side=c_side,
                token_id=c_q.token_id,
                reason="SYMMETRIC_PRICE_CONTROL",
                spec_id=spec.spec_id,
                spec_hash=spec.spec_hash,
                decision_at=decision_at,
                time_left_sec=round(time_left_sec, 2),
                market_id=market_mapping.market_id,
                asset=market_mapping.asset,
                limit_price=c_q.best_ask,
                budget_usdc=spec.max_purchase_cost_usdc,
                selected_ask=c_q.best_ask,
                selected_mid=c_q.mid_price,
                selected_bid=c_q.best_bid,
                other_mid=o_mid,
                outsider_margin=(o_mid - c_q.mid_price) if o_mid and c_q.mid_price else None,
                ct_regime="NOT_EVALUATED",
                ct_features={},
                data_ids={"mode": "SYMMETRIC_PRICE_CONTROL"},
                is_executable=False,  # DIAGNOSTIC ONLY
            )

    # 3. Symmetric CT policy (the real executable policy)
    # Determine outsider first to pick the corresponding token history
    if up_quote.mid_price is not None and down_quote.mid_price is not None and (up_quote.mid_price - down_quote.mid_price) > spec.parity_tolerance:
        chosen_hist = down_history
    else:
        chosen_hist = up_history

    exec_dec = evaluate_ct_policy(
        spec=spec,
        decision_at=decision_at,
        market_mapping=market_mapping,
        up_quote=up_quote,
        down_quote=down_quote,
        chosen_token_history=chosen_hist,
    )

    return CTMarketDiagnostics(
        market_id=market_mapping.market_id,
        decision_at=decision_at,
        historical_yes_only=yes_only_dec,
        symmetric_price_control=symm_price_dec,
        symmetric_ct_policy=exec_dec,
    )


# ==============================================================================
# 6. Economic Scenario Accounting (Requirement 7 & 23)
# ==============================================================================

def calculate_scenario_economics(
    ask: float,
    outcome: str,  # "WIN" or "LOSS" (or "YES"/"NO" if aligned to bought outcome)
    budget_usdc: float = 1.0,
    taker_fee_rate: float = 0.002,
) -> dict[str, float]:
    """
    Pure line-item accounting function reproducing exact historical research economics.
    1. shares = budget / ask
    2. target = 1.0 if win else 0.0
    3. gross_pnl = shares * (target - ask)
    4. fee = budget * taker_fee_rate
    5. net_pnl = gross_pnl - fee
    """
    if ask <= 0.0 or not math.isfinite(ask):
        return {
            "shares": 0.0,
            "gross_pnl": 0.0,
            "fee": 0.0,
            "net_pnl": 0.0,
            "return_on_spent": 0.0,
        }

    target = 1.0 if str(outcome).strip().upper() in {"WIN", "YES", "1"} else 0.0
    shares = budget_usdc / ask
    gross_pnl = shares * (target - ask)
    fee = budget_usdc * taker_fee_rate
    net_pnl = gross_pnl - fee
    return_on_spent = (net_pnl / budget_usdc) if budget_usdc > 0 else 0.0

    return {
        "shares": shares,
        "gross_pnl": gross_pnl,
        "fee": fee,
        "net_pnl": net_pnl,
        "return_on_spent": return_on_spent,
    }


# ==============================================================================
# Helper Functions
# ==============================================================================

def _make_skip(
    spec: CTSpecification,
    decision_at: datetime,
    market_mapping: MarketTokenMapping,
    reason: str,
    time_left_sec: float,
    side: str | None = None,
    token_id: str | None = None,
    selected_ask: float | None = None,
    selected_mid: float | None = None,
    selected_bid: float | None = None,
    other_mid: float | None = None,
    outsider_margin: float | None = None,
    ct_regime: str = "NOT_EVALUATED",
    ct_features: dict[str, Any] | None = None,
    data_ids: dict[str, Any] | None = None,
) -> CTDecision:
    dec_dt = decision_at if decision_at.tzinfo is not None else decision_at.replace(tzinfo=timezone.utc)
    out_data_ids = dict(data_ids or {})
    if "opportunity_id" not in out_data_ids:
        out_data_ids["opportunity_id"] = f"{market_mapping.market_id}_{dec_dt.isoformat()}"
    if "decision_id" not in out_data_ids:
        out_data_ids["decision_id"] = f"CT:{spec.spec_id}:{market_mapping.market_id}"

    return CTDecision(
        action="SKIP",
        side=side,
        token_id=token_id,
        reason=reason,
        spec_id=spec.spec_id,
        spec_hash=spec.spec_hash,
        decision_at=decision_at,
        time_left_sec=round(time_left_sec, 2),
        market_id=market_mapping.market_id,
        asset=market_mapping.asset,
        limit_price=None,
        budget_usdc=spec.max_purchase_cost_usdc,
        selected_ask=selected_ask,
        selected_mid=selected_mid,
        selected_bid=selected_bid,
        other_mid=other_mid,
        outsider_margin=outsider_margin,
        ct_regime=ct_regime,
        ct_features=ct_features or {},
        data_ids=out_data_ids,
        is_executable=False,
    )


def _pack_ct_features(ct_res: CTRegimeResult) -> dict[str, Any]:
    return {
        "efficiency_ratio": ct_res.efficiency_ratio,
        "sign_change_freq": ct_res.sign_change_freq,
        "autocorr_lag1": ct_res.autocorr_lag1,
        "local_mean": ct_res.local_mean,
        "local_std": ct_res.local_std,
        "slope_norm": ct_res.slope_norm,
        "observations_count": ct_res.observations_count,
        "status": ct_res.status,
    }
