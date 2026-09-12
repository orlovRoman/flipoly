"""Effective trading-policy contract exposed to the dashboard and telemetry.

The settings table contains historical knobs from several decision paths.  A
raw dump therefore cannot answer which values can affect the next decision.
This module parses the same values as the engine, records the active subset,
marks legacy keys explicitly, and produces a stable hash that can be attached
to a decision or trade.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from polyflip.trading.trading_config import TradingConfig, parse_trading_settings


SCHEMA_VERSION = 1

# These values are accepted for backwards compatibility but do not affect the
# current COMBINED evaluator.  Keeping them visible prevents a dashboard
# operator from mistaking an editable field for an active control.
LEGACY_KEYS = (
    "DEAD_ZONE_WIDTH",
    "ENTRY_STRATEGY",
    "NO_MIN_EDGE",
    "OUTSIDER_PWIN_DISCOUNT",
    "TRADE_FLIP_THRESHOLD",
)


def _json_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _active_values(raw: Mapping[str, Any], cfg: TradingConfig) -> dict[str, Any]:
    """Return only fields read by the current decision/execution path."""
    return {
        "routing": {
            "trading_mode": cfg.trading_mode,
            "per_asset_trading_mode": {
                str(key)[len("TRADING_MODE_"):].upper(): value
                for key, value in raw.items()
                if str(key).startswith("TRADING_MODE_") and value not in (None, "")
            },
            "trading_policy_mode": cfg.trading_policy_mode,
            "lightgbm_decision_mode": cfg.lightgbm_decision_mode,
            "trade_assets": list(cfg.trade_assets),
            "trade_on_favorite": cfg.trade_on_favorite,
            "trade_on_flip": cfg.trade_on_flip,
            "require_reversion_regime": cfg.require_reversion_regime,
        },
        "model_gates": {
            "min_direction_prob": cfg.min_direction_prob,
            "min_win_prob": cfg.min_win_prob,
            "combined_require_consensus": cfg.combined_require_consensus,
            "combined_fallback_to_logreg_on_none": cfg.combined_fallback_to_logreg_on_none,
            "combined_lgbm_unavailable_policy": cfg.lgbm_unavailable_policy,
            "combined_logreg_abstain_band": cfg.combined_logreg_abstain_band,
            "combined_dir_discount_weight": cfg.combined_dir_discount_weight,
            "combined_dir_strong_threshold": cfg.combined_dir_strong_threshold,
            "invert_lgbm_signal": cfg.invert_lgbm_signal,
            "enable_ece_correction": cfg.enable_ece_correction,
        },
        "entry": {
            "flip_threshold": cfg.flip_threshold,
            "favorite_threshold": cfg.favorite_threshold,
            "favorite_min_edge": cfg.favorite_min_edge,
            "outsider_min_edge": cfg.outs_min_edge,
            "favorite_price": [cfg.favorite_min_price, cfg.favorite_max_price],
            "outsider_max_price": cfg.outsider_max_price,
            "max_spread_pct": cfg.max_spread_pct,
            "combined_cost_buffer_usdc": cfg.combined_cost_buffer,
        },
        "time": {
            "favorite_sec": [cfg.favor_min_time_left, cfg.favor_max_time_left],
            "outsider_sec": [cfg.outs_min_time_left, cfg.outs_max_time_left],
        },
        "sizing": {
            "mode": cfg.bet_sizing_mode,
            "bet_size_usdc": cfg.bet_size,
            "max_bet_size_usdc": cfg.max_bet_size_usdc,
            "liquidity_fraction": cfg.liquidity_fraction,
        },
        "execution": {
            "max_price_drift_usdc_per_share": cfg.max_price_drift,
            "fee_rate_legacy": cfg.fee_rate,
            "slippage_rate_legacy": cfg.slippage_rate,
            "paper_execution_profile": raw.get("PAPER_EXECUTION_PROFILE"),
            "paper_fee_model": raw.get("PAPER_FEE_MODEL"),
            "paper_fee_rate": raw.get("PAPER_FEE_RATE"),
            "paper_fee_exponent": raw.get("PAPER_FEE_EXPONENT"),
            "live_order_mode": raw.get("LIVE_ORDER_MODE"),
            "live_gtc_ttl_seconds": raw.get("LIVE_GTC_TTL_SECONDS"),
            "maker_reprice_on_cross": raw.get("LIVE_MAKER_REPRICE_ON_CROSS"),
            "maker_reprice_max_retries": raw.get("LIVE_MAKER_REPRICE_MAX_RETRIES"),
        },
        "weighted_policy": {
            "policy_id": cfg.weighted_policy_id,
            "market_weight": cfg.weighted_market_weight,
            "logreg_weight": cfg.weighted_logreg_weight,
            "lgbm_weight": cfg.weighted_lgbm_weight,
            "fee_rate": cfg.weighted_fee_rate,
            "fee_exponent": cfg.weighted_fee_exponent,
            "slippage_rate": cfg.weighted_slippage_rate,
            "execution_role": cfg.weighted_execution_role,
            "min_net_ev_favorite": cfg.weighted_min_net_ev_favorite,
            "min_net_ev_outsider": cfg.weighted_min_net_ev_outsider,
            "sizing_mode": cfg.weighted_sizing_mode,
        },
    }


def effective_policy_snapshot(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Build a deterministic, human-readable policy snapshot.

    The hash excludes itself and is stable across dictionary insertion order.
    ``raw`` is retained only for legacy-key status and the small set of
    execution fields not represented by ``TradingConfig``.
    """
    normalized = {str(key): _json_value(value) for key, value in raw.items()}
    cfg = parse_trading_settings(normalized)
    warnings: list[dict[str, str]] = []
    try:
        raw_direction_floor = float(normalized.get("MIN_DIRECTION_PROB"))
        if raw_direction_floor > 1.0:
            raw_direction_floor /= 100.0
        if raw_direction_floor < 0.5:
            warnings.append(
                {
                    "code": "DIRECTION_FLOOR_CLAMPED",
                    "severity": "WARNING",
                    "message": "MIN_DIRECTION_PROB below 50% is replaced by the parser safety floor.",
                }
            )
    except (TypeError, ValueError):
        pass
    if cfg.combined_require_consensus and cfg.combined_fallback_to_logreg_on_none:
        warnings.append(
            {
                "code": "FALLBACK_INACTIVE_UNDER_CONSENSUS",
                "severity": "INFO",
                "message": "LogReg fallback is ignored while consensus is required.",
            }
        )
    try:
        raw_strong_threshold = float(normalized.get("COMBINED_DIR_STRONG_THRESHOLD"))
        if raw_strong_threshold > 1.0:
            raw_strong_threshold /= 100.0
        if raw_strong_threshold < cfg.min_direction_prob:
            warnings.append(
                {
                    "code": "STRONG_THRESHOLD_CLAMPED",
                    "severity": "WARNING",
                    "message": "COMBINED_DIR_STRONG_THRESHOLD below MIN_DIRECTION_PROB is replaced by the direction floor.",
                }
            )
    except (TypeError, ValueError):
        pass
    if cfg.combined_dir_strong_threshold < cfg.min_direction_prob:
        warnings.append(
            {
                "code": "STRONG_THRESHOLD_CLAMPED",
                "severity": "WARNING",
                "message": "Strong-direction threshold was clamped to the direction floor.",
            }
        )
    if str(cfg.trading_mode).lower() == "combined":
        warnings.append(
            {
                "code": "LEGACY_FIELDS_IGNORED",
                "severity": "INFO",
                "message": "Legacy fields are shown separately and do not alter COMBINED decisions.",
            }
        )
    warnings.sort(key=lambda item: (item["code"], item["message"]))
    legacy = {
        key: {
            "value": normalized.get(key),
            "status": "LEGACY",
        }
        for key in LEGACY_KEYS
        if key in normalized
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "active": _active_values(normalized, cfg),
        "legacy": legacy,
        "warnings": warnings,
    }
    payload["policy_hash"] = _canonical_hash(payload)
    return payload
