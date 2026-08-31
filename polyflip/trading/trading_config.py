import dataclasses
from dataclasses import dataclass
from math import isfinite

import structlog

from polyflip.config import settings
from polyflip.crypto.market_regime import MIN_HISTORY_CANDLES
from polyflip.utils import parse_float_setting

logger = structlog.get_logger(__name__)


def _parse_bool(val, default: bool) -> bool:
    if val is None or str(val).strip() == "":
        return default
    return str(val).lower() == "true"

def _parse_int(val, default: int) -> int:
    if val is None or str(val).strip() == "":
        return default
    try:
        return int(float(val))  # handle "300.0" if any
    except (TypeError, ValueError, OverflowError):
        return default


@dataclass(frozen=True)
class TradingConfig:
    trading_enabled: bool
    trading_mode: str
    favor_min_time_left: int
    favor_max_time_left: int
    outs_min_time_left: int
    outs_max_time_left: int
    bet_size: float
    dead_zone: float
    daily_limit: float
    trade_min_price: float
    trade_max_price: float
    capital: float
    active_features_str: str
    trade_on_favorite: bool
    trade_on_flip: bool
    flip_threshold: float
    outs_min_edge: float
    favorite_threshold: float
    trade_assets: list[str]
    bet_sizing_mode: str
    max_bet_size_usdc: float
    favorite_min_price: float
    favorite_max_price: float
    favorite_min_edge: float
    outsider_max_price: float
    liquidity_fraction: float
    bypass_bet_size_check: bool
    stop_loss_enabled: bool
    take_profit_enabled: bool
    take_profit_multiplier: float
    max_price_drift: float
    stop_loss_pct_favorite: float
    stop_loss_pct_outsider: float
    fee_rate: float
    slippage_rate: float
    max_exposure_pct: float
    min_direction_prob: float
    min_win_prob: float
    combined_dir_discount_weight: float = 0.0
    combined_dir_strong_threshold: float = 0.65
    combined_require_consensus: bool = True
    combined_fallback_to_logreg_on_none: bool = True
    combined_logreg_abstain_band: float = 0.05
    invert_lgbm_signal: bool = False
    max_bet_edge: float = 0.40
    outsider_pwin_discount: float = 0.65
    max_spread_pct: float = 0.08
    combined_cost_buffer: float = 0.02
    lgbm_unavailable_policy: str = "SKIP"
    lightgbm_decision_mode: str = "SHADOW"
    enable_ece_correction: bool = True
    # Weighted trading policy rollout.  Keep LEGACY as the safe default so
    # adding these settings cannot silently change an existing deployment.
    trading_policy_mode: str = "LEGACY"
    weighted_market_weight: float = 0.90
    weighted_logreg_weight: float = 0.05
    weighted_lgbm_weight: float = 0.05
    weighted_mrf_beta: float = 0.0
    weighted_fee_rate: float = 0.07
    weighted_fee_exponent: float = 1.0
    weighted_slippage_rate: float = 0.005
    weighted_execution_role: str = "TAKER"
    # ── Market Regime Filter (MRF-T09) ──────────────────────
    mrf_mode: str = "OFF"                 # OFF|SHADOW|ACTIVE
    mrf_version: int = 1
    mrf_min_history: int = 97             # candles
    mrf_outsider_trend_multiplier: float = 0.0
    mrf_unknown_multiplier: float = 0.8
    mrf_breadth_threshold: float = 0.65
    mrf_efficiency_threshold: float = 0.4
    mrf_veto_threshold: float = 0.15
    mrf_edge_override_margin: float = 0.05
    mrf_asset_weight: float = 0.70
    mrf_global_weight: float = 0.30

    def get_min_edge(self, is_outsider: bool) -> float:
        """Единый источник правды для минимального Edge."""
        return self.outs_min_edge if is_outsider else self.favorite_min_edge

    def is_time_valid(self, time_left_sec: float, is_outsider: bool) -> tuple[bool, str]:
        if time_left_sec <= 0:
            return False, f"time_left={time_left_sec:.0f}s — market closed or unknown"
        lo = self.outs_min_time_left if is_outsider else self.favor_min_time_left
        hi = self.outs_max_time_left if is_outsider else self.favor_max_time_left
        if not (lo <= time_left_sec <= hi):
            role = "outsider" if is_outsider else "favorite"
            return False, f"{role}: time_left={time_left_sec:.0f}s out of [{lo}, {hi}]"
        return True, "OK"

    def is_price_valid(self, price: float, is_outsider: bool) -> tuple[bool, str]:
        """Единый источник правды для фильтрации по цене."""
        if not (self.trade_min_price <= price <= self.trade_max_price):
            return False, f"Price {price:.3f} out of global bounds [{self.trade_min_price}, {self.trade_max_price}]"
        
        if is_outsider:
            if price > self.outsider_max_price:
                return False, f"Outsider price {price:.3f} > outsider max {self.outsider_max_price}"
        else:
            if not (self.favorite_min_price <= price <= self.favorite_max_price):
                return False, f"Favorite price {price:.3f} out of bounds [{self.favorite_min_price}, {self.favorite_max_price}]"
        
        return True, "OK"

def parse_trading_settings(raw: dict[str, str]) -> TradingConfig:
    trade_assets_str = raw.get("TRADE_ASSETS", getattr(settings, "TRADE_ASSETS", "BTC,ETH"))
    trade_assets = [a.strip() for a in trade_assets_str.split(",") if a.strip()]

    mode_raw = raw.get("TRADING_MODE", getattr(settings, "TRADING_MODE", "combined")).lower()
    if mode_raw == "combined":
        mode = mode_raw
    elif mode_raw in ("ml", "lightgbm", "favorite", "pure_favorite", "outsider"):
        import structlog
        structlog.get_logger(__name__).warning("legacy_trading_mode", mode=mode_raw, new_mode="combined")
        mode = "combined"
    else:
        import structlog
        structlog.get_logger(__name__).warning("unknown_trading_mode", mode=mode_raw, new_mode="combined")
        mode = "combined"

    mrf_version = _parse_int(
        raw.get("MARKET_REGIME_FILTER_VERSION"),
        getattr(settings, "MARKET_REGIME_FILTER_VERSION", 1),
    )
    if mrf_version not in (1, 2, 3):
        logger.warning(
            "invalid_market_regime_filter_version",
            value=mrf_version,
            fallback=1,
        )
        # Settings are read on every scheduler cycle. A stale or manually
        # edited DB value must not take down the trading loop.
        mrf_version = 1

    mrf_veto_threshold = parse_float_setting(
        raw, "MARKET_REGIME_VETO_THRESHOLD", 0.15,
    )
    mrf_edge_override_margin = parse_float_setting(
        raw, "MARKET_REGIME_EDGE_OVERRIDE_MARGIN", 0.05,
    )
    mrf_asset_weight = parse_float_setting(
        raw, "MARKET_REGIME_ASSET_WEIGHT", 0.70,
    )
    mrf_global_weight = parse_float_setting(
        raw, "MARKET_REGIME_GLOBAL_WEIGHT", 0.30,
    )
    # Validate the complete v3 gate contract while parsing settings instead
    # of waiting for the first decision (where an exception would otherwise
    # be swallowed by the MRF wrapper). Invalid values use safe defaults.
    if (
        not all(isfinite(value) for value in (
            mrf_asset_weight,
            mrf_global_weight,
            mrf_veto_threshold,
            mrf_edge_override_margin,
        ))
        or mrf_asset_weight < 0
        or mrf_global_weight < 0
        or mrf_asset_weight + mrf_global_weight <= 0
        or not 0 <= mrf_veto_threshold <= 1
        or mrf_edge_override_margin < 0
    ):
        logger.warning(
            "invalid_market_regime_filter_config",
            asset_weight=mrf_asset_weight,
            global_weight=mrf_global_weight,
            veto_threshold=mrf_veto_threshold,
            edge_override_margin=mrf_edge_override_margin,
            fallback="asset_weight=0.70,global_weight=0.30,veto_threshold=0.15,edge_override_margin=0.05",
        )
        mrf_asset_weight = 0.70
        mrf_global_weight = 0.30
        mrf_veto_threshold = 0.15
        mrf_edge_override_margin = 0.05

    trading_policy_mode = str(
        raw.get("TRADING_POLICY_MODE", getattr(settings, "TRADING_POLICY_MODE", "LEGACY"))
        or "LEGACY"
    ).strip().upper()
    if trading_policy_mode not in {"LEGACY", "WEIGHTED_SHADOW", "WEIGHTED_ACTIVE"}:
        logger.warning(
            "invalid_trading_policy_mode",
            value=trading_policy_mode,
            fallback="LEGACY",
        )
        trading_policy_mode = "LEGACY"

    weighted_execution_role = str(
        raw.get("WEIGHTED_EXECUTION_ROLE", getattr(settings, "WEIGHTED_EXECUTION_ROLE", "TAKER"))
        or "TAKER"
    ).strip().upper()
    if weighted_execution_role not in {"MAKER", "TAKER"}:
        logger.warning(
            "invalid_weighted_execution_role",
            value=weighted_execution_role,
            fallback="TAKER",
        )
        weighted_execution_role = "TAKER"

    weighted_mrf_beta = parse_float_setting(
        raw, "WEIGHTED_MRF_BETA", getattr(settings, "WEIGHTED_MRF_BETA", 0.0)
    )
    # Keep the regime log-odds adjustment bounded.  A beta of +/-2 already
    # changes odds by roughly 7.4x; larger values would let an uncalibrated
    # regime classifier overpower the market prior.
    if not isfinite(weighted_mrf_beta) or not -2.0 <= weighted_mrf_beta <= 2.0:
        logger.warning(
            "invalid_weighted_mrf_beta",
            value=weighted_mrf_beta,
            fallback=0.0,
        )
        weighted_mrf_beta = 0.0

    weighted_fee_exponent = parse_float_setting(
        raw, "WEIGHTED_FEE_EXPONENT",
        getattr(settings, "WEIGHTED_FEE_EXPONENT", 1.0),
    )
    if not isfinite(weighted_fee_exponent) or not 0.0 <= weighted_fee_exponent <= 16.0:
        logger.warning(
            "invalid_weighted_fee_exponent",
            value=weighted_fee_exponent,
            fallback=1.0,
        )
        weighted_fee_exponent = 1.0

    return TradingConfig(
        trading_enabled=_parse_bool(raw.get("TRADING_ENABLED"), getattr(settings, "TRADING_ENABLED", True)),
        trading_mode=mode,
        favor_min_time_left=_parse_int(raw.get("FAVOR_MIN_TIME_LEFT_SEC"), getattr(settings, "FAVOR_MIN_TIME_LEFT_SEC", 60)),
        favor_max_time_left=_parse_int(raw.get("FAVOR_MAX_TIME_LEFT_SEC"), getattr(settings, "FAVOR_MAX_TIME_LEFT_SEC", 600)),
        outs_min_time_left=_parse_int(raw.get("OUTS_MIN_TIME_LEFT_SEC"), getattr(settings, "OUTS_MIN_TIME_LEFT_SEC", 30)),
        outs_max_time_left=_parse_int(raw.get("OUTS_MAX_TIME_LEFT_SEC"), getattr(settings, "OUTS_MAX_TIME_LEFT_SEC", 300)),
        bet_size=parse_float_setting(raw, "TRADE_BET_SIZE_USDC", getattr(settings, "TRADE_BET_SIZE_USDC", 10.0)),
        dead_zone=parse_float_setting(raw, "DEAD_ZONE_WIDTH", getattr(settings, "DEAD_ZONE_WIDTH", 0.05)),
        daily_limit=parse_float_setting(raw, "DAILY_LOSS_LIMIT_USDC", getattr(settings, "DAILY_LOSS_LIMIT_USDC", -100.0)),
        trade_min_price=parse_float_setting(raw, "TRADE_MIN_PRICE", getattr(settings, "TRADE_MIN_PRICE", 0.05)),
        trade_max_price=parse_float_setting(raw, "TRADE_MAX_PRICE", getattr(settings, "TRADE_MAX_PRICE", 0.95)),
        capital=parse_float_setting(raw, "INITIAL_CAPITAL", getattr(settings, "INITIAL_CAPITAL", 100.0)),
        active_features_str=raw.get("ACTIVE_FEATURES", getattr(settings, "ACTIVE_FEATURES", "")),
        trade_on_favorite=_parse_bool(raw.get("TRADE_ON_FAVORITE"), getattr(settings, "TRADE_ON_FAVORITE", True)),
        trade_on_flip=_parse_bool(raw.get("TRADE_ON_FLIP"), getattr(settings, "TRADE_ON_FLIP", False)),
        flip_threshold=parse_float_setting(raw, "FLIP_THRESHOLD", getattr(settings, "FLIP_THRESHOLD", 0.60)),
        outs_min_edge=parse_float_setting(raw, "OUTS_MIN_EDGE", getattr(settings, "OUTS_MIN_EDGE", 0.04)),
        favorite_threshold=parse_float_setting(raw, "FAVORITE_THRESHOLD", getattr(settings, "FAVORITE_THRESHOLD", 0.70)),
        trade_assets=trade_assets,
        bet_sizing_mode=raw.get("BET_SIZING_MODE", getattr(settings, "BET_SIZING_MODE", "fixed")),
        max_bet_size_usdc=parse_float_setting(raw, "MAX_BET_SIZE_USDC", getattr(settings, "MAX_BET_SIZE_USDC", 50.0)),
        favorite_min_price=parse_float_setting(raw, "FAVORITE_MIN_PRICE", getattr(settings, "FAVORITE_MIN_PRICE", 0.55)),
        favorite_max_price=parse_float_setting(raw, "FAVORITE_MAX_PRICE", getattr(settings, "FAVORITE_MAX_PRICE", 0.95)),
        favorite_min_edge=parse_float_setting(raw, "FAVORITE_MIN_EDGE", getattr(settings, "FAVORITE_MIN_EDGE", 0.05)),
        outsider_max_price=parse_float_setting(raw, "OUTSIDER_MAX_PRICE", getattr(settings, "OUTSIDER_MAX_PRICE", 0.40)),
        liquidity_fraction=parse_float_setting(raw, "LIQUIDITY_FRACTION", getattr(settings, "LIQUIDITY_FRACTION", 0.1)),
        bypass_bet_size_check=_parse_bool(raw.get("BYPASS_BET_SIZE_CHECK"), getattr(settings, "BYPASS_BET_SIZE_CHECK", False)),
        stop_loss_enabled=_parse_bool(raw.get("STOP_LOSS_ENABLED"), getattr(settings, "STOP_LOSS_ENABLED", False)),
        take_profit_enabled=_parse_bool(raw.get("TAKE_PROFIT_ENABLED"), getattr(settings, "TAKE_PROFIT_ENABLED", False)),
        take_profit_multiplier=parse_float_setting(raw, "TAKE_PROFIT_MULTIPLIER", getattr(settings, "TAKE_PROFIT_MULTIPLIER", 2.0)),
        max_price_drift=parse_float_setting(raw, "MAX_PRICE_DRIFT", getattr(settings, "MAX_PRICE_DRIFT", 0.03)),
        stop_loss_pct_favorite=parse_float_setting(raw, "STOP_LOSS_PCT_FAVORITE", getattr(settings, "STOP_LOSS_PCT_FAVORITE", 40.0)),
        stop_loss_pct_outsider=parse_float_setting(raw, "STOP_LOSS_PCT_OUTSIDER", getattr(settings, "STOP_LOSS_PCT_OUTSIDER", 60.0)),
        fee_rate=parse_float_setting(raw, "FEE_RATE", getattr(settings, "FEE_RATE", 0.0)),
        slippage_rate=parse_float_setting(raw, "SLIPPAGE_RATE", getattr(settings, "SLIPPAGE_RATE", 0.0)),
        max_exposure_pct=parse_float_setting(raw, "MAX_EXPOSURE_PCT", getattr(settings, "MAX_EXPOSURE_PCT", 15.0)),
        min_direction_prob=parse_float_setting(raw, "MIN_DIRECTION_PROB", getattr(settings, "MIN_DIRECTION_PROB", 0.505)),
        min_win_prob=parse_float_setting(raw, "MIN_WIN_PROB", getattr(settings, "MIN_WIN_PROB", 0.51)),
        combined_dir_discount_weight=parse_float_setting(raw, "COMBINED_DIR_DISCOUNT_WEIGHT", 0.0),
        combined_dir_strong_threshold=parse_float_setting(raw, "COMBINED_DIR_STRONG_THRESHOLD", 0.65),
        combined_require_consensus=_parse_bool(raw.get("COMBINED_REQUIRE_CONSENSUS"), getattr(settings, "COMBINED_REQUIRE_CONSENSUS", True)),
        combined_fallback_to_logreg_on_none=_parse_bool(raw.get("COMBINED_FALLBACK_TO_LOGREG_ON_NONE"), getattr(settings, "COMBINED_FALLBACK_TO_LOGREG_ON_NONE", True)),
        combined_logreg_abstain_band=parse_float_setting(raw, "COMBINED_LOGREG_ABSTAIN_BAND", 0.05),
        invert_lgbm_signal=_parse_bool(raw.get("INVERT_LGBM_SIGNAL"), False),
        max_bet_edge=parse_float_setting(raw, "MAX_BET_EDGE", getattr(settings, "MAX_BET_EDGE", 0.40)),
        outsider_pwin_discount=parse_float_setting(raw, "OUTSIDER_PWIN_DISCOUNT", getattr(settings, "OUTSIDER_PWIN_DISCOUNT", 0.65)),
        max_spread_pct=parse_float_setting(raw, "MAX_SPREAD_PCT", getattr(settings, "MAX_SPREAD_PCT", 0.08)),
        combined_cost_buffer=parse_float_setting(raw, "COMBINED_COST_BUFFER", 0.02),
        lgbm_unavailable_policy=raw.get("COMBINED_LGBM_UNAVAILABLE_POLICY", getattr(settings, "COMBINED_LGBM_UNAVAILABLE_POLICY", "SKIP")).strip().upper(),
        lightgbm_decision_mode=(
            raw.get("LIGHTGBM_DECISION_MODE", "SHADOW").strip().upper()
            if raw.get("LIGHTGBM_DECISION_MODE", "SHADOW").strip().upper() in {"OFF", "SHADOW", "ACTIVE"}
            else "SHADOW"
        ),
        enable_ece_correction=_parse_bool(raw.get("ENABLE_ECE_CORRECTION"), getattr(settings, "ENABLE_ECE_CORRECTION", True)),
        trading_policy_mode=trading_policy_mode,
        weighted_market_weight=parse_float_setting(
            raw, "WEIGHTED_MARKET_WEIGHT", getattr(settings, "WEIGHTED_MARKET_WEIGHT", 0.90)
        ),
        weighted_logreg_weight=parse_float_setting(
            raw, "WEIGHTED_LOGREG_WEIGHT", getattr(settings, "WEIGHTED_LOGREG_WEIGHT", 0.05)
        ),
        weighted_lgbm_weight=parse_float_setting(
            raw, "WEIGHTED_LGBM_WEIGHT", getattr(settings, "WEIGHTED_LGBM_WEIGHT", 0.05)
        ),
        weighted_mrf_beta=weighted_mrf_beta,
       weighted_fee_rate=parse_float_setting(
           raw, "WEIGHTED_FEE_RATE", getattr(settings, "WEIGHTED_FEE_RATE", 0.07)
       ),
        weighted_fee_exponent=weighted_fee_exponent,
        weighted_slippage_rate=parse_float_setting(
           raw, "WEIGHTED_SLIPPAGE_RATE", getattr(settings, "WEIGHTED_SLIPPAGE_RATE", 0.005)
       ),
        weighted_execution_role=weighted_execution_role,
        # ── Market Regime Filter (MRF-T09) ──────────────────────
        mrf_mode=(
            raw.get("MARKET_REGIME_FILTER_MODE", "OFF").strip().upper()
            if raw.get("MARKET_REGIME_FILTER_MODE", "OFF").strip().upper() in {"OFF", "SHADOW", "ACTIVE"}
            else "OFF"
        ),
        mrf_version=mrf_version,
        mrf_min_history=max(
            MIN_HISTORY_CANDLES,
            _parse_int(
                raw.get("MARKET_REGIME_MIN_HISTORY"),
                MIN_HISTORY_CANDLES,
            ),
        ),
        mrf_outsider_trend_multiplier=parse_float_setting(raw, "MARKET_REGIME_OUTSIDER_TREND_MULTIPLIER", 0.0),
        mrf_unknown_multiplier=parse_float_setting(raw, "MARKET_REGIME_UNKNOWN_MULTIPLIER", 0.8),
        mrf_breadth_threshold=parse_float_setting(raw, "MARKET_REGIME_BREADTH_THRESHOLD", 0.65),
        mrf_efficiency_threshold=parse_float_setting(raw, "MARKET_REGIME_EFFICIENCY_THRESHOLD", 0.4),
        mrf_veto_threshold=mrf_veto_threshold,
        mrf_edge_override_margin=mrf_edge_override_margin,
        mrf_asset_weight=mrf_asset_weight,
        mrf_global_weight=mrf_global_weight,
    )
