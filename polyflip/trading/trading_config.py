import dataclasses
from dataclasses import dataclass
from polyflip.config import settings
from polyflip.utils import parse_float_setting

def _parse_bool(val, default: bool) -> bool:
    if val is None or str(val).strip() == "":
        return default
    return str(val).lower() == "true"

def _parse_int(val, default: int) -> int:
    if val is None or str(val).strip() == "":
        return default
    try:
        return int(float(val))  # handle "300.0" if any
    except ValueError:
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
    entry_sec: int
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
    max_bet_edge: float = 0.40
    outsider_pwin_discount: float = 0.65
    max_spread_pct: float = 0.08
    combined_cost_buffer: float = 0.02

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
    if mode_raw in ("ml", "lightgbm"):
        import structlog
        structlog.get_logger(__name__).warning("legacy_trading_mode", mode=mode_raw, new_mode="combined")
        mode = "combined"
    elif mode_raw in ("favorite", "combined", "pure_favorite", "outsider"):
        mode = mode_raw
    else:
        import structlog
        structlog.get_logger(__name__).warning("unknown_trading_mode", mode=mode_raw, new_mode="combined")
        mode = "combined"

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
        entry_sec=_parse_int(raw.get("FAVORITE_MODE_ENTRY_SEC"), getattr(settings, "FAVORITE_MODE_ENTRY_SEC", 120)),
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
        combined_require_consensus=_parse_bool(raw.get("COMBINED_REQUIRE_CONSENSUS"), True),
        combined_fallback_to_logreg_on_none=_parse_bool(raw.get("COMBINED_FALLBACK_TO_LOGREG_ON_NONE"), True),
        max_bet_edge=parse_float_setting(raw, "MAX_BET_EDGE", getattr(settings, "MAX_BET_EDGE", 0.40)),
        outsider_pwin_discount=parse_float_setting(raw, "OUTSIDER_PWIN_DISCOUNT", getattr(settings, "OUTSIDER_PWIN_DISCOUNT", 0.65)),
        max_spread_pct=parse_float_setting(raw, "MAX_SPREAD_PCT", getattr(settings, "MAX_SPREAD_PCT", 0.08)),
        combined_cost_buffer=parse_float_setting(raw, "COMBINED_COST_BUFFER", 0.02),
    )
