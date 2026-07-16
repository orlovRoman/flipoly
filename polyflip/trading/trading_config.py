import dataclasses
from dataclasses import dataclass
from polyflip.config import settings

def _parse_bool(val, default: bool) -> bool:
    if val is None or str(val).strip() == "":
        return default
    return str(val).lower() == "true"

def _parse_float(val, default: float) -> float:
    if val is None or str(val).strip() == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default

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
    min_time_left: int
    max_time_left: int
    bet_size: float
    no_flip_threshold: float
    dead_zone: float
    daily_limit: float
    trade_min_price: float
    trade_max_price: float
    capital: float
    active_features_str: str
    trade_on_favorite: bool
    trade_on_flip: bool
    flip_threshold: float
    auto_dead_zone: bool
    no_max_price: float
    no_min_edge: float
    entry_sec: int
    min_edge: float
    max_bet_edge: float
    max_edge_filter: float
    favorite_threshold: float
    trade_assets: list[str]
    use_crypto_confirm: bool
    crypto_standalone: bool
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

def parse_trading_settings(raw: dict[str, str]) -> TradingConfig:
    trade_assets_str = raw.get("TRADE_ASSETS", getattr(settings, "TRADE_ASSETS", "BTC,ETH"))
    trade_assets = [a.strip() for a in trade_assets_str.split(",") if a.strip()]

    return TradingConfig(
        trading_enabled=_parse_bool(raw.get("TRADING_ENABLED"), getattr(settings, "TRADING_ENABLED", True)),
        trading_mode=raw.get("TRADING_MODE", getattr(settings, "TRADING_MODE", "ml")),
        min_time_left=_parse_int(raw.get("TRADE_MIN_TIME_LEFT_SEC"), getattr(settings, "TRADE_MIN_TIME_LEFT_SEC", 300)),
        max_time_left=_parse_int(raw.get("TRADE_MAX_TIME_LEFT_SEC"), getattr(settings, "TRADE_MAX_TIME_LEFT_SEC", 900)),
        bet_size=_parse_float(raw.get("TRADE_BET_SIZE_USDC"), getattr(settings, "TRADE_BET_SIZE_USDC", 10.0)),
        no_flip_threshold=_parse_float(raw.get("TRADE_NO_FLIP_THRESHOLD"), getattr(settings, "TRADE_NO_FLIP_THRESHOLD", 0.55)),
        dead_zone=_parse_float(raw.get("DEAD_ZONE_WIDTH"), getattr(settings, "DEAD_ZONE_WIDTH", 0.05)),
        daily_limit=_parse_float(raw.get("DAILY_LOSS_LIMIT_USDC"), getattr(settings, "DAILY_LOSS_LIMIT_USDC", -100.0)),
        trade_min_price=_parse_float(raw.get("TRADE_MIN_PRICE"), getattr(settings, "TRADE_MIN_PRICE", 0.05)),
        trade_max_price=_parse_float(raw.get("TRADE_MAX_PRICE"), getattr(settings, "TRADE_MAX_PRICE", 0.95)),
        capital=_parse_float(raw.get("INITIAL_CAPITAL"), getattr(settings, "INITIAL_CAPITAL", 100.0)),
        active_features_str=raw.get("ACTIVE_FEATURES", getattr(settings, "ACTIVE_FEATURES", "")),
        trade_on_favorite=_parse_bool(raw.get("TRADE_ON_FAVORITE"), getattr(settings, "TRADE_ON_FAVORITE", True)),
        trade_on_flip=_parse_bool(raw.get("TRADE_ON_FLIP"), getattr(settings, "TRADE_ON_FLIP", False)),
        flip_threshold=_parse_float(raw.get("FLIP_THRESHOLD"), getattr(settings, "FLIP_THRESHOLD", 0.60)),
        auto_dead_zone=_parse_bool(raw.get("AUTO_DEAD_ZONE"), getattr(settings, "AUTO_DEAD_ZONE", False)),
        no_max_price=_parse_float(raw.get("OUTSIDER_MAX_PRICE"), getattr(settings, "OUTSIDER_MAX_PRICE", 0.40)),
        no_min_edge=_parse_float(raw.get("NO_MIN_EDGE"), getattr(settings, "NO_MIN_EDGE", 0.03)),
        entry_sec=_parse_int(raw.get("FAVORITE_MODE_ENTRY_SEC"), getattr(settings, "FAVORITE_MODE_ENTRY_SEC", 120)),
        min_edge=_parse_float(raw.get("MIN_EDGE"), getattr(settings, "MIN_EDGE", 0.05)),
        max_bet_edge=_parse_float(raw.get("MAX_BET_EDGE"), getattr(settings, "MAX_BET_EDGE", 0.30)),
        max_edge_filter=_parse_float(raw.get("MAX_EDGE_FILTER"), getattr(settings, "MAX_EDGE_FILTER", 0.99)),
        favorite_threshold=_parse_float(raw.get("FAVORITE_THRESHOLD"), getattr(settings, "FAVORITE_THRESHOLD", 0.70)),
        trade_assets=trade_assets,
        use_crypto_confirm=_parse_bool(raw.get("USE_CRYPTO_CONFIRM"), getattr(settings, "USE_CRYPTO_CONFIRM", False)),
        crypto_standalone=_parse_bool(raw.get("CRYPTO_STANDALONE"), getattr(settings, "CRYPTO_STANDALONE", False)),
        bet_sizing_mode=raw.get("BET_SIZING_MODE", getattr(settings, "BET_SIZING_MODE", "fixed")),
        max_bet_size_usdc=_parse_float(raw.get("MAX_BET_SIZE_USDC"), getattr(settings, "MAX_BET_SIZE_USDC", 50.0)),
        favorite_min_price=_parse_float(raw.get("FAVORITE_MIN_PRICE"), getattr(settings, "FAVORITE_MIN_PRICE", 0.55)),
        favorite_max_price=_parse_float(raw.get("FAVORITE_MAX_PRICE"), getattr(settings, "FAVORITE_MAX_PRICE", 0.95)),
        favorite_min_edge=_parse_float(raw.get("FAVORITE_MIN_EDGE"), getattr(settings, "FAVORITE_MIN_EDGE", 0.02)),
        outsider_max_price=_parse_float(raw.get("OUTSIDER_MAX_PRICE"), getattr(settings, "OUTSIDER_MAX_PRICE", 0.40)),
        liquidity_fraction=_parse_float(raw.get("LIQUIDITY_FRACTION"), getattr(settings, "LIQUIDITY_FRACTION", 0.1)),
        bypass_bet_size_check=_parse_bool(raw.get("BYPASS_BET_SIZE_CHECK"), getattr(settings, "BYPASS_BET_SIZE_CHECK", False)),
        stop_loss_enabled=_parse_bool(raw.get("STOP_LOSS_ENABLED"), getattr(settings, "STOP_LOSS_ENABLED", False)),
        take_profit_enabled=_parse_bool(raw.get("TAKE_PROFIT_ENABLED"), getattr(settings, "TAKE_PROFIT_ENABLED", False)),
        take_profit_multiplier=_parse_float(raw.get("TAKE_PROFIT_MULTIPLIER"), getattr(settings, "TAKE_PROFIT_MULTIPLIER", 2.0)),
        max_price_drift=_parse_float(raw.get("MAX_PRICE_DRIFT"), getattr(settings, "MAX_PRICE_DRIFT", 0.03)),
        stop_loss_pct_favorite=_parse_float(raw.get("STOP_LOSS_PCT_FAVORITE"), getattr(settings, "STOP_LOSS_PCT_FAVORITE", 40.0)),
        stop_loss_pct_outsider=_parse_float(raw.get("STOP_LOSS_PCT_OUTSIDER"), getattr(settings, "STOP_LOSS_PCT_OUTSIDER", 60.0)),
    )
