import dataclasses
from dataclasses import dataclass
from typing import Optional, Any
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
import structlog
from polyflip.db.models import LiveMarket, TradeHistory
from polyflip.trading.trading_config import TradingConfig
from polyflip.trading.decision_logic import TradeDecision, MarketSignal, decide_favorite, decide_crypto_trend, decide_ml_trend, decide_outsider
from polyflip.trading.ml_inference import build_inference_dataframe, run_model_inference
from polyflip.crypto.predictor import MIN_CANDLES_REQUIRED
from polyflip.crypto.candle_repository import get_recent_candles

logger = structlog.get_logger(__name__)

@dataclass
class DecisionResult:
    decision_obj: Optional[TradeDecision]
    p_flip: float
    model_ver: Optional[int]
    edge: Optional[float]
    skip_reason: Optional[str]

async def decide_favorite_mode(
    market: LiveMarket,
    cfg: TradingConfig,
    asset_min_edge: float,
    asset_max_price: float,
    start_time: datetime,
    time_left_sec: float,
) -> DecisionResult:
    if market.current_yes_price == 0.5:
        logger.info("favorite_mode_skip_no_favorite", market_id=market.market_id)
        return DecisionResult(None, 0.0, None, None, "Pure Favorite: no clear favorite (price == 0.5)")

    signal = MarketSignal(
        asset=market.asset,
        mid_price=market.current_yes_price,
        spread=market.current_spread or 0.01,
        volume_5min=market.volume_5min or 0.0,
        price_velocity=market.price_velocity or 0.0,
        hour_of_day=start_time.hour,
        time_left_min=time_left_sec / 60.0
    )
    
    local_fav_config = {
        "FAVORITE_THRESHOLD": str(cfg.favorite_threshold),
        "FAVORITE_MIN_EDGE": str(cfg.favorite_min_edge),
        "DEAD_ZONE_WIDTH": str(cfg.dead_zone),
        "FAVORITE_MIN_PRICE": str(cfg.favorite_min_price),
        "FAVORITE_MAX_PRICE": str(cfg.favorite_max_price),
        "TRADE_BET_SIZE_USDC": str(cfg.bet_size),
        "MAX_BET_SIZE_USDC": str(cfg.max_bet_size_usdc),
        "MAX_BET_EDGE": str(cfg.max_bet_edge),
        "MIN_EDGE": str(asset_min_edge),
        "TRADE_MAX_PRICE": str(asset_max_price),
        "LIQUIDITY_FRACTION": str(cfg.liquidity_fraction),
        "BET_SIZING_MODE": str(cfg.bet_sizing_mode),
    }
    
    decision_obj = decide_favorite(signal, local_fav_config)
    if not cfg.trade_on_favorite:
        decision_obj = dataclasses.replace(decision_obj, action="SKIP", reason="Favorite trades disabled (TRADE_ON_FAVORITE=False)")
    
    return DecisionResult(
        decision_obj=decision_obj,
        p_flip=0.0,
        model_ver=None,
        edge=decision_obj.edge,
        skip_reason=decision_obj.reason if decision_obj.action == "SKIP" else None
    )

async def decide_ml_mode(
    db_session: AsyncSession,
    api_client: Any,
    market: LiveMarket,
    cfg: TradingConfig,
    models_cache: Any,
    crypto_predictor: Any,
    start_time: datetime,
    time_left_sec: float,
    existing_skipped: Optional[TradeHistory],
) -> DecisionResult:
    fresh_yes_prices = await api_client.get_market_prices(market.yes_token_id)
    if not fresh_yes_prices or "current_yes_price" not in fresh_yes_prices:
        error_msg = fresh_yes_prices.get("error", "No fresh YES prices from API") if fresh_yes_prices else "No fresh YES prices from API"
        return DecisionResult(None, 0.0, None, None, error_msg)

    fresh_yes_price = fresh_yes_prices["current_yes_price"]
    fresh_spread = fresh_yes_prices.get("current_spread", market.current_spread)
    
    asset_upper = market.asset.upper()
    model = models_cache.models.get(asset_upper)
    
    if not model:
        from polyflip.trading.ml_inference import populate_models_cache
        await populate_models_cache(db_session)
        model = models_cache.models.get(asset_upper)

    model_ver = models_cache.versions.get(asset_upper)
    active_features = models_cache.features.get(asset_upper, [])
    
    if not model:
        return DecisionResult(None, 0.0, None, None, f"No active model found for {asset_upper}")
        
    df = build_inference_dataframe(
        market=market,
        history_snaps=[], 
        fresh_yes_price=fresh_yes_price,
        fresh_spread=fresh_spread,
        global_max=market.current_yes_price,
        start_time=start_time,
        time_left_sec=time_left_sec,
    )
    
    p_flip = float(run_model_inference(df, model, active_features))
    
    signal = MarketSignal(
        asset=market.asset,
        mid_price=fresh_yes_price,
        spread=fresh_spread or 0.01,
        volume_5min=market.volume_5min or 0.0,
        price_velocity=market.price_velocity or 0.0,
        hour_of_day=start_time.hour,
        time_left_min=time_left_sec / 60.0
    )

    local_config = {
        "DEAD_ZONE_WIDTH": str(cfg.dead_zone),
        "AUTO_DEAD_ZONE": str(cfg.auto_dead_zone),
        "TRADE_ON_FAVORITE": str(cfg.trade_on_favorite),
        "TRADE_ON_FLIP": str(cfg.trade_on_flip),
        "FLIP_THRESHOLD": str(cfg.flip_threshold),
        "OUTSIDER_MAX_PRICE": "0.49",
        "TRADE_BET_SIZE_USDC": str(cfg.bet_size),
        "MAX_BET_SIZE_USDC": str(cfg.max_bet_size_usdc),
        "MAX_BET_EDGE": str(cfg.max_bet_edge),
        "MIN_EDGE": str(cfg.min_edge),
        "TRADE_MAX_PRICE": str(cfg.trade_max_price),
        "BET_SIZING_MODE": str(cfg.bet_sizing_mode),
    }

    if cfg.trade_on_favorite:
        decision_obj = decide_ml_trend(signal, p_flip, local_config)
    else:
        decision_obj = TradeDecision(action="SKIP", buy_price=0.0, bet_size_usdc=0.0, strategy_type="ML_TREND", reason="Favorite trades disabled (TRADE_ON_FAVORITE=False)", edge=0.0)

    if decision_obj.action == "SKIP" and cfg.trade_on_flip:
        decision_obj = decide_outsider(signal, p_flip, local_config)

    # Confirm Gate
    use_crypto_confirm = getattr(cfg, 'use_crypto_confirm', False)
    if decision_obj.action != "SKIP" and use_crypto_confirm and crypto_predictor:
        binance_symbol = "BTCUSDT" if market.asset.upper() == "BTC" else "ETHUSDT"
        await crypto_predictor.load(db_session, binance_symbol)
        
        model_interval = crypto_predictor.get_interval(binance_symbol)
        candles = await get_recent_candles(db_session, binance_symbol, interval=model_interval, limit=MIN_CANDLES_REQUIRED)
        crypto_sig = crypto_predictor.predict(candles, binance_symbol)
        
        if not crypto_sig.features_ok:
            decision_obj = dataclasses.replace(
                decision_obj, action="SKIP", reason="Crypto confirm: features invalid"
            )
        else:
            market_direction = "UP" if decision_obj.action == "BUY_YES" else "DOWN"
            if crypto_sig.direction != market_direction:
                decision_obj = dataclasses.replace(
                    decision_obj, action="SKIP", reason=f"Crypto confirm veto: direction is {crypto_sig.direction} vs market {market_direction}"
                )

    return DecisionResult(
        decision_obj=decision_obj,
        p_flip=p_flip,
        model_ver=model_ver,
        edge=decision_obj.edge if decision_obj else None,
        skip_reason=decision_obj.reason if decision_obj and decision_obj.action == "SKIP" else None
    )

async def decide_crypto_mode(
    db_session: AsyncSession,
    api_client: Any,
    market: LiveMarket,
    cfg: TradingConfig,
    raw_settings: dict,
    crypto_predictor: Any,
    start_time: datetime,
    time_left_sec: float
) -> DecisionResult:
    binance_symbol = "BTCUSDT" if market.asset.upper() == "BTC" else "ETHUSDT"
    
    await crypto_predictor.load(db_session, binance_symbol)
    
    model_interval = crypto_predictor.get_interval(binance_symbol)
    candles = await get_recent_candles(db_session, binance_symbol, interval=model_interval, limit=MIN_CANDLES_REQUIRED)
    crypto_sig = crypto_predictor.predict(candles, binance_symbol)
    
    if not crypto_sig.features_ok:
        return DecisionResult(None, 0.0, crypto_sig.model_version, None, "Invalid crypto features")

    fresh_yes_prices = await api_client.get_market_prices(market.yes_token_id)
    if not fresh_yes_prices or "current_yes_price" not in fresh_yes_prices:
        error_detail = fresh_yes_prices.get("error", "unknown") if fresh_yes_prices else "None returned"
        return DecisionResult(None, 0.0, crypto_sig.model_version, None, f"No fresh yes price: {error_detail}")
        
    fresh_yes_price = fresh_yes_prices["current_yes_price"]

    decision_obj = decide_crypto_trend(crypto_sig, fresh_yes_price, market.volume_5min or 0.0, raw_settings)
    if not cfg.trade_on_favorite:
        decision_obj = dataclasses.replace(decision_obj, action="SKIP", reason="Favorite trades disabled (TRADE_ON_FAVORITE=False)")
    
    return DecisionResult(
        decision_obj=decision_obj,
        p_flip=0.0,
        model_ver=crypto_sig.model_version if crypto_sig else None,
        edge=decision_obj.edge if decision_obj else None,
        skip_reason=decision_obj.reason if decision_obj and decision_obj.action == "SKIP" else None
    )
