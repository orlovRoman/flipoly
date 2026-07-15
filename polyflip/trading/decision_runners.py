import dataclasses
from dataclasses import dataclass
from typing import Optional, Any
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
import structlog
from polyflip.db.models import LiveMarket, TradeHistory
from polyflip.trading.trading_config import TradingConfig
from polyflip.trading.decision_logic import TradeDecision, MarketSignal, decide_favorite, decide_crypto_trend
from polyflip.trading.ml_inference import ModelsCache, build_inference_dataframe, run_model_inference

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
    models_cache: ModelsCache,
    crypto_predictor: Any,
    start_time: datetime,
    time_left_sec: float,
    existing_skipped: Optional[TradeHistory],
) -> DecisionResult:
    fresh_yes_prices = await api_client.get_market_prices(market.yes_token_id)
    if not fresh_yes_prices or "error" in fresh_yes_prices:
        error_msg = fresh_yes_prices.get("error", "No fresh YES prices from API") if fresh_yes_prices else "No fresh YES prices from API"
        return DecisionResult(None, 0.0, None, None, error_msg)

    fresh_yes_price = fresh_yes_prices.get("current_yes_price", market.current_yes_price)
    fresh_spread = fresh_yes_prices.get("current_spread", market.current_spread)
    
    asset_upper = market.asset.upper()
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
    
    p_flip = run_model_inference(df, model, active_features)
    
    return DecisionResult(None, p_flip, model_ver, None, "Not implemented fully in runner yet")
