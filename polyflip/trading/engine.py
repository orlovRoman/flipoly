import os
import pickle
import dataclasses
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func

logger = structlog.get_logger(__name__)

from polyflip.config import settings
from polyflip.trading.feature_builder import MarketSignal
from polyflip.trading.decision_logic import decide_favorite, decide_ml_trend, decide_outsider, decide_crypto_trend
from polyflip.crypto.predictor import CryptoPredictor, MIN_CANDLES_REQUIRED
from polyflip.crypto.candle_repository import get_recent_candles

from polyflip.db.models import LiveMarket, ModelRegistry, RuntimeSettings, TradeHistory, SlippageLog
from polyflip.trading.trader import PolyTrader
from polyflip.collector.client import PolymarketClient
from polyflip.trading.utils import compute_dead_zone
from polyflip.trading.position_sizing import compute_edge, compute_bet_size_edge_scaled
from polyflip.models.trainer import add_derived_features
from polyflip.api.trading_dashboard import invalidate_stats_cache
from polyflip.api.dashboard import invalidate_dashboard_cache
from polyflip.constants import (
    DEAD_ZONE_WIDTH,
    DAILY_LOSS_LIMIT_USDC,
    FAVORITE_THRESHOLD,
    TRADE_CHECK_LIMIT,
    TRADING_MODE_ML,
    TRADING_MODE_FAVORITE,
    TRADING_MODE_CRYPTO,
    FAVORITE_MODE_ENTRY_SEC,
    FAVORITE_MODE_ENTRY_WINDOW_SEC,
    TRADE_ON_FLIP,
    FLIP_THRESHOLD,
    OUTSIDER_MAX_PRICE,
    NO_MIN_EDGE,
    AUTO_DEAD_ZONE,
    MAX_EDGE_SCALING,
    MAX_EDGE_FILTER,
)


_crypto_predictor: Optional[CryptoPredictor] = None

def _get_crypto_predictor() -> CryptoPredictor:
    global _crypto_predictor
    if _crypto_predictor is None:
        _crypto_predictor = CryptoPredictor()
    return _crypto_predictor


logger = structlog.get_logger(__name__)


async def save_or_update_skipped_trade(
    db_session: AsyncSession,
    market,
    reason: str,
    p_flip_val: float,
    model_version: Optional[int],
    start_time: datetime,
    existing_skipped: Optional[TradeHistory] = None,
    edge: Optional[float] = None
):
    """Сохраняет запись о пропуске сделки в БД или обновляет её причину."""
    if existing_skipped:
        if (existing_skipped.error_msg != reason or 
            existing_skipped.predicted_flip_prob != p_flip_val or 
            existing_skipped.edge != edge):
            existing_skipped.error_msg = reason
            existing_skipped.predicted_flip_prob = p_flip_val
            existing_skipped.model_version = model_version
            existing_skipped.edge = edge
            existing_skipped.updated_at = start_time
    else:
        history = TradeHistory(
            market_id=market.market_id,
            asset=market.asset,
            outcome_bought="NONE",
            amount_usdc=0.0,
            executed_price=0.0,
            predicted_flip_prob=p_flip_val,
            active_features="",
            model_version=model_version,
            status="SKIPPED",
            error_msg=reason,
            mode="LIVE" if bool(os.getenv("POLYGON_PRIVATE_KEY") and os.getenv("POLYGON_ADDRESS")) else "PAPER",
            edge=edge,
            created_at=start_time
        )
        db_session.add(history)



async def trade_worker_cycle(db_session: AsyncSession, trader: PolyTrader, api_client: PolymarketClient):
    """
    Фоновый процесс торгового движка.
    """
    start_time = datetime.now(timezone.utc)
    
    # 1. Загружаем базовые торговые настройки
    settings_keys = [
        "TRADING_ENABLED", 
        "TRADE_MIN_TIME_LEFT_SEC",
        "TRADE_MAX_TIME_LEFT_SEC",
        "TRADE_BET_SIZE_USDC",
        "TRADE_NO_FLIP_THRESHOLD",
        "DEAD_ZONE_WIDTH",
        "DAILY_LOSS_LIMIT_USDC",
        "ACTIVE_FEATURES",
        "TRADE_MIN_PRICE",
        "TRADE_MAX_PRICE",
        "TRADE_ASSETS",
        "TRADING_MODE",
        "FAVORITE_MODE_ENTRY_SEC",
        "MIN_EDGE",
        "MAX_BET_EDGE",        # потолок масштабирования ставки
        "MAX_EDGE_FILTER",     # фильтр аномального edge (SKIP если превышен)
        "FAVORITE_THRESHOLD",
        "TRADE_ON_FLIP",
        "FLIP_THRESHOLD",
        "NO_MIN_EDGE",
        "AUTO_DEAD_ZONE",
        # AUTO_DEAD_ZONE_WIDTH удалён — вместо него используется DEAD_ZONE_WIDTH
        "MAX_PRICE_DRIFT",
        "BET_SIZING_MODE",
        "MAX_BET_SIZE_USDC",
        "FAVORITE_MIN_PRICE",
        "FAVORITE_MAX_PRICE",
        "FAVORITE_MIN_EDGE",
        "OUTSIDER_MAX_PRICE",
        "LIQUIDITY_FRACTION",
        "BYPASS_BET_SIZE_CHECK",
        "USE_CRYPTO_CONFIRM",
        "CRYPTO_STANDALONE",
        "CRYPTO_MIN_EDGE",
        # NOTE: vol-режимные пороги (CRYPTO_THRESHOLD_BTCUSDT_low_vol и т.д.)
        # загружаются predictor.load() напрямую из RuntimeSettings — не передавать через settings_db.
    ]
    stmt = select(RuntimeSettings).where(RuntimeSettings.key.in_(settings_keys))
    result = await db_session.execute(stmt)
    settings_db = {s.key: s.value for s in result.scalars().all()}
    
    # Сначала определяем список торгуемых активов
    trade_assets_str = settings_db.get("TRADE_ASSETS", settings.TRADE_ASSETS)
    trade_assets = [a.strip() for a in trade_assets_str.split(",") if a.strip()]
    
    # 2. Загружаем per-asset пороги и настройки для найденных активов
    threshold_keys = []
    for asset in trade_assets:
        asset_upper = asset.upper()
        threshold_keys.append(f"TRADE_FLIP_THRESHOLD_{asset_upper}")
        threshold_keys.append(f"TRADING_MODE_{asset_upper}")
        threshold_keys.append(f"MIN_EDGE_{asset_upper}")
        threshold_keys.append(f"TRADE_MAX_PRICE_{asset_upper}")
        
    if threshold_keys:
        t_stmt = select(RuntimeSettings).where(RuntimeSettings.key.in_(threshold_keys))
        t_result = await db_session.execute(t_stmt)
        for s in t_result.scalars().all():
            settings_db[s.key] = s.value

    trading_enabled = settings_db.get("TRADING_ENABLED", str(settings.TRADING_ENABLED)).lower() == "true"
    trading_mode = settings_db.get("TRADING_MODE", settings.TRADING_MODE)
    
    if not trading_enabled:
        logger.info("trading_disabled_skipping", mode=trading_mode)
        return
        
    min_time_left = int(settings_db.get("TRADE_MIN_TIME_LEFT_SEC", settings.TRADE_MIN_TIME_LEFT_SEC))
    max_time_left = int(settings_db.get("TRADE_MAX_TIME_LEFT_SEC", settings.TRADE_MAX_TIME_LEFT_SEC))
    bet_size = float(settings_db.get("TRADE_BET_SIZE_USDC", settings.TRADE_BET_SIZE_USDC))
    no_flip_threshold = float(settings_db.get("TRADE_NO_FLIP_THRESHOLD", settings.TRADE_NO_FLIP_THRESHOLD))
    
    dead_zone = float(settings_db.get("DEAD_ZONE_WIDTH", str(settings.DEAD_ZONE_WIDTH)))
    daily_limit = float(settings_db.get("DAILY_LOSS_LIMIT_USDC", str(settings.DAILY_LOSS_LIMIT_USDC)))
    trade_min_price = float(settings_db.get("TRADE_MIN_PRICE", settings.TRADE_MIN_PRICE))
    trade_max_price = float(settings_db.get("TRADE_MAX_PRICE", settings.TRADE_MAX_PRICE))
    capital = float(settings_db.get("INITIAL_CAPITAL", getattr(settings, 'INITIAL_CAPITAL', 100.0)))
    
    active_features_str = settings_db.get("ACTIVE_FEATURES", settings.ACTIVE_FEATURES)
    trade_on_flip = settings_db.get("TRADE_ON_FLIP", "true" if settings.TRADE_ON_FLIP else "false").lower() == "true"
    flip_threshold = float(settings_db.get("FLIP_THRESHOLD", str(settings.FLIP_THRESHOLD)))
    no_max_price = float(settings_db.get("OUTSIDER_MAX_PRICE", str(settings.OUTSIDER_MAX_PRICE)))
    no_min_edge = float(settings_db.get("NO_MIN_EDGE", str(settings.NO_MIN_EDGE)))
    entry_sec = int(settings_db.get("FAVORITE_MODE_ENTRY_SEC", str(settings.FAVORITE_MODE_ENTRY_SEC)))
    min_edge = float(settings_db.get("MIN_EDGE", str(settings.MIN_EDGE)))
    max_bet_edge = float(settings_db.get("MAX_BET_EDGE", str(settings.MAX_BET_EDGE)))       # потолок масштабирования
    max_edge_filter = float(settings_db.get("MAX_EDGE_FILTER", str(settings.MAX_EDGE_FILTER))) # фильтр аномалий
    favorite_threshold = float(settings_db.get("FAVORITE_THRESHOLD", str(settings.FAVORITE_THRESHOLD)))

    # Вычисляем объединенный временной интервал для запроса рынков
    union_min_sec = min(min_time_left, entry_sec)
    union_max_sec = max(max_time_left, entry_sec + FAVORITE_MODE_ENTRY_WINDOW_SEC)
    
    min_td = timedelta(seconds=union_min_sec)
    max_td = timedelta(seconds=union_max_sec)
    
    live_markets_stmt = select(LiveMarket).where(
        and_(
            LiveMarket.end_time_est >= start_time + min_td,
            LiveMarket.end_time_est <= start_time + max_td
        )
    )
    markets = (await db_session.execute(live_markets_stmt)).scalars().all()
    
    if markets:
        # Проверяем дневной PnL перед тем как торговать (лимит $100 убытка в день)
        start_of_today = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_today = start_of_today + timedelta(days=1)
        
        daily_pnl_stmt = select(func.sum(TradeHistory.pnl)).where(
            and_(
                TradeHistory.created_at >= start_of_today,
                TradeHistory.created_at < end_of_today,
                TradeHistory.status == "SUCCESS",
                TradeHistory.pnl.is_not(None)
            )
        )

        daily_pnl = (await db_session.execute(daily_pnl_stmt)).scalar() or 0.0
        if daily_pnl <= daily_limit:
            logger.warning("daily_loss_limit_reached", pnl=daily_pnl, limit=daily_limit)
            return
            
    models_by_asset = None
    model_versions = {}
    model_features = {}
    crypto_predictor = _get_crypto_predictor()
    use_crypto_confirm = settings_db.get("USE_CRYPTO_CONFIRM", "false").lower() == "true"
    crypto_standalone = settings_db.get("CRYPTO_STANDALONE", "false").lower() == "true"


    try:
        # ══════════════════════════════════════════════════
        # ЕДИИЫЙ ЦИКЛ ТОРГОВЛИ (ML и FAVORITE для каждого актива отдельно)
        # ══════════════════════════════════════════════════
        for market in markets:
            end_time = market.end_time_est
            if end_time.tzinfo is None:
                end_time = end_time.replace(tzinfo=timezone.utc)
            time_left_sec = (end_time - start_time).total_seconds()
            
            if time_left_sec <= 0:
                continue

            asset_upper = market.asset.upper()
            
            # Определяем индивидуальный режим актива
            asset_mode = settings_db.get(f"TRADING_MODE_{asset_upper}", trading_mode)
            if not asset_mode:
                asset_mode = trading_mode

            # Проверяем временное окно входа для выбранного режима
            if asset_mode == TRADING_MODE_FAVORITE:
                window_min = entry_sec
                window_max = entry_sec + FAVORITE_MODE_ENTRY_WINDOW_SEC
                if not (window_min <= time_left_sec <= window_max):
                    continue
            else:
                if not (min_time_left <= time_left_sec <= max_time_left):
                    continue

            # Проверка дублей — не торговать дважды на одном рынке
            trade_check = select(TradeHistory).where(
                TradeHistory.market_id == market.market_id
            ).limit(TRADE_CHECK_LIMIT)
            existing_trades = (await db_session.execute(trade_check)).scalars().all()
            
            existing_statuses = [t.status for t in existing_trades]
            if any(s in ("SUCCESS", "LIVE", "FAILED") for s in existing_statuses):
                continue
            
            existing_skipped = next((t for t in existing_trades if t.status == "SKIPPED"), None)
            
            # Актив разрешён?
            if market.asset not in trade_assets:
                continue

            # Разрешаем индивидуальные настройки
            asset_min_edge_val = settings_db.get(f"MIN_EDGE_{asset_upper}")
            if asset_min_edge_val is not None and asset_min_edge_val != "":
                asset_min_edge = float(asset_min_edge_val)
            else:
                asset_min_edge = min_edge

            asset_max_price_val = settings_db.get(f"TRADE_MAX_PRICE_{asset_upper}")
            if asset_max_price_val is not None and asset_max_price_val != "":
                asset_max_price = float(asset_max_price_val)
            else:
                asset_max_price = trade_max_price

            yes_token_id = market.yes_token_id
            no_token_id = market.no_token_id
            
            if not yes_token_id or not no_token_id or yes_token_id == 'N/A' or no_token_id == 'N/A':
                logger.error("cannot_find_token_id_in_db", market_id=market.market_id)
                await save_or_update_skipped_trade(db_session, market, "Token IDs missing in DB", 0.0, None, start_time, existing_skipped=existing_skipped)
                continue

            if asset_mode == TRADING_MODE_CRYPTO:
                # --- CRYPTO STANDALONE MODE ---
                binance_symbol = "BTCUSDT" if market.asset.upper() == "BTC" else "ETHUSDT"
                
                # Загружаем модель и пороги (запросы к БД произойдут только при первом вызове для символа)
                await crypto_predictor.load(db_session, binance_symbol)
                
                model_interval = crypto_predictor.get_interval(binance_symbol)
                candles = await get_recent_candles(db_session, binance_symbol, interval=model_interval, limit=MIN_CANDLES_REQUIRED)
                crypto_sig = crypto_predictor.predict(candles, binance_symbol)
                
                if not crypto_sig.features_ok:
                    await save_or_update_skipped_trade(
                        db_session, market, "Invalid crypto features", 
                        0.0, crypto_sig.model_version, start_time, existing_skipped=existing_skipped
                    )
                    continue

                # Запрашиваем свежие цены Polymarket YES-токена
                fresh_yes_prices = await api_client.get_market_prices(yes_token_id)
                if not fresh_yes_prices or "current_yes_price" not in fresh_yes_prices:
                    error_detail = fresh_yes_prices.get("error", "unknown") if fresh_yes_prices else "None returned"
                    await save_or_update_skipped_trade(
                        db_session, market, f"No fresh yes price: {error_detail}",
                        0.0, crypto_sig.model_version, start_time, existing_skipped=existing_skipped
                    )
                    continue
                fresh_yes_price = fresh_yes_prices["current_yes_price"]

                decision_obj = decide_crypto_trend(crypto_sig, fresh_yes_price, market.volume_5min or 0.0, settings_db)
                p_flip = 0.0  # Для крипто-стратегии p_flip семантически не имеет значения
                model_ver = crypto_sig.model_version
                edge = decision_obj.edge

            elif asset_mode == TRADING_MODE_FAVORITE:
                # --- FAVORITE MODE ---
                if market.current_yes_price == 0.5:
                    logger.info("favorite_mode_skip_no_favorite", market_id=market.market_id)
                    await save_or_update_skipped_trade(
                        db_session, market,
                        "Pure Favorite: no clear favorite (price == 0.5)",
                        0.0, None, start_time,
                        existing_skipped=existing_skipped
                    )
                    continue

                signal = MarketSignal(
                    asset=market.asset,
                    mid_price=market.current_yes_price,
                    spread=market.current_spread or 0.01,
                    volume_5min=market.volume_5min or 0.0,
                    price_velocity=market.price_velocity or 0.0,
                    hour_of_day=start_time.hour,
                    time_left_min=time_left_sec / 60.0
                )
                
                local_fav_config = {**settings_db}
                local_fav_config["MIN_EDGE"] = asset_min_edge
                local_fav_config["TRADE_MAX_PRICE"] = asset_max_price
                if asset_min_edge_val is not None and asset_min_edge_val != "":
                    local_fav_config["FAVORITE_MIN_EDGE"] = asset_min_edge
                else:
                    local_fav_config["FAVORITE_MIN_EDGE"] = float(settings_db.get("FAVORITE_MIN_EDGE", "-0.01"))
                
                decision_obj = decide_favorite(signal, local_fav_config)
                
                p_flip = 0.0
                model_ver = None
                edge = decision_obj.edge
            else:
                # --- ML MODE ---
                if models_by_asset is None:
                    # Загружаем active models
                    models_stmt = select(ModelRegistry).where(ModelRegistry.is_active)
                    active_models = (await db_session.execute(models_stmt)).scalars().all()
                    
                    models_by_asset = {}
                    for m in active_models:
                        try:
                            model_obj = pickle.loads(m.model_blob)
                            models_by_asset[m.asset] = model_obj
                            model_versions[m.asset] = m.version
                            
                            m_feats = [f.strip() for f in m.features.split(",") if f.strip()] if m.features else []
                            if not m_feats and hasattr(model_obj, "feature_names_in_"):
                                m_feats = list(model_obj.feature_names_in_)
                                
                            model_features[m.asset] = m_feats
                        except Exception as e:
                            logger.error("failed_to_load_model", asset=m.asset, error=str(e))

                model = models_by_asset.get(market.asset)
                model_ver = model_versions.get(market.asset)
                m_features = model_features.get(market.asset, [])
                
                if not model:
                    logger.warning("no_active_model_for_trade", asset=market.asset, market_id=market.market_id)
                    await save_or_update_skipped_trade(db_session, market, "No active model", 0.0, model_ver, start_time, existing_skipped=existing_skipped)
                    continue
                    
                if not m_features:
                    logger.warning("model_has_no_features", asset=market.asset, version=model_ver)
                    await save_or_update_skipped_trade(db_session, market, f"Model v{model_ver} has no features", 0.0, model_ver, start_time, existing_skipped=existing_skipped)
                    continue

                fresh_yes_prices = await api_client.get_market_prices(yes_token_id)
                if not fresh_yes_prices or "error" in fresh_yes_prices:
                    error_msg = fresh_yes_prices.get("error", "No fresh YES prices from API") if fresh_yes_prices else "No fresh YES prices from API"
                    logger.warning("no_fresh_yes_prices", market_id=market.market_id, error=error_msg)
                    await save_or_update_skipped_trade(
                        db_session, market, error_msg,
                        0.0, model_ver, start_time,
                        existing_skipped=existing_skipped
                    )
                    continue

                fresh_yes_price = fresh_yes_prices["current_yes_price"]
                fresh_spread = fresh_yes_prices["current_spread"]
                fresh_price_velocity = market.price_velocity

                feature_data = {
                    "time_left_min": time_left_sec / 60.0,
                    "mid_price": fresh_yes_price,
                    "spread": fresh_spread,
                    "price_velocity": fresh_price_velocity,
                    "volume_5min": market.volume_5min,
                    "hour_of_day": start_time.hour
                }
                
                df_features = pd.DataFrame([feature_data])
                df_features = add_derived_features(df_features)
                
                missing = [f for f in m_features if f not in df_features.columns]
                if missing:
                    logger.error("inference_missing_features", asset=market.asset, missing=missing)
                    await save_or_update_skipped_trade(db_session, market, f"Missing features: {missing}", 0.0, model_ver, start_time, existing_skipped=existing_skipped)
                    continue
                
                X_real = df_features[m_features]
                
                proba = model.predict_proba(X_real)[0]
                p_flip = proba[1] if len(proba) > 1 else 0.0
                
                signal = MarketSignal(
                    asset=market.asset,
                    mid_price=fresh_yes_price,
                    spread=fresh_spread,
                    volume_5min=market.volume_5min,
                    price_velocity=market.price_velocity,
                    hour_of_day=start_time.hour,
                    time_left_min=time_left_sec / 60.0
                )

                flip_threshold_key = f"TRADE_FLIP_THRESHOLD_{market.asset.upper()}"
                has_per_asset_threshold = flip_threshold_key in settings_db
                
                base_flip_threshold = (
                    float(settings_db[flip_threshold_key])
                    if has_per_asset_threshold
                    else no_flip_threshold + dead_zone
                )

                auto_dead_zone = settings_db.get("AUTO_DEAD_ZONE", "true" if settings.AUTO_DEAD_ZONE else "false").lower() == "true"
                # dead_zone единый параметр ширины — используется и в авто-, и в ручном режиме
                
                lower, upper = compute_dead_zone(
                    flip_threshold=base_flip_threshold,
                    dead_zone_width=dead_zone,
                    auto_mode=auto_dead_zone,
                )

                local_config = {**settings_db}
                local_config["NO_FLIP_THRESHOLD"] = lower
                local_config["FLIP_THRESHOLD"] = upper
                local_config["MIN_EDGE"] = -100.0
                local_config["MAX_BET_EDGE"] = 100.0
                local_config["BYPASS_BET_SIZE_CHECK"] = "true"

                decision_obj = decide_ml_trend(signal, p_flip, local_config)
                
                if decision_obj.action == "SKIP" and trade_on_flip:
                    decision_obj = decide_outsider(signal, p_flip, local_config)

                # Проверка подтверждения криптой (Confirm Gate)
                if decision_obj.action != "SKIP" and use_crypto_confirm:
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
                                decision_obj, action="SKIP", 
                                reason=f"Crypto confirm veto: direction is {crypto_sig.direction} vs market {market_direction}"
                            )

                edge = decision_obj.edge


            # Единая логика SKIP
            if decision_obj.action == "SKIP":
                reason = decision_obj.reason
                if asset_mode == TRADING_MODE_ML:
                    if lower <= p_flip < upper:
                        reason = f"Мёртвая зона (p_flip={p_flip:.2f} в [{lower:.2f}, {upper:.2f}])"
                    elif not trade_on_flip and p_flip >= upper:
                        reason = f"Ожидается флип (p_flip={p_flip:.2f} >= {upper:.2f})"
                
                logger.info(
                    "trade_decision",
                    asset=market.asset,
                    market_id=market.market_id,
                    action="SKIP",
                    p_flip=round(p_flip, 4),
                    edge=round(edge, 4) if edge is not None else None,
                    buy_price=0.0,
                    strategy=decision_obj.strategy_type,
                    reason=reason
                )
                await save_or_update_skipped_trade(
                    db_session, market, reason, p_flip, model_ver, start_time,
                    existing_skipped=existing_skipped,
                    edge=edge
                )
                continue

            decision = decision_obj.action.replace("BUY_", "")
            buy_price = decision_obj.buy_price
            actual_bet_size = decision_obj.bet_size_usdc
            token_to_buy = yes_token_id if decision == "YES" else no_token_id
            
            fresh_prices = await api_client.get_market_prices(token_to_buy)
            if not fresh_prices or fresh_prices.get("best_ask") is None:
                await save_or_update_skipped_trade(
                    db_session, market, f"No fresh prices from API for {asset_mode} ({decision})",
                    0.0, model_ver, start_time, existing_skipped=existing_skipped
                )
                continue

            fresh_ask = fresh_prices["best_ask"]
            price_drift = abs(fresh_ask - buy_price)
            if price_drift > float(settings_db.get("MAX_PRICE_DRIFT", str(settings.MAX_PRICE_DRIFT))):
                await save_or_update_skipped_trade(
                    db_session, market, f"Price drift too large: {price_drift:.3f}",
                    0.0, model_ver, start_time, existing_skipped=existing_skipped
                )
                continue

            buy_price = fresh_ask
            
            # Пересчитываем edge
            if asset_mode == TRADING_MODE_CRYPTO:
                # В крипто-режиме edge берется из предсказания, а не из цен Polymarket
                edge = decision_obj.edge
                actual_bet_size = decision_obj.bet_size_usdc
            else:
                if asset_mode == TRADING_MODE_ML:
                    p_win = 1.0 - p_flip if decision_obj.strategy_type == "ML_TREND" else p_flip
                    current_min_edge = asset_min_edge
                else:
                    p_win = market.current_yes_price if decision == "YES" else (1.0 - market.current_yes_price)
                    favorite_min_edge = settings_db.get("FAVORITE_MIN_EDGE")
                    if favorite_min_edge is not None and favorite_min_edge != "":
                        current_min_edge = float(favorite_min_edge)
                    else:
                        current_min_edge = asset_min_edge  # fallback на общий min_edge
                edge = compute_edge(p_win, buy_price)
                # Сначала фильтр аномального edge (SKIP если edge выходит за [min_edge, max_edge_filter])
                if edge < current_min_edge or edge > max_edge_filter:
                    await save_or_update_skipped_trade(
                        db_session, market, f"Edge out of bounds (edge={edge:.4f})",
                        p_flip, model_ver, start_time,
                        existing_skipped=existing_skipped,
                        edge=edge
                    )
                    continue

            if not (trade_min_price <= buy_price <= asset_max_price):
                await save_or_update_skipped_trade(
                    db_session, market, f"Price out of bounds: {buy_price:.3f} [{trade_min_price}, {asset_max_price}]",
                    p_flip, model_ver, start_time,
                    existing_skipped=existing_skipped,
                    edge=edge
                )
                continue
            
            if asset_mode != TRADING_MODE_CRYPTO:
                sizing_mode = settings_db.get("BET_SIZING_MODE", settings.BET_SIZING_MODE)
                if sizing_mode == "fixed":
                    actual_bet_size = bet_size
                else:
                    max_bet_usdc = float(settings_db.get("MAX_BET_SIZE_USDC", str(settings.MAX_BET_SIZE_USDC)))
                    actual_bet_size = compute_bet_size_edge_scaled(
                        edge=edge,
                        min_bet_usdc=bet_size,
                        max_bet_usdc=max_bet_usdc,
                        min_edge=current_min_edge,
                        max_edge=max_bet_edge    # потолок масштабирования
                    )
                if asset_mode == TRADING_MODE_FAVORITE and actual_bet_size < bet_size:
                    actual_bet_size = bet_size

            if actual_bet_size <= 0:
                await save_or_update_skipped_trade(
                    db_session, market, "Bet size <= 0",
                    p_flip, model_ver, start_time,
                    existing_skipped=existing_skipped,
                    edge=edge
                )
                continue
            
            num_shares = round(actual_bet_size / buy_price, 2)
            
            logger.info(
                "trade_decision",
                asset=market.asset,
                market_id=market.market_id,
                action=decision_obj.action,
                p_flip=round(p_flip, 4) if p_flip is not None else None,
                p_up=round(decision_obj.p_up, 4) if decision_obj.p_up is not None else None,
                strike=decision_obj.strike,
                edge=round(edge, 4) if edge is not None else None,
                buy_price=buy_price,
                strategy=decision_obj.strategy_type,
                bet_size=actual_bet_size
            )
            
            trade_res = await trader.execute_trade(
                market_id=market.market_id,
                token_id=token_to_buy,
                side="BUY",
                price=buy_price,
                size=num_shares
            )
            
            if existing_skipped:
                await db_session.delete(existing_skipped)

            history = TradeHistory(
                market_id=market.market_id,
                asset=market.asset,
                outcome_bought=decision,
                amount_usdc=trade_res.get("executed_usdc", actual_bet_size),
                executed_price=trade_res.get("executed_price", buy_price),
                predicted_flip_prob=p_flip,
                p_up=decision_obj.p_up,
                strike=decision_obj.strike,
                active_features="CRYPTO_TREND" if asset_mode == TRADING_MODE_CRYPTO else (
                    (f"{active_features_str.strip().rstrip(',')},{decision_obj.strategy_type.lower()}" if (active_features_str and active_features_str.strip().rstrip(',')) else decision_obj.strategy_type.lower()) if asset_mode == TRADING_MODE_ML else "PURE_FAVORITE"
                ),
                model_version=model_ver,
                status=trade_res.get("status", "FAILED"),
                error_msg=trade_res.get("error_msg"),
                mode=trade_res.get("mode", "PAPER"),
                edge=round(edge, 4) if edge is not None else None,
                created_at=start_time
            )
            db_session.add(history)
            await db_session.flush()

            if trade_res.get("status") == "SUCCESS":
                exec_p = trade_res.get("executed_price", buy_price)
                slip = round(exec_p - buy_price, 6)
                slip_pct = round(slip / buy_price * 100, 4) if buy_price > 0 else 0.0
                slip_cost = round(slip * (actual_bet_size / exec_p), 4) if exec_p > 0 else 0.0

                slippage_record = SlippageLog(
                    trade_id=history.id,
                    market_id=market.market_id,
                    asset=market.asset,
                    outcome_bought=decision,
                    expected_price=buy_price,
                    executed_price=exec_p,
                    slippage=slip,
                    slippage_pct=slip_pct,
                    bet_size_usdc=actual_bet_size,
                    slippage_cost_usdc=slip_cost,
                    mode=trade_res.get("mode", "PAPER"),
                    created_at=start_time,
                )
                db_session.add(slippage_record)

            invalidate_stats_cache()
            invalidate_dashboard_cache()
                    
    except Exception as e:
        logger.exception("trade_worker_error", error=str(e))
    finally:
        # Commit in finally to ensure successfully executed trades are saved even if cycle crashed mid-way
        try:
            await db_session.commit()
        except Exception as e_commit:
            logger.error("failed_to_commit_in_finally", error=str(e_commit))
