import asyncio
import pickle
import pandas as pd
from datetime import datetime, timezone
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from polyflip.db.models import LiveMarket, ModelRegistry, RuntimeSettings, TradeHistory
from polyflip.trading.trader import PolyTrader
from polyflip.collector.client import PolymarketClient

logger = structlog.get_logger(__name__)

async def trade_worker_cycle(db_session: AsyncSession):
    """
    Фоновый процесс торгового движка.
    """
    start_time = datetime.now(timezone.utc)
    
    # 1. Загружаем торговые настройки
    settings_keys = [
        "TRADING_ENABLED", 
        "TRADE_EXECUTION_TIME_SEC", 
        "TRADE_BET_SIZE_USDC",
        "TRADE_NO_FLIP_THRESHOLD",
        "TRADE_FLIP_THRESHOLD",
        "ACTIVE_FEATURES"
    ]
    
    stmt = select(RuntimeSettings).where(RuntimeSettings.key.in_(settings_keys))
    result = await db_session.execute(stmt)
    settings_db = {s.key: s.value for s in result.scalars().all()}
    
    # Дефолтные значения из config, если нет в БД
    from polyflip.config import settings
    trading_enabled = settings_db.get("TRADING_ENABLED", str(settings.TRADING_ENABLED)).lower() == "true"
    
    if not trading_enabled:
        return # Торговля выключена
        
    execution_time_sec = int(settings_db.get("TRADE_EXECUTION_TIME_SEC", settings.TRADE_EXECUTION_TIME_SEC))
    bet_size = float(settings_db.get("TRADE_BET_SIZE_USDC", settings.TRADE_BET_SIZE_USDC))
    no_flip_threshold = float(settings_db.get("TRADE_NO_FLIP_THRESHOLD", settings.TRADE_NO_FLIP_THRESHOLD))
    flip_threshold = float(settings_db.get("TRADE_FLIP_THRESHOLD", settings.TRADE_FLIP_THRESHOLD))
    
    active_features_str = settings_db.get("ACTIVE_FEATURES", settings.ACTIVE_FEATURES)
    active_features = [f.strip() for f in active_features_str.split(",") if f.strip()]

    # 2. Ищем рынки, которые подходят по времени для ставки (например, ровно 30 сек до конца)
    # Так как цикл работает раз в 5-10 сек, мы даем окно, например, от (X) до (X + 15) секунд, 
    # чтобы не пропустить рынок. Чтобы не ставить дважды, проверяем TradeHistory.
    
    min_time_left = execution_time_sec - 5
    max_time_left = execution_time_sec + 15
    
    from datetime import timedelta
    min_td = timedelta(seconds=min_time_left)
    max_td = timedelta(seconds=max_time_left)
    
    live_markets_stmt = select(LiveMarket).where(
        and_(
            LiveMarket.end_time_est >= start_time + min_td,
            LiveMarket.end_time_est <= start_time + max_td
        )
    )
    markets = (await db_session.execute(live_markets_stmt)).scalars().all()
    
    # Загружаем активные модели
    models_stmt = select(ModelRegistry).where(ModelRegistry.is_active == True)
    active_models = (await db_session.execute(models_stmt)).scalars().all()
    
    models_by_asset = {}
    for m in active_models:
        try:
            models_by_asset[m.asset] = pickle.loads(m.model_blob)
        except Exception as e:
            logger.error("failed_to_load_model", asset=m.asset, error=str(e))
    
    trader = None
    api_client = None
    
    try:
        trader = PolyTrader()
        api_client = PolymarketClient()
        
        for market in markets:
            time_left_sec = (market.end_time_est - start_time).total_seconds()
            
            if time_left_sec > 0:
                # Проверяем, делали ли мы уже ставку на этот рынок
                trade_check = select(TradeHistory.id).where(TradeHistory.market_id == market.market_id)
                already_traded = (await db_session.execute(trade_check)).scalar() is not None
                
                if already_traded:
                    continue
                    
                model = models_by_asset.get(market.asset)
                if not model:
                    logger.warning("no_active_model_for_trade", asset=market.asset, market_id=market.market_id)
                    continue
                    
                # Формируем X
                # Для правильного X нам нужны те же фичи, что и при обучении
                feature_data = {
                    "time_left_min": time_left_sec / 60.0,
                    "mid_price": market.current_yes_price,
                    "spread": market.current_spread,
                    "price_velocity": market.price_velocity,
                    "volume_5min": market.volume_5min,
                    "hour_of_day": start_time.hour
                }
                
                df_features = pd.DataFrame([feature_data])
                
                # Фильтруем только активные фичи
                missing_features = [f for f in active_features if f not in df_features.columns]
                if missing_features:
                    logger.error("trade_engine_missing_features", missing=missing_features)
                    continue
                    
                X_real = df_features[active_features]
                
                # Предсказываем вероятность флипа (класс 1)
                proba = model.predict_proba(X_real)[0]
                p_flip = proba[1] if len(proba) > 1 else 0.0
                
                # Логика принятия решения
                decision = None
                
                if p_flip > flip_threshold:
                    # Модель ждет флип. Покупаем аутсайдера.
                    decision = "NO" if market.current_yes_price > 0.5 else "YES"
                elif p_flip < no_flip_threshold:
                    # Модель считает, что рынок прав. Покупаем фаворита.
                    decision = "YES" if market.current_yes_price > 0.5 else "NO"
                    
                if decision:
                    logger.info("trade_decision_made", market_id=market.market_id, p_flip=p_flip, decision=decision)
                    
                    # Извлекаем токены напрямую из БД (без дополнительных API запросов)
                    yes_token_id = market.yes_token_id
                    no_token_id = market.no_token_id
                    
                    if not yes_token_id or not no_token_id or yes_token_id == 'N/A' or no_token_id == 'N/A':
                        logger.error("cannot_find_token_id_in_db", market_id=market.market_id)
                        continue
                        
                    token_to_buy = yes_token_id if decision == "YES" else no_token_id
                    
                    # Цена = лучший Ask. Запрашиваем книгу именно для того токена, который покупаем
                    fresh_prices = await api_client.get_market_prices(token_to_buy)
                    if not fresh_prices:
                        continue
                        
                    buy_price = fresh_prices.get("best_ask", 0)
                    
                    if buy_price <= 0 or buy_price >= 1:
                        logger.warning("invalid_buy_price", price=buy_price)
                        continue
                        
                    # Кол-во акций = size / price
                    num_shares = round(bet_size / buy_price, 2)
                    
                    # Исполняем
                    trade_res = await trader.execute_trade(
                        market_id=market.market_id,
                        token_id=token_to_buy,
                        side="BUY",
                        price=buy_price,
                        size=num_shares
                    )
                    
                    # Сохраняем в БД
                    history = TradeHistory(
                        market_id=market.market_id,
                        asset=market.asset,
                        outcome_bought=decision,
                        amount_usdc=bet_size,
                        executed_price=buy_price,
                        predicted_flip_prob=p_flip,
                        active_features=active_features_str,
                        status=trade_res["status"],
                        error_msg=trade_res["error_msg"],
                        created_at=start_time
                    )
                    db_session.add(history)
                else:
                    logger.info("trade_skipped", market_id=market.market_id, p_flip=p_flip)
                    
    except Exception as e:
        logger.exception("trade_worker_error", error=str(e))
    finally:
        # BUG-N03 FIX: Commit moved to finally to ensure trades are saved even if loop breaks
        try:
            await db_session.commit()
        except Exception as e_commit:
            logger.error("failed_to_commit_in_finally", error=str(e_commit))
            
        if api_client:
            await api_client.close()
