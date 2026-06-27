import os
import pickle
import pandas as pd
from datetime import datetime, timezone
from typing import Optional
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func

from polyflip.config import settings
from polyflip.db.models import LiveMarket, ModelRegistry, RuntimeSettings, TradeHistory
from polyflip.trading.trader import PolyTrader
from polyflip.collector.client import PolymarketClient
from polyflip.trading.utils import compute_kelly_multiplier

logger = structlog.get_logger(__name__)

async def save_or_update_skipped_trade(
    db_session: AsyncSession,
    market,
    reason: str,
    p_flip_val: float,
    model_version: Optional[int],
    start_time: datetime,
    existing_skipped: Optional[TradeHistory] = None,
    kelly_fraction: Optional[float] = None,
    kelly_multiplier: Optional[float] = None
):
    """Сохраняет запись о пропуске сделки в БД или обновляет её причину."""
    if existing_skipped:
        if existing_skipped.error_msg != reason or existing_skipped.predicted_flip_prob != p_flip_val:
            existing_skipped.error_msg = reason
            existing_skipped.predicted_flip_prob = p_flip_val
            existing_skipped.model_version = model_version
            existing_skipped.kelly_fraction = kelly_fraction
            existing_skipped.kelly_multiplier = kelly_multiplier
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
            kelly_fraction=kelly_fraction,
            kelly_multiplier=kelly_multiplier,
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
        "TRADE_FLIP_THRESHOLD",
        "ACTIVE_FEATURES",
        "TRADE_ONLY_FAVORITE",
        "TRADE_MIN_PRICE",
        "TRADE_MAX_PRICE",
        "TRADE_ASSETS",
        "TRADE_CAPITAL_USDC"
    ]
    stmt = select(RuntimeSettings).where(RuntimeSettings.key.in_(settings_keys))
    result = await db_session.execute(stmt)
    settings_db = {s.key: s.value for s in result.scalars().all()}
    
    # Сначала определяем список торгуемых активов
    trade_assets_str = settings_db.get("TRADE_ASSETS", settings.TRADE_ASSETS)
    trade_assets = [a.strip() for a in trade_assets_str.split(",") if a.strip()]
    
    # 2. Загружаем per-asset пороги для найденных активов
    threshold_keys = [f"TRADE_FLIP_THRESHOLD_{asset.upper()}" for asset in trade_assets]
    if threshold_keys:
        t_stmt = select(RuntimeSettings).where(RuntimeSettings.key.in_(threshold_keys))
        t_result = await db_session.execute(t_stmt)
        for s in t_result.scalars().all():
            settings_db[s.key] = s.value

    # Дефолтные значения из config, если нет в БД
    trading_enabled = settings_db.get("TRADING_ENABLED", str(settings.TRADING_ENABLED)).lower() == "true"
    
    if not trading_enabled:
        return # Торговля выключена
        
    min_time_left = int(settings_db.get("TRADE_MIN_TIME_LEFT_SEC", settings.TRADE_MIN_TIME_LEFT_SEC))
    max_time_left = int(settings_db.get("TRADE_MAX_TIME_LEFT_SEC", settings.TRADE_MAX_TIME_LEFT_SEC))
    bet_size = float(settings_db.get("TRADE_BET_SIZE_USDC", settings.TRADE_BET_SIZE_USDC))
    no_flip_threshold = float(settings_db.get("TRADE_NO_FLIP_THRESHOLD", settings.TRADE_NO_FLIP_THRESHOLD))
    
    
    trade_only_favorite = settings_db.get("TRADE_ONLY_FAVORITE", str(settings.TRADE_ONLY_FAVORITE)).lower() == "true"
    trade_min_price = float(settings_db.get("TRADE_MIN_PRICE", settings.TRADE_MIN_PRICE))
    trade_max_price = float(settings_db.get("TRADE_MAX_PRICE", settings.TRADE_MAX_PRICE))
    
    active_features_str = settings_db.get("ACTIVE_FEATURES", settings.ACTIVE_FEATURES)

    # 2. Ищем рынки, которые подходят по времени для ставки (в пределах настраиваемого диапазона)
    # Чтобы не ставить дважды, проверяем TradeHistory.
    
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
    
    if markets:
        # Проверяем дневной PnL перед тем как торговать (лимит $100 убытка в день)
        # BUG-005 FIX: Явно фильтруем по диапазону UTC времени для надежного подсчета
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
        if daily_pnl <= -100.0:
            logger.warning("daily_loss_limit_reached", pnl=daily_pnl, limit=-100.0)
            return
            
    # Загружаем активные модели
    models_stmt = select(ModelRegistry).where(ModelRegistry.is_active)
    active_models = (await db_session.execute(models_stmt)).scalars().all()
    
    models_by_asset = {}
    model_versions = {}
    model_features = {}
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
    
    try:
        for market in markets:
            end_time = market.end_time_est
            if end_time.tzinfo is None:
                end_time = end_time.replace(tzinfo=timezone.utc)
            time_left_sec = (end_time - start_time).total_seconds()
            
            if time_left_sec > 0:
                # Проверяем, делали ли мы уже ставку на этот рынок (или логировали пропуск)
                trade_check = select(TradeHistory).where(TradeHistory.market_id == market.market_id)
                existing_trades = (await db_session.execute(trade_check)).scalars().all()
                
                existing_statuses = [t.status for t in existing_trades]
                if any(s in ("SUCCESS", "LIVE", "FAILED") for s in existing_statuses):
                    continue
                    
                existing_skipped = next((t for t in existing_trades if t.status == "SKIPPED"), None)
                    
                if market.asset not in trade_assets:
                    logger.info("trade_skipped_asset_not_enabled", asset=market.asset)
                    continue

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
                
                try:
                    X_real = df_features[m_features]
                except KeyError as e:
                    logger.error("trade_engine_missing_features", error=str(e))
                    await save_or_update_skipped_trade(db_session, market, f"Missing features: {str(e)}", 0.0, model_ver, start_time, existing_skipped=existing_skipped)
                    continue
                
                # Предсказываем вероятность флипа (класс 1)
                proba = model.predict_proba(X_real)[0]
                p_flip = proba[1] if len(proba) > 1 else 0.0
                
                # Логика принятия решения
                decision = None
                
                # Шаг 3: Калибровка порога. Считываем TRADE_FLIP_THRESHOLD_{asset}
                flip_threshold_key = f"TRADE_FLIP_THRESHOLD_{market.asset.upper()}"
                has_per_asset_threshold = flip_threshold_key in settings_db
                current_flip_threshold = float(settings_db.get(
                    flip_threshold_key,
                    settings_db.get("TRADE_FLIP_THRESHOLD", settings.TRADE_FLIP_THRESHOLD)
                ))
                
                # Задаем no_flip порог: если порог per-asset, то вычисляем (flip - 0.15),
                # иначе используем глобальный.
                if has_per_asset_threshold:
                    current_no_flip_threshold = round(current_flip_threshold - 0.15, 4)
                else:
                    current_no_flip_threshold = no_flip_threshold
                
                if p_flip > current_flip_threshold:
                    if not trade_only_favorite:
                        # Модель ждет флип. Покупаем аутсайдера.
                        decision = "NO" if market.current_yes_price > 0.5 else "YES"
                    else:
                        logger.info("trade_flip_signal_skipped_only_favorite", market_id=market.market_id, p_flip=p_flip)
                        await save_or_update_skipped_trade(db_session, market, "Flip expected, but Only Favorite is enabled", p_flip, model_ver, start_time, existing_skipped=existing_skipped)
                        continue
                elif p_flip < current_no_flip_threshold:
                    # Модель считает, что рынок прав. Покупаем фаворита.
                    decision = "YES" if market.current_yes_price > 0.5 else "NO"
                    
                if decision:
                    logger.info("trade_decision_made", market_id=market.market_id, p_flip=p_flip, decision=decision)
                    
                    # Извлекаем токены напрямую из БД (без дополнительных API запросов)
                    yes_token_id = market.yes_token_id
                    no_token_id = market.no_token_id
                    
                    if not yes_token_id or not no_token_id or yes_token_id == 'N/A' or no_token_id == 'N/A':
                        logger.error("cannot_find_token_id_in_db", market_id=market.market_id)
                        await save_or_update_skipped_trade(db_session, market, "Token IDs missing in DB", p_flip, model_ver, start_time, existing_skipped=existing_skipped)
                        continue
                        
                    token_to_buy = yes_token_id if decision == "YES" else no_token_id
                    
                    # Цена = лучший Ask. Запрашиваем книгу именно для того токена, который покупаем
                    fresh_prices = await api_client.get_market_prices(token_to_buy)
                    if not fresh_prices:
                        continue
                        
                    buy_price = fresh_prices.get("best_ask", 0)
                    
                    if buy_price < trade_min_price or buy_price > trade_max_price:
                        logger.warning("trade_skipped_price_out_of_range", price=buy_price, min=trade_min_price, max=trade_max_price)
                        await save_or_update_skipped_trade(db_session, market, f"Price out of range: {buy_price}", p_flip, model_ver, start_time, existing_skipped=existing_skipped)
                        continue
                        
                    # Шаг 4: Вычисляем размер ставки по критерию Келли
                    is_flip_bet = (p_flip > current_flip_threshold)
                    p_win = p_flip if is_flip_bet else (1.0 - p_flip)
                    
                    kelly_f, kelly_multiplier = compute_kelly_multiplier(p_win, buy_price, max_fraction=0.10)
                    actual_bet_size = round(bet_size * kelly_multiplier, 2)
                    
                    logger.info(
                        "kelly_calculated",
                        p_win=round(p_win, 3),
                        buy_price=buy_price,
                        kelly_f=kelly_f,
                        kelly_multiplier=kelly_multiplier,
                        actual_bet_size=actual_bet_size,
                    )
                    
                    # Кол-во акций = size / price
                    num_shares = round(actual_bet_size / buy_price, 2)
                    
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
                        amount_usdc=actual_bet_size,
                        executed_price=buy_price,
                        predicted_flip_prob=p_flip,
                        active_features=active_features_str,
                        model_version=model_ver,
                        status=trade_res.get("status", "FAILED"),
                        error_msg=trade_res.get("error_msg"),
                        mode=trade_res.get("mode", "PAPER"),
                        kelly_fraction=round(kelly_f, 4),
                        kelly_multiplier=round(kelly_multiplier, 2),
                        created_at=start_time
                    )
                    db_session.add(history)
                else:
                    logger.info("trade_skipped", market_id=market.market_id, p_flip=p_flip)
                    reason = f"P(flip) {p_flip:.1%} is within [{current_no_flip_threshold:.1%} - {current_flip_threshold:.1%}] range"
                    await save_or_update_skipped_trade(
                        db_session, market, reason, p_flip, model_ver, start_time,
                        existing_skipped=existing_skipped,
                        kelly_fraction=None, kelly_multiplier=None
                    )
                    
    except Exception as e:
        logger.exception("trade_worker_error", error=str(e))
    finally:
        # Commit in finally to ensure successfully executed trades are saved even if cycle crashed mid-way
        try:
            await db_session.commit()
        except Exception as e_commit:
            logger.error("failed_to_commit_in_finally", error=str(e_commit))
