import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

from polyflip.config import settings
from polyflip.db.models import LiveMarket, ModelRegistry, RuntimeSettings, TradeHistory
from polyflip.trading.trader import PolyTrader
from polyflip.collector.client import PolymarketClient
from polyflip.trading.utils import compute_kelly_multiplier
from polyflip.models.trainer import add_derived_features
from polyflip.api.trading_dashboard import invalidate_stats_cache
from polyflip.api.dashboard import invalidate_dashboard_cache
from polyflip.constants import (
    DEAD_ZONE_WIDTH,
    KELLY_MAX_FRACTION,
    DAILY_LOSS_LIMIT_USDC,
    FAVORITE_THRESHOLD,
    TRADE_CHECK_LIMIT,
    TRADING_MODE_ML,
    TRADING_MODE_FAVORITE,
    FAVORITE_MODE_ENTRY_SEC,
    FAVORITE_MODE_ENTRY_WINDOW_SEC
)

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
    kelly_multiplier: Optional[float] = None,
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
            existing_skipped.kelly_fraction = kelly_fraction
            existing_skipped.kelly_multiplier = kelly_multiplier
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
            kelly_fraction=kelly_fraction,
            kelly_multiplier=kelly_multiplier,
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
        "KELLY_MAX_FRACTION",
        "DAILY_LOSS_LIMIT_USDC",
        "ACTIVE_FEATURES",
        "TRADE_MIN_PRICE",
        "TRADE_MAX_PRICE",
        "TRADE_ASSETS",
        "TRADE_CAPITAL_USDC",  # Зарезервировано для будущего Kelly по % от капитала
        "KELLY_ENABLED",
        "TRADING_MODE",
        "FAVORITE_MODE_ENTRY_SEC",
        "MIN_EDGE",
        "MAX_EDGE",
        "FAVORITE_THRESHOLD"
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

    trading_enabled = settings_db.get("TRADING_ENABLED", str(settings.TRADING_ENABLED)).lower() == "true"
    trading_mode = settings_db.get("TRADING_MODE", settings.TRADING_MODE)
    
    if not trading_enabled:
        logger.info("trading_disabled_skipping", mode=trading_mode)
        return
        
    min_time_left = int(settings_db.get("TRADE_MIN_TIME_LEFT_SEC", settings.TRADE_MIN_TIME_LEFT_SEC))
    max_time_left = int(settings_db.get("TRADE_MAX_TIME_LEFT_SEC", settings.TRADE_MAX_TIME_LEFT_SEC))
    bet_size = float(settings_db.get("TRADE_BET_SIZE_USDC", settings.TRADE_BET_SIZE_USDC))
    no_flip_threshold = float(settings_db.get("TRADE_NO_FLIP_THRESHOLD", settings.TRADE_NO_FLIP_THRESHOLD))
    
    
    dead_zone = float(settings_db.get("DEAD_ZONE_WIDTH", str(DEAD_ZONE_WIDTH)))
    kelly_max = float(settings_db.get("KELLY_MAX_FRACTION", str(KELLY_MAX_FRACTION)))
    daily_limit = float(settings_db.get("DAILY_LOSS_LIMIT_USDC", str(DAILY_LOSS_LIMIT_USDC)))
    trade_min_price = float(settings_db.get("TRADE_MIN_PRICE", settings.TRADE_MIN_PRICE))
    trade_max_price = float(settings_db.get("TRADE_MAX_PRICE", settings.TRADE_MAX_PRICE))
    
    active_features_str = settings_db.get("ACTIVE_FEATURES", settings.ACTIVE_FEATURES)
    entry_sec = int(settings_db.get("FAVORITE_MODE_ENTRY_SEC", str(settings.FAVORITE_MODE_ENTRY_SEC)))
    min_edge = float(settings_db.get("MIN_EDGE", str(settings.MIN_EDGE)))
    max_edge = float(settings_db.get("MAX_EDGE", str(settings.MAX_EDGE)))
    favorite_threshold = float(settings_db.get("FAVORITE_THRESHOLD", str(settings.FAVORITE_THRESHOLD)))

    # 2. Ищем рынки, которые подходят по времени для ставки (в пределах настраиваемого диапазона)
    # Чтобы не ставить дважды, проверяем TradeHistory.
    
    if trading_mode == TRADING_MODE_FAVORITE:
        min_td = timedelta(seconds=entry_sec)
        max_td = timedelta(seconds=entry_sec + FAVORITE_MODE_ENTRY_WINDOW_SEC)
    else:
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
        if daily_pnl <= daily_limit:
            logger.warning("daily_loss_limit_reached", pnl=daily_pnl, limit=daily_limit)
            return
            
    try:
        # ══════════════════════════════════════════════════
        # PURE FAVORITE MODE — без ML, без Kelly
        # ══════════════════════════════════════════════════
        if trading_mode == TRADING_MODE_FAVORITE:
            logger.info("pure_favorite_mode_active", entry_sec=entry_sec, markets_count=len(markets))
            
            for market in markets:
                end_time = market.end_time_est
                if end_time.tzinfo is None:
                    end_time = end_time.replace(tzinfo=timezone.utc)
                time_left_sec = (end_time - start_time).total_seconds()
                
                if time_left_sec <= 0:
                    continue

                # Проверка временного окна входа (двойная защита)
                window_min = entry_sec
                window_max = entry_sec + FAVORITE_MODE_ENTRY_WINDOW_SEC
                if not (window_min <= time_left_sec <= window_max):
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
                
                # Нет явного фаворита — пропускаем
                if market.current_yes_price == 0.5:
                    logger.info("favorite_mode_skip_no_favorite", market_id=market.market_id)
                    await save_or_update_skipped_trade(
                        db_session, market,
                        "Pure Favorite: no clear favorite (price == 0.5)",
                        0.0, None, start_time,
                        existing_skipped=existing_skipped
                    )
                    continue
                
                # Определяем фаворита по цене из БД
                decision = "YES" if market.current_yes_price > favorite_threshold else "NO"
                token_to_buy = market.yes_token_id if decision == "YES" else market.no_token_id
                
                if not token_to_buy or token_to_buy == "N/A":
                    logger.error("favorite_mode_missing_token", market_id=market.market_id)
                    continue
                
                # Запрашиваем свежую цену — исполнять только по актуальному ask
                fresh_prices = await api_client.get_market_prices(token_to_buy)
                if not fresh_prices:
                    logger.warning("favorite_mode_no_fresh_prices", market_id=market.market_id)
                    await save_or_update_skipped_trade(
                        db_session, market,
                        "Pure Favorite: no fresh prices from API",
                        0.0, None, start_time,
                        existing_skipped=existing_skipped
                    )
                    continue
                
                buy_price = fresh_prices.get("best_ask", 0)
                
                # Проверяем ценовые границы
                if buy_price < trade_min_price or buy_price > trade_max_price:
                    logger.warning("favorite_mode_price_out_of_range", price=buy_price)
                    await save_or_update_skipped_trade(
                        db_session, market,
                        f"Pure Favorite: price {buy_price} out of range [{trade_min_price}, {trade_max_price}]",
                        0.0, None, start_time,
                        existing_skipped=existing_skipped
                    )
                    continue
                
                # Цена по API должна подтверждать фаворита
                if decision == "YES" and buy_price < favorite_threshold:
                    logger.warning("favorite_mode_no_longer_favorite", price=buy_price, decision=decision)
                    await save_or_update_skipped_trade(
                        db_session, market,
                        f"Pure Favorite: fresh YES price {buy_price} below threshold {favorite_threshold}",
                        0.0, None, start_time,
                        existing_skipped=existing_skipped
                    )
                    continue
                elif decision == "NO" and buy_price < favorite_threshold:
                    logger.warning("favorite_mode_no_longer_favorite", price=buy_price, decision=decision)
                    await save_or_update_skipped_trade(
                        db_session, market,
                        f"Pure Favorite: NO token price {buy_price} — YES recovered to {round(1.0 - buy_price, 4)}, no longer valid",
                        0.0, None, start_time,
                        existing_skipped=existing_skipped
                    )
                    continue
                
                # Фиксированная ставка, Kelly не применяется
                actual_bet_size = bet_size
                num_shares = round(actual_bet_size / buy_price, 2)
                
                logger.info(
                    "favorite_mode_trade",
                    market_id=market.market_id,
                    asset=market.asset,
                    decision=decision,
                    buy_price=buy_price,
                    bet_size=actual_bet_size,
                    time_left_sec=round(time_left_sec),
                )
                
                trade_res = await trader.execute_trade(
                    market_id=market.market_id,
                    token_id=token_to_buy,
                    side="BUY",
                    price=buy_price,
                    size=num_shares,
                )
                
                if existing_skipped:
                    await db_session.delete(existing_skipped)

                history = TradeHistory(
                    market_id=market.market_id,
                    asset=market.asset,
                    outcome_bought=decision,
                    amount_usdc=trade_res.get("executed_usdc", actual_bet_size),
                    executed_price=trade_res.get("executed_price", buy_price),
                    predicted_flip_prob=0.0,       # ML не использовался
                    active_features="PURE_FAVORITE", # маркер режима
                    model_version=None,
                    status=trade_res.get("status", "FAILED"),
                    error_msg=trade_res.get("error_msg"),
                    mode=trade_res.get("mode", "PAPER"),
                    kelly_fraction=None,
                    kelly_multiplier=1.0,
                    created_at=start_time,
                )
                db_session.add(history)
                invalidate_stats_cache()
                invalidate_dashboard_cache()
            
            return  # <- выходим, ML-блок не запускаем

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

        for market in markets:
            end_time = market.end_time_est
            if end_time.tzinfo is None:
                end_time = end_time.replace(tzinfo=timezone.utc)
            time_left_sec = (end_time - start_time).total_seconds()
            
            if time_left_sec > 0:
                # Проверяем, делали ли мы уже ставку на этот рынок (или логировали пропуск)
                trade_check = select(TradeHistory).where(TradeHistory.market_id == market.market_id).limit(TRADE_CHECK_LIMIT)
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
                    
                # Извлекаем токены напрямую из БД (без дополнительных API запросов)
                yes_token_id = market.yes_token_id
                no_token_id = market.no_token_id
                
                if not yes_token_id or not no_token_id or yes_token_id == 'N/A' or no_token_id == 'N/A':
                    logger.error("cannot_find_token_id_in_db", market_id=market.market_id)
                    await save_or_update_skipped_trade(db_session, market, "Token IDs missing in DB", 0.0, model_ver, start_time, existing_skipped=existing_skipped)
                    continue

                # Запрашиваем свежие цены стакана для YES-токена перед предсказанием модели
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
                
                # Используем price_velocity из БД, так как дельта цен за секунды дает нестабильную скорость
                fresh_price_velocity = market.price_velocity

                # Формируем X с использованием свежих цен
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
                
                # Предсказываем вероятность флипа (класс 1)
                proba = model.predict_proba(X_real)[0]
                p_flip = proba[1] if len(proba) > 1 else 0.0
                
                # Логика принятия решения
                decision = None
                
                # Шаг 3: Калибровка порога. Считываем TRADE_FLIP_THRESHOLD_{asset}
                flip_threshold_key = f"TRADE_FLIP_THRESHOLD_{market.asset.upper()}"
                has_per_asset_threshold = flip_threshold_key in settings_db
                
                if has_per_asset_threshold:
                    calibrated_val = float(settings_db[flip_threshold_key])
                    current_no_flip_threshold = round(calibrated_val - dead_zone, 4)
                else:
                    # Нет per-asset калибровки — используем глобальные настройки напрямую
                    current_no_flip_threshold = no_flip_threshold
                    calibrated_val = no_flip_threshold + dead_zone  # только для reason в логах
                
                if p_flip < current_no_flip_threshold:
                    # Модель считает, что рынок прав. Покупаем фаворита.
                    decision = "YES" if fresh_yes_price > favorite_threshold else "NO"
                    
                if decision:
                    logger.info("trade_decision_made", market_id=market.market_id, p_flip=p_flip, decision=decision)
                    
                    token_to_buy = yes_token_id if decision == "YES" else no_token_id
                    
                    # Проверяем наличие ask цены для YES-токена
                    fresh_yes_ask = fresh_yes_prices.get("best_ask")
                    if fresh_yes_ask is None:
                        logger.warning("no_best_ask_in_yes_prices", market_id=market.market_id)
                        await save_or_update_skipped_trade(db_session, market, "No best_ask in YES prices", p_flip, model_ver, start_time, existing_skipped=existing_skipped)
                        continue
                    
                    # Цена = лучший Ask
                    if decision == "YES":
                        buy_price = fresh_yes_ask
                    else:
                        # Для NO запрашиваем стакан NO-токена
                        fresh_no_prices = await api_client.get_market_prices(no_token_id)
                        if not fresh_no_prices or "error" in fresh_no_prices or fresh_no_prices.get("best_ask") is None:
                            error_msg = "No fresh NO prices (best_ask) from API"
                            if fresh_no_prices and "error" in fresh_no_prices:
                                error_msg = f"NO price error: {fresh_no_prices['error']}"
                            logger.warning("no_fresh_no_prices", market_id=market.market_id, error=error_msg)
                            await save_or_update_skipped_trade(
                                db_session, market, error_msg,
                                p_flip, model_ver, start_time,
                                existing_skipped=existing_skipped
                            )
                            continue
                        buy_price = fresh_no_prices["best_ask"]
                    
                    if buy_price < trade_min_price or buy_price > trade_max_price:
                        logger.warning("trade_skipped_price_out_of_range", price=buy_price, min=trade_min_price, max=trade_max_price)
                        await save_or_update_skipped_trade(db_session, market, f"Price out of range: {buy_price}", p_flip, model_ver, start_time, existing_skipped=existing_skipped)
                        continue
                        
                    # Защита от покупки аутсайдера при резком изменении цены между обновлением БД и вызовом API
                    if buy_price < favorite_threshold:
                        logger.warning("trade_skipped_no_longer_favorite", price=buy_price, threshold=favorite_threshold)
                        await save_or_update_skipped_trade(db_session, market, f"Price dropped to {buy_price}, no longer favorite", p_flip, model_ver, start_time, existing_skipped=existing_skipped)
                        continue
                        
                    p_win = 1.0 - p_flip
                    
                    # Edge = преимущество модели над mid-ценой рынка (НЕ над ask)
                    # mid_price = (best_bid + best_ask) / 2 — справедливая оценка рынка
                    implied_prob = fresh_yes_price if decision == "YES" else (1.0 - fresh_yes_price)
                    edge = round(p_win - implied_prob, 4)
                    if edge < min_edge or edge > max_edge:
                        logger.warning("trade_skipped_edge_out_of_bounds", edge=edge, min_edge=min_edge, max_edge=max_edge, p_win=p_win, implied_prob=implied_prob)
                        await save_or_update_skipped_trade(
                            db_session, market,
                            f"Edge out of bounds: {edge:.3f} not in [{min_edge:.3f}, {max_edge:.3f}]",
                            p_flip, model_ver, start_time,
                            existing_skipped=existing_skipped,
                            edge=edge
                        )
                        continue
                        
                    # Шаг 4: Вычисляем размер ставки по критерию Келли
                    
                    kelly_enabled = settings_db.get("KELLY_ENABLED", "true").lower() == "true"
                    if kelly_enabled:
                        kelly_f, kelly_multiplier = compute_kelly_multiplier(p_win, buy_price, max_fraction=kelly_max)
                        actual_bet_size = round(bet_size * kelly_multiplier, 2)
                    else:
                        kelly_f = None
                        kelly_multiplier = 1.0
                        actual_bet_size = bet_size
                    
                    logger.info(
                        "kelly_calculated",
                        kelly_enabled=kelly_enabled,
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
                    
                    # Если была SKIPPED-запись, удаляем её перед записью реальной сделки
                    if existing_skipped:
                        await db_session.delete(existing_skipped)

                    # Сохраняем в БД
                    history = TradeHistory(
                        market_id=market.market_id,
                        asset=market.asset,
                        outcome_bought=decision,
                        amount_usdc=trade_res.get("executed_usdc", actual_bet_size),
                        executed_price=trade_res.get("executed_price", buy_price),
                        predicted_flip_prob=p_flip,
                        active_features=active_features_str,
                        model_version=model_ver,
                        status=trade_res.get("status", "FAILED"),
                        error_msg=trade_res.get("error_msg"),
                        mode=trade_res.get("mode", "PAPER"),
                        kelly_fraction=round(kelly_f, 4) if kelly_f is not None else None,
                        kelly_multiplier=round(kelly_multiplier, 2),
                        edge=round(edge, 4) if edge is not None else None,
                        created_at=start_time
                    )
                    db_session.add(history)
                    invalidate_stats_cache()
                    invalidate_dashboard_cache()
                else:
                    logger.info("trade_skipped", market_id=market.market_id, p_flip=p_flip)
                    if p_flip >= calibrated_val:
                        reason = f"Ожидается флип: P(flip)={p_flip:.1%} >= порог флипа {calibrated_val:.1%}"
                    else:
                        reason = f"Мёртвая зона: P(flip)={p_flip:.1%} в [{current_no_flip_threshold:.1%}–{calibrated_val:.1%}]"
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
