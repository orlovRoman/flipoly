from datetime import datetime, timezone
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from polyflip.collector.client import PolymarketClient
from polyflip.db.models import MarketSnapshot, LiveMarket, CollectorStatus
from polyflip.config import settings

logger = structlog.get_logger(__name__)

async def run_collector_cycle(db_session: AsyncSession):
    """
    Основной цикл сбора данных:
    1. Ищем активные 15m рынки
    2. Скачиваем стаканы для расчета spread и mid_price
    3. Считаем velocity и volume_5min
    4. Сохраняем в БД
    """
    start_time = datetime.now(timezone.utc)
    client = PolymarketClient()
    markets_found = 0
    markets_saved = 0
    error_msg = None

    try:
        # 1. Получаем список 15-минутных рынков для наших активов
        active_markets = await client.get_active_15m_markets(settings.asset_list)
        markets_found = len(active_markets)
        logger.info("found_active_markets", count=markets_found)

        for m_data in active_markets:
            market_id = m_data["market_id"]
            yes_token_id = m_data["yes_token_id"]
            
            # 2. Получаем текущие цены
            prices = await client.get_market_prices(yes_token_id)
            if not prices:
                continue

            mid_price = prices["current_yes_price"]
            spread = prices["current_spread"]

            # Вычисляем time_left_min
            current_time = datetime.now(timezone.utc)
            end_date = datetime.fromisoformat(m_data["end_date_iso"].replace("Z", "+00:00"))
            time_left_min = (end_date - current_time).total_seconds() / 60.0

            if time_left_min < 0:
                continue # Рынок уже закрылся

            # 3. Получаем предыдущее состояние рынка из БД для вычисления дельт
            result = await db_session.execute(
                select(LiveMarket).where(LiveMarket.market_id == market_id)
            )
            live_m = result.scalar_one_or_none()

            # BUG-003 FIX: Расчет реального объема через историю сделок CLOB
            volume_5min = await client.get_recent_trades_volume(yes_token_id, minutes=5)
            price_velocity = 0.0

            if live_m:
                # Считаем дельту скорости цены
                price_velocity = mid_price - live_m.current_yes_price
                
                # Обновляем LiveMarket
                live_m.current_yes_price = mid_price
                live_m.current_no_price = prices["current_no_price"]
                live_m.current_spread = spread
                live_m.price_velocity = price_velocity
                live_m.volume_5min = volume_5min
                live_m.last_updated = current_time
                # На всякий случай обновляем token_id, если добавились
                live_m.yes_token_id = yes_token_id
                live_m.no_token_id = m_data["no_token_id"]
            else:
                # Создаем новую запись в LiveMarket
                live_m = LiveMarket(
                    market_id=market_id,
                    asset=m_data["asset"],
                    question=m_data["question"],
                    yes_token_id=yes_token_id,
                    no_token_id=m_data["no_token_id"],
                    end_time_est=end_date,
                    current_yes_price=mid_price,
                    current_no_price=prices["current_no_price"],
                    current_spread=spread,
                    volume_5min=volume_5min,
                    price_velocity=0.0,
                    last_updated=current_time
                )
                db_session.add(live_m)

            # 4. Сохраняем Snapshot
            # Внимание: final_outcome и flip_vs_final мы пока не знаем, 
            # они заполняются позже (при резолве рынка)
            snapshot = MarketSnapshot(
                asset=m_data["asset"],
                market_id=market_id,
                time_left_min=time_left_min,
                mid_price=mid_price,
                spread=spread,
                volume_5min=volume_5min,
                price_velocity=price_velocity,
                hour_of_day=current_time.hour,
                final_outcome="PENDING",
                flip_vs_final=False, # Обновится позже
                recorded_at=current_time
            )
            db_session.add(snapshot)
            markets_saved += 1

        await db_session.commit()
        status_str = "success"

    except Exception as e:
        logger.exception("collector_cycle_error")
        error_msg = str(e)
        status_str = "error"
        await db_session.rollback()
    finally:
        await client.close()

    duration = (datetime.now(timezone.utc) - start_time).total_seconds()
    
    # Записываем статистику работы сборщика
    try:
        status_record = CollectorStatus(
            run_at=start_time,
            status=status_str,
            markets_found=markets_found,
            markets_saved=markets_saved,
            error_message=error_msg,
            duration_sec=duration
        )
        db_session.add(status_record)
        await db_session.commit()
    except Exception as e:
        logger.error("error_saving_collector_status", error=str(e))
