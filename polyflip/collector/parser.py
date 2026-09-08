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
            
            prices = await client.get_market_prices(yes_token_id)
            if not prices or "error" in prices:
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
            vol_val = 0.0
            vol_status = "UNKNOWN"
            if hasattr(volume_5min, "volume"):
                vol_val = float(volume_5min.volume) if volume_5min.volume is not None else 0.0
                vol_status = getattr(volume_5min, "status", "VALID")
            elif volume_5min is not None:
                try:
                    vol_val = float(volume_5min)
                    vol_status = "VALID"
                except (TypeError, ValueError):
                    vol_val = 0.0
                    vol_status = "UNKNOWN"

            prov = m_data.get("strike_provenance")
            strike_val = getattr(prov, "strike_value", None) if prov else m_data.get("underlying_price")
            strike_src = getattr(prov, "strike_source", None) if prov else ("UNKNOWN" if strike_val is not None else None)
            strike_eff = getattr(prov, "strike_effective_at", None) if prov else None
            strike_rec = getattr(prov, "strike_received_at", None) if prov else current_time

            # Retrieve real-time spot & oracle prices from ObservationWriter
            from polyflip.crypto.underlying_observations import get_observation_writer
            obs_writer = get_observation_writer()
            binance_price = obs_writer.get_latest_cached_price(m_data["asset"], "BINANCE")
            oracle_price = obs_writer.get_latest_cached_price(m_data["asset"], "ORACLE")
            if oracle_price is None and strike_val is not None:
                oracle_price = float(strike_val)

            if strike_val is not None and strike_val > 0:
                obs_writer.record_tick(
                    instrument=m_data["asset"],
                    price=float(strike_val),
                    source="ORACLE",
                    event_at=strike_eff or current_time,
                    received_at=strike_rec,
                    extra_data={"market_id": market_id, "strike_source": strike_src},
                )

            price_velocity = 0.0

            if live_m:
                # Считаем дельту скорости цены
                price_velocity = mid_price - live_m.current_yes_price
                
                # Обновляем LiveMarket
                live_m.current_yes_price = mid_price
                live_m.current_no_price = prices["current_no_price"]
                live_m.current_spread = spread
                live_m.price_velocity = price_velocity
                live_m.volume_5min = vol_val
                live_m.volume_status = vol_status
                live_m.last_updated = current_time
                # На всякий случай обновляем token_id, если добавились
                live_m.yes_token_id = yes_token_id
                live_m.no_token_id = m_data["no_token_id"]
                if getattr(live_m, "underlying_price", None) is None:
                    live_m.underlying_price = strike_val
                if getattr(live_m, "strike_value", None) is None:
                    live_m.strike_value = strike_val
                    live_m.strike_source = strike_src
                    live_m.strike_effective_at = strike_eff
                    live_m.strike_received_at = strike_rec
                if binance_price is not None:
                    live_m.binance_price = binance_price
                if oracle_price is not None:
                    live_m.oracle_price = oracle_price
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
                    volume_5min=vol_val,
                    volume_status=vol_status,
                    price_velocity=0.0,
                    last_updated=current_time,
                    underlying_price=strike_val,
                    strike_value=strike_val,
                    strike_source=strike_src,
                    strike_effective_at=strike_eff,
                    strike_received_at=strike_rec,
                    binance_price=binance_price,
                    oracle_price=oracle_price,
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
                best_bid=prices.get("best_bid"),
                best_ask=prices.get("best_ask"),
                volume_5min=vol_val,
                volume_status=vol_status,
                price_velocity=price_velocity,
                hour_of_day=current_time.hour,
                final_outcome="PENDING",
                flip_vs_final=False, # Обновится позже
                recorded_at=current_time,
                strike_value=strike_val,
                strike_source=strike_src,
                strike_effective_at=strike_eff,
                strike_received_at=strike_rec,
                binance_price=binance_price,
                oracle_price=oracle_price,
            )
            db_session.add(snapshot)
            markets_saved += 1

        # Flush pending observations to db
        await obs_writer.flush(session=db_session)
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
