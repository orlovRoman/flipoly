from datetime import datetime, timezone
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from polyflip.collector.client import PolymarketClient
from polyflip.db.models import MarketSnapshot, LiveMarket, CollectorStatus, OrderbookDepthSnapshot
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

    from polyflip.crypto.underlying_observations import get_observation_writer
    obs_writer = get_observation_writer()

    try:
        # 1. Получаем список 15-минутных рынков для наших активов
        active_markets = await client.get_active_15m_markets(settings.asset_list)
        markets_found = len(active_markets)
        logger.info("found_active_markets", count=markets_found)

        for m_data in active_markets:
            market_id = m_data["market_id"]
            yes_token_id = m_data["yes_token_id"]
            no_token_id = m_data.get("no_token_id")
            
            # Fetch real orderbooks for BOTH sides (Points 6 & 7)
            prices = await client.get_market_prices(yes_token_id, no_token_id=no_token_id, market_id=market_id)
            if not prices or "error" in prices:
                continue

            mid_price = prices.get("current_yes_price")
            spread = prices.get("current_spread")

            # Вычисляем time_left_min
            current_time = datetime.now(timezone.utc)
            end_date = datetime.fromisoformat(m_data["end_date_iso"].replace("Z", "+00:00"))
            time_left_min = (end_date - current_time).total_seconds() / 60.0

            if time_left_min < 0:
                continue # Рынок уже закрылся

            start_date_iso = m_data.get("start_date_iso")
            start_date = None
            if start_date_iso:
                try:
                    start_date = datetime.fromisoformat(start_date_iso.replace("Z", "+00:00"))
                except Exception:
                    start_date = None

            prov = m_data.get("strike_provenance")
            if start_date is None and prov and getattr(prov, "market_start_at", None):
                start_date = prov.market_start_at

            settlement_src = m_data.get("settlement_source") or getattr(prov, "settlement_price_source", "CHAINLINK_ORACLE")

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

            strike_val = getattr(prov, "strike_value", None) if prov else m_data.get("underlying_price")
            strike_src = getattr(prov, "strike_source", None) if prov else ("UNKNOWN" if strike_val is not None else None)
            strike_eff = getattr(prov, "strike_effective_at", None) if prov else None
            strike_rec = getattr(prov, "strike_received_at", None) if prov else current_time

            # Retrieve real-time spot & oracle prices from ObservationWriter
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
                if mid_price is not None and live_m.current_yes_price is not None:
                    price_velocity = mid_price - live_m.current_yes_price
                
                # Обновляем LiveMarket
                if mid_price is not None:
                    live_m.current_yes_price = mid_price
                    live_m.current_no_price = prices.get("current_no_price")
                    live_m.current_spread = spread
                    live_m.price_velocity = price_velocity
                    live_m.last_updated = current_time

                live_m.volume_5min = vol_val
                live_m.volume_status = vol_status
                if live_m.last_updated is None or (mid_price is None and live_m.last_updated < current_time):
                    # update last_updated even if no price, to reflect recent activity poll? 
                    # The user said "пропускается обновление цены", volume can still be updated
                    live_m.last_updated = current_time
                live_m.yes_token_id = yes_token_id
                live_m.no_token_id = m_data["no_token_id"]
                if getattr(live_m, "underlying_price", None) is None:
                    live_m.underlying_price = strike_val
                if getattr(live_m, "strike_value", None) is None:
                    live_m.strike_value = strike_val
                    live_m.strike_source = strike_src
                    live_m.strike_effective_at = strike_eff
                    live_m.strike_received_at = strike_rec
                    live_m.strike_observed_at = strike_rec
                if getattr(live_m, "market_start_at", None) is None and start_date:
                    live_m.market_start_at = start_date
                if getattr(live_m, "market_end_at", None) is None and end_date:
                    live_m.market_end_at = end_date
                if getattr(live_m, "settlement_price_source", None) is None:
                    live_m.settlement_price_source = settlement_src
                if binance_price is not None:
                    live_m.binance_price = binance_price
                if oracle_price is not None:
                    live_m.oracle_price = oracle_price
            elif mid_price is not None:
                # Создаем новую запись в LiveMarket
                live_m = LiveMarket(
                    market_id=market_id,
                    asset=m_data["asset"],
                    question=m_data["question"],
                    yes_token_id=yes_token_id,
                    no_token_id=m_data["no_token_id"],
                    end_time_est=end_date,
                    current_yes_price=mid_price,
                    current_no_price=prices.get("current_no_price"),
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
                    strike_observed_at=strike_rec,
                    market_start_at=start_date,
                    market_end_at=end_date,
                    settlement_price_source=settlement_src,
                    binance_price=binance_price,
                    oracle_price=oracle_price,
                )
                db_session.add(live_m)

            # 4. Сохраняем Snapshot
            snapshot = None
            if mid_price is not None and spread is not None:
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
                    flip_vs_final=False,
                    recorded_at=current_time,
                    strike_value=strike_val,
                    strike_source=strike_src,
                    strike_effective_at=strike_eff,
                    strike_received_at=strike_rec,
                    strike_observed_at=strike_rec,
                    market_start_at=start_date,
                    market_end_at=end_date,
                    settlement_price_source=settlement_src,
                    binance_price=binance_price,
                    oracle_price=oracle_price,
                )
                db_session.add(snapshot)
                await db_session.flush()

            # 5. Сохраняем OrderbookDepthSnapshot для YES и NO (Point 6)
            yb = prices.get("yes_orderbook")
            if yb:
                db_session.add(
                    OrderbookDepthSnapshot(
                        snapshot_id=snapshot.id if snapshot else None,
                        market_id=market_id,
                        token_id=yes_token_id,
                        outcome_side="YES",
                        event_at=yb.event_at,
                        received_at=yb.received_at,
                        bids=yb.bids,
                        asks=yb.asks,
                        sequence_id=yb.sequence_id,
                        is_truncated=yb.is_truncated,
                        depth_limit=yb.depth_limit,
                        source=yb.source,
                        quality_status=yb.quality_status,
                        quality_notes=yb.quality_notes,
                        best_bid_price=yb.best_bid_price,
                        best_bid_size=yb.best_bid_size,
                        best_ask_price=yb.best_ask_price,
                        best_ask_size=yb.best_ask_size,
                        depth_usdc_bid=yb.depth_usdc_bid,
                        depth_usdc_ask=yb.depth_usdc_ask,
                    )
                )

            nb = prices.get("no_orderbook")
            if nb and no_token_id:
                db_session.add(
                    OrderbookDepthSnapshot(
                        snapshot_id=snapshot.id if snapshot else None,
                        market_id=market_id,
                        token_id=no_token_id,
                        outcome_side="NO",
                        event_at=nb.event_at,
                        received_at=nb.received_at,
                        bids=nb.bids,
                        asks=nb.asks,
                        sequence_id=nb.sequence_id,
                        is_truncated=nb.is_truncated,
                        depth_limit=nb.depth_limit,
                        source=nb.source,
                        quality_status=nb.quality_status,
                        quality_notes=nb.quality_notes,
                        best_bid_price=nb.best_bid_price,
                        best_bid_size=nb.best_bid_size,
                        best_ask_price=nb.best_ask_price,
                        best_ask_size=nb.best_ask_size,
                        depth_usdc_bid=nb.depth_usdc_bid,
                        depth_usdc_ask=nb.depth_usdc_ask,
                    )
                )

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
            service_name="collector",
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
        try:
            await db_session.rollback()
        except Exception:
            pass
