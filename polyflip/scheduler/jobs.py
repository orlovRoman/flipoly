import asyncio
import signal
import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from polyflip.collector.parser import run_collector_cycle
from polyflip.collector.resolver import resolve_pending_markets
from polyflip.trading.engine import trade_worker_cycle
from polyflip.trading.stoploss_worker import stoploss_worker_cycle
from polyflip.trading.takeprofit_worker import takeprofit_worker_cycle
from polyflip.collector.client import PolymarketClient
from polyflip.collector.client import PolymarketClient
from polyflip.db.connection import async_session
from sqlalchemy.ext.asyncio import AsyncSession
from polyflip.config import settings
from sqlalchemy import select, and_, delete, func
import os
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse
from pathlib import Path

from polyflip.db.models import RuntimeSettings, TradeHistory, MarketSnapshot, CollectorStatus
from polyflip.models.trainer import ModelTrainer
from polyflip.crypto.candle_collector import collect_new_candles, refresh_funding_rates
from polyflip.crypto.candle_pruner import prune_old_candles
from polyflip.crypto.historical_loader import load_history_all


logger = structlog.get_logger(__name__)

async def collector_job():
    logger.info("starting_collector_job")
    async with async_session() as session:
        await run_collector_cycle(session)
    logger.info("finished_collector_job")

async def scheduler_heartbeat_job() -> None:
    try:
        heartbeat_path = Path("/tmp/scheduler_alive")
        heartbeat_path.write_text(
            datetime.now(timezone.utc).isoformat(),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning("failed_to_write_scheduler_health_marker", error=str(e))

async def initial_backfill_job() -> None:
    try:
        from sqlalchemy import select
        from polyflip.db.models import RuntimeSettings
        
        async with async_session() as session:
            stmt = select(RuntimeSettings).where(RuntimeSettings.key == "STARTUP_CANDLE_BACKFILL_ENABLED")
            res = await session.execute(stmt)
            setting = res.scalar_one_or_none()
            if not setting or setting.value.lower() != "true":
                logger.info("startup_backfill_disabled")
                return

            await candle_backfill_job(session)
    except Exception as exc:
        logger.exception("startup_backfill_failed", error=str(exc))

async def resolver_job():
    logger.info("starting_resolver_job")
    async with async_session() as session:
        await resolve_pending_markets(session)
    logger.info("finished_resolver_job")

async def _check_ath_checkpoint(session: AsyncSession):
    try:
        from polyflip.services.preset_service import PresetService
        
        pnl_stmt = select(func.sum(TradeHistory.pnl)).where(
            TradeHistory.status == "SUCCESS",
            TradeHistory.pnl.is_not(None)
        )
        total_pnl = (await session.execute(pnl_stmt)).scalar() or 0.0
        
        init_cap_row = (await session.execute(
            select(RuntimeSettings).where(RuntimeSettings.key == "INITIAL_CAPITAL")
        )).scalar_one_or_none()
        initial_capital = float(init_cap_row.value) if init_cap_row else settings.INITIAL_CAPITAL
        current_capital = initial_capital + total_pnl
        
        ath_preset = await PresetService.check_and_save_ath(
            session, current_capital=current_capital, current_pnl=total_pnl
        )
        if ath_preset:
            logger.info("ath_checkpoint_auto_saved", preset_id=ath_preset.id, name=ath_preset.name, capital=current_capital, pnl=total_pnl)
    except Exception as e:
        logger.warning("ath_checkpoint_check_failed", error=str(e))

async def trade_job(api_client):
    async with async_session() as session:
        await trade_worker_cycle(session, api_client)
        # await _check_ath_checkpoint(session)


async def stoploss_job(api_client):
    try:
        async with async_session() as session:
            await stoploss_worker_cycle(session, api_client)
    except Exception as e:
        logger.exception("stoploss_job_error", error=str(e))

async def takeprofit_job(api_client):
    try:
        async with async_session() as session:
            await takeprofit_worker_cycle(session, api_client)
    except Exception as e:
        logger.exception("takeprofit_worker_failed", error=str(e))


async def backup_job():
    logger.info("starting_backup_job")
    try:
        backup_dir = "/app/backups"
        os.makedirs(backup_dir, exist_ok=True)
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(backup_dir, f"backup_polyflip_{timestamp}.sql")
        
        db_url = os.environ.get("DATABASE_URL", "")
        if not db_url:
            logger.error("backup_job_failed", error="DATABASE_URL is not set")
            return
            
        pg_url = db_url.replace("+asyncpg", "")
        parsed = urlparse(pg_url)
        
        if not parsed.hostname or not parsed.username:
            logger.error("backup_job_failed", error="Invalid DATABASE_URL, missing hostname or username")
            return
            
        env = os.environ.copy()
        env["PGPASSWORD"] = parsed.password or ""
        
        cmd = [
            "pg_dump",
            "-h", parsed.hostname or "",
            "-U", parsed.username or "",
            "-d", (parsed.path or "").lstrip("/"),
            "-f", filepath,
            "-F", "p"
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            logger.info("backup_job_success", filepath=filepath)
            
            # Rotate backups
            backups = sorted(Path(backup_dir).glob("backup_polyflip_*.sql"))
            max_backups = int(os.environ.get("MAX_BACKUPS", "7"))
            for old in backups[:-max_backups]:
                old.unlink()
                logger.info("backup_rotated", removed=str(old))
        else:
            logger.error("backup_job_failed", stderr=stderr.decode() if stderr else "unknown error")
            
    except Exception as e:
        logger.exception("backup_job_error", error=str(e))

async def retrain_job():
    logger.info("starting_retrain_job")
    try:
        async with async_session() as session:
            # Получаем список торгуемых активов из БД
            stmt = select(RuntimeSettings).where(RuntimeSettings.key == "TRADE_ASSETS")
            res = await session.execute(stmt)
            setting = res.scalar_one_or_none()
            
            if setting and setting.value.strip():
                trade_assets = list(dict.fromkeys(
                    a.strip().upper() for a in setting.value.split(",") if a.strip()
                ))
            else:
                trade_assets = list(dict.fromkeys(
                    a.strip().upper() for a in settings.TRADE_ASSETS.split(",") if a.strip()
                ))

            trainer = ModelTrainer(session)
            for asset in trade_assets:
                await trainer.train_model(asset)
        logger.info("finished_retrain_job")
    except Exception as e:
        logger.exception("retrain_job_error", error=str(e))

async def resolve_trades_job():
    logger.info("starting_resolve_trades_job")
    try:
        from decimal import Decimal
        from polyflip.execution.settlement_service import settle_resolved_position
        from polyflip.execution.states import ACTIVE_POSITION_STATES

        async with async_session() as session:
            # Только PAPER и SHADOW закрываются scheduler-ом автоматически.
            # LIVE-позиции переходят в RESOLVED_REDEEMABLE — требуют ручного redemption
            # через CTF adapter (client.redeem_positions()).
            stmt = select(TradeHistory).where(
                and_(
                    TradeHistory.position_status.in_(ACTIVE_POSITION_STATES),
                    TradeHistory.status.in_(["SUCCESS", "PAPER"]),
                    TradeHistory.mode.in_(("PAPER", "SHADOW")),
                )
            )
            trades = (await session.execute(stmt)).scalars().all()

            # Также находим LIVE-позиции для перевода в RESOLVED_REDEEMABLE
            live_stmt = select(TradeHistory).where(
                and_(
                    TradeHistory.mode == "LIVE",
                    TradeHistory.position_status.in_(["OPEN", "PARTIALLY_CLOSED"]),
                    TradeHistory.status.in_(["SUCCESS"]),
                    TradeHistory.entry_filled_shares > 0,
                    TradeHistory.remaining_shares > 0,
                )
            )
            live_trades = (await session.execute(live_stmt)).scalars().all()

            all_trade_market_ids = list(
                {t.market_id for t in trades} | {t.market_id for t in live_trades}
            )

            if not all_trade_market_ids:
                return

            outcomes_stmt = select(
                MarketSnapshot.market_id, MarketSnapshot.final_outcome
            ).where(
                and_(
                    MarketSnapshot.market_id.in_(all_trade_market_ids),
                    MarketSnapshot.final_outcome != "PENDING",
                )
            )
            outcomes = (await session.execute(outcomes_stmt)).all()
            market_outcomes = {r.market_id: r.final_outcome for r in outcomes}

            resolved_count = 0

            # PAPER/SHADOW: полное закрытие через settle_resolved_position
            for t in trades:
                raw_outcome = market_outcomes.get(t.market_id)
                if not raw_outcome:
                    continue

                if raw_outcome == "INVALID":
                    # Polymarket: INVALID → каждый бинарный токен = $0.50
                    await settle_resolved_position(
                        session,
                        trade_id=t.id,
                        winning_outcome="INVALID",
                        payout_per_share=Decimal("0.5"),
                        settlement_fee_usdc=Decimal("0"),
                    )
                    t.status = "INVALID"
                else:
                    # Polymarket fee взимается при match, не при разрешении рынка.
                    # settlement_fee_usdc = 0.
                    await settle_resolved_position(
                        session,
                        trade_id=t.id,
                        winning_outcome=raw_outcome,
                        payout_per_share=Decimal("1"),
                        settlement_fee_usdc=Decimal("0"),
                    )

                resolved_count += 1

            # LIVE: переводим в RESOLVED_REDEEMABLE — оператор должен запустить
            # scripts/setup_approvals.py → client.redeem_positions()
            for t in live_trades:
                raw_outcome = market_outcomes.get(t.market_id)
                if not raw_outcome:
                    continue
                if Decimal(str(t.entry_filled_shares or 0)) <= 0:
                    logger.error(
                        "resolver_skipped_zero_fill_trade",
                        trade_id=t.id,
                    )
                    continue
                if t.position_status not in ("RESOLVED_REDEEMABLE",):
                    t.position_status = "RESOLVED_REDEEMABLE"
                    logger.info(
                        "live_position_resolved_redeemable",
                        trade_id=t.id,
                        winning_outcome=raw_outcome,
                    )
                    resolved_count += 1

            await session.commit()
            logger.info("finished_resolve_trades_job", resolved=resolved_count)
    except Exception as e:
        logger.exception("resolve_trades_job_error", error=str(e))



async def cleanup_job():
    logger.info("starting_cleanup_job")
    try:
        async with async_session() as session:
            threshold = datetime.now(timezone.utc) - timedelta(days=7)
            stmt = delete(CollectorStatus).where(CollectorStatus.run_at < threshold)
            result = await session.execute(stmt)
            await session.commit()
            logger.info("finished_cleanup_job", deleted_rows=result.rowcount)
    except Exception as e:
        logger.exception("cleanup_job_error", error=str(e))

async def candle_collector_job():
    logger.info("starting_candle_collector_job")
    try:
        async with async_session() as session:
            results = await collect_new_candles(session)
        logger.info("finished_candle_collector_job", results=results)
    except Exception as e:
        logger.exception("candle_collector_job_error", error=str(e))


async def candle_backfill_job(session: AsyncSession) -> None:
    from polyflip.crypto.candle_repository import get_latest_open_time, has_incomplete_candles
    from polyflip.crypto.historical_loader import load_history_all, DEFAULT_SYMBOLS

    symbols_to_backfill = []
    for symbol in DEFAULT_SYMBOLS:
        latest = await get_latest_open_time(session, symbol, "15m")
        has_incomplete = await has_incomplete_candles(session, symbol, "15m")
        needs = (
            latest is None or
            (datetime.now(timezone.utc) - latest) > timedelta(days=7) or
            has_incomplete
        )
        if needs:
            symbols_to_backfill.append(symbol)

    if symbols_to_backfill:
        logger.info("backfill_triggered", symbols=symbols_to_backfill)
        await load_history_all(session, symbols=symbols_to_backfill)
    else:
        logger.info("backfill_skipped")


async def candle_pruning_job():
    """
    Запускается раз в 24 часа. Удаляет свечи старше retention_days.
    """
    logger.info("starting_candle_pruning_job")
    try:
        async with async_session() as session:
            deleted = await prune_old_candles(session, retention_days=90)
            logger.info("finished_candle_pruning_job", deleted_rows=deleted)
    except Exception as e:
        logger.exception("candle_pruning_job_error", error=str(e))


async def check_settings_job(scheduler):
    try:
        async with async_session() as session:
            # 1. LIVE_POLL_INTERVAL_SECONDS
            stmt = select(RuntimeSettings).where(RuntimeSettings.key == "LIVE_POLL_INTERVAL_SECONDS")
            res = await session.execute(stmt)
            setting = res.scalar_one_or_none()
            if setting:
                new_interval = int(setting.value)
                job = scheduler.get_job("collector_job")
                if job:
                    try:
                        current_interval = job.trigger.interval.total_seconds()
                        if int(current_interval) != new_interval:
                            logger.info("rescheduling_collector_job", new_interval=new_interval)
                            scheduler.reschedule_job(
                                "collector_job",
                                trigger=IntervalTrigger(seconds=new_interval)
                            )
                    except AttributeError:
                        logger.warning("check_settings_job_trigger_has_no_interval_rescheduling", job_id="collector_job", new_interval=new_interval)
            
            # 2. STOP_LOSS_CHECK_SEC
            stmt = select(RuntimeSettings).where(RuntimeSettings.key == "STOP_LOSS_CHECK_SEC")
            res = await session.execute(stmt)
            setting = res.scalar_one_or_none()
            if setting:
                new_interval = int(setting.value)
                job = scheduler.get_job("stoploss_job")
                if job:
                    try:
                        current_interval = job.trigger.interval.total_seconds()
                        if int(current_interval) != new_interval:
                            logger.info("rescheduling_stoploss_job", new_interval=new_interval)
                            scheduler.reschedule_job(
                                "stoploss_job",
                                trigger=IntervalTrigger(seconds=new_interval)
                            )
                    except AttributeError:
                        logger.warning("check_settings_job_trigger_has_no_interval_rescheduling", job_id="stoploss_job", new_interval=new_interval)
            
            # 3. TAKE_PROFIT_CHECK_INTERVAL_SEC
            stmt = select(RuntimeSettings).where(RuntimeSettings.key == "TAKE_PROFIT_CHECK_INTERVAL_SEC")
            res = await session.execute(stmt)
            setting = res.scalar_one_or_none()
            if setting:
                new_interval = int(setting.value)
                job = scheduler.get_job("takeprofit_job")
                if job:
                    try:
                        current_interval = job.trigger.interval.total_seconds()
                        if int(current_interval) != new_interval:
                            logger.info("rescheduling_takeprofit_job", new_interval=new_interval)
                            scheduler.reschedule_job(
                                "takeprofit_job",
                                trigger=IntervalTrigger(seconds=new_interval)
                            )
                    except AttributeError:
                        logger.warning("check_settings_job_trigger_has_no_interval_rescheduling", job_id="takeprofit_job", new_interval=new_interval)
    except Exception as e:
        logger.exception("check_settings_job_error", error=str(e))

async def main():
    poll_interval = settings.LIVE_POLL_INTERVAL_SECONDS
    stoploss_interval = 30
    takeprofit_interval = 30
    try:
        async with async_session() as session:
            stmt = select(RuntimeSettings).where(RuntimeSettings.key == "LIVE_POLL_INTERVAL_SECONDS")
            res = await session.execute(stmt)
            setting = res.scalar_one_or_none()
            if setting:
                poll_interval = int(setting.value)
                
            stmt_sl = select(RuntimeSettings).where(RuntimeSettings.key == "STOP_LOSS_CHECK_SEC")
            res_sl = await session.execute(stmt_sl)
            setting_sl = res_sl.scalar_one_or_none()
            if setting_sl:
                stoploss_interval = int(setting_sl.value)

            stmt_tp = select(RuntimeSettings).where(RuntimeSettings.key == "TAKE_PROFIT_CHECK_INTERVAL_SEC")
            res_tp = await session.execute(stmt_tp)
            setting_tp = res_tp.scalar_one_or_none()
            if setting_tp:
                takeprofit_interval = int(setting_tp.value)
    except Exception as e:
        logger.warning("failed_to_load_initial_intervals", error=str(e))

    logger.info("scheduler_starting", interval=poll_interval, stoploss_interval=stoploss_interval, takeprofit_interval=takeprofit_interval)
    
    # Инициализируем общие клиенты для переиспользования соединений
    # Вызов одноразового backfill свечей и обновления ставок финансирования при старте
    api_client = PolymarketClient()

    scheduler = AsyncIOScheduler()
    
    scheduler.add_job(
        collector_job,
        trigger=IntervalTrigger(seconds=poll_interval),
        id="collector_job",
        replace_existing=True
    )
    
    now = datetime.now(timezone.utc)
    
    scheduler.add_job(
        scheduler_heartbeat_job,
        trigger=IntervalTrigger(seconds=15),
        id="scheduler_heartbeat",
        next_run_time=now,
        max_instances=1,
        replace_existing=True
    )

    scheduler.add_job(
        initial_backfill_job,
        trigger="date",
        run_date=now + timedelta(seconds=10),
        id="initial_candle_backfill",
        max_instances=1,
        replace_existing=True
    )
    
    # Запускаем воркер стоп-лосса с передачей общих клиентов
    scheduler.add_job(
        stoploss_job,
        trigger=IntervalTrigger(seconds=stoploss_interval),
        id="stoploss_job",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=10,
        kwargs={"api_client": api_client}
    )
    
    # Запускаем воркер тейк-профита с передачей общих клиентов
    scheduler.add_job(
        takeprofit_job,
        trigger=IntervalTrigger(seconds=takeprofit_interval),
        id="takeprofit_job",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=10,
        kwargs={"api_client": api_client}
    )
    
    # Проверяем настройки интервала каждые 10 секунд
    scheduler.add_job(
        check_settings_job,
        trigger=IntervalTrigger(seconds=10),
        id="check_settings_job",
        replace_existing=True,
        max_instances=1,
        kwargs={"scheduler": scheduler}
    )
    

    # Запускаем резолвер каждые 2 минуты (120 сек)
    scheduler.add_job(
        resolver_job,
        trigger=IntervalTrigger(seconds=120),
        id="resolver_job",
        replace_existing=True
    )
    
    # Сбор новых криптосвечей каждые 15 минут
    scheduler.add_job(
        candle_collector_job,
        trigger=IntervalTrigger(minutes=15),
        id="candle_collector_job",
        replace_existing=True,
        max_instances=1,
    )
    
    # Ежедневно переобучаем модели (раз в 24 часа) - ОТКЛЮЧЕНО в пользу ручного обучения
    # scheduler.add_job(
    #     retrain_job,
    #     trigger=IntervalTrigger(hours=settings.RETRAIN_INTERVAL_HOURS),
    #     id="retrain_job",
    #     replace_existing=True
    # )
    
    # Расчет PnL для закрытых сделок (каждые 10 минут)
    scheduler.add_job(
        resolve_trades_job,
        trigger=IntervalTrigger(minutes=10),
        id="resolve_trades_job",
        replace_existing=True
    )
    
    # Очистка старых статусов (раз в 24 часа)
    scheduler.add_job(
        cleanup_job,
        trigger=IntervalTrigger(hours=24),
        id="cleanup_job",
        replace_existing=True
    )
    
    # Очистка старых свечей по retention-периоду (раз в 24 часа)
    scheduler.add_job(
        candle_pruning_job,
        trigger=IntervalTrigger(hours=24),
        id="candle_pruning",
        replace_existing=True,
        coalesce=True,
        misfire_grace_time=3600,
    )
    
    # Запускаем торговый движок каждые 5 секунд с передачей общих клиентов
    scheduler.add_job(
        trade_job,
        trigger=IntervalTrigger(seconds=5),
        id="trade_job",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=3,
        kwargs={"api_client": api_client}
    )
    
    # Ежедневный бэкап базы данных (раз в 24 часа)
    scheduler.add_job(
        backup_job,
        trigger=IntervalTrigger(hours=24),
        id="backup_job",
        replace_existing=True
    )
    
    scheduler.start()
    
    shutdown_event = asyncio.Event()
    
    def signal_handler():
        logger.info("shutdown_signal_received")
        shutdown_event.set()
        
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            pass
            
    try:
        await shutdown_event.wait()
    except (KeyboardInterrupt, SystemExit):
        logger.info("interrupted_by_signal")
        
    logger.info("scheduler_stopping")
    scheduler.shutdown(wait=True)
    
    # Закрываем общие сетевые клиенты
    await api_client.close()
    logger.info("scheduler_stopped")

if __name__ == "__main__":
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ]
    )
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
