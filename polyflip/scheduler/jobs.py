import asyncio
import signal
import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from polyflip.collector.parser import run_collector_cycle
from polyflip.collector.resolver import resolve_pending_markets
from polyflip.trading.engine import trade_worker_cycle
from polyflip.trading.trader import PolyTrader
from polyflip.collector.client import PolymarketClient
from polyflip.db.connection import async_session
from polyflip.config import settings
from polyflip.db.models import RuntimeSettings
from sqlalchemy import select
import os
import subprocess
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse
from pathlib import Path

logger = structlog.get_logger(__name__)

async def collector_job():
    logger.info("starting_collector_job")
    async with async_session() as session:
        await run_collector_cycle(session)
    logger.info("finished_collector_job")

async def resolver_job():
    logger.info("starting_resolver_job")
    async with async_session() as session:
        await resolve_pending_markets(session)
    logger.info("finished_resolver_job")

async def trade_job(trader, api_client):
    async with async_session() as session:
        await trade_worker_cycle(session, trader, api_client)

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
    from polyflip.models.trainer import ModelTrainer
    try:
        async with async_session() as session:
            trainer = ModelTrainer(session)
            for asset in settings.asset_list:
                await trainer.train_model(asset)
        logger.info("finished_retrain_job")
    except Exception as e:
        logger.exception("retrain_job_error", error=str(e))

async def resolve_trades_job():
    logger.info("starting_resolve_trades_job")
    from polyflip.db.models import TradeHistory, MarketSnapshot
    from sqlalchemy import and_
    try:
        async with async_session() as session:
            # Ищем сделки без PnL, которые были SUCCESS или PAPER
            stmt = select(TradeHistory).where(
                and_(TradeHistory.pnl.is_(None), TradeHistory.status == "SUCCESS")
            )
            trades = (await session.execute(stmt)).scalars().all()
            
            if not trades:
                return
                
            market_ids = list(set([t.market_id for t in trades]))
            outcomes_stmt = select(MarketSnapshot.market_id, MarketSnapshot.final_outcome).where(
                and_(MarketSnapshot.market_id.in_(market_ids), MarketSnapshot.final_outcome != "PENDING")
            )
            outcomes = (await session.execute(outcomes_stmt)).all()
            market_outcomes = {r.market_id: r.final_outcome for r in outcomes}
            
            for t in trades:
                outcome = market_outcomes.get(t.market_id)
                if outcome:
                    is_win = (t.outcome_bought == outcome)
                    if is_win:
                        if t.executed_price > 0:
                            t.pnl = (t.amount_usdc / t.executed_price) - t.amount_usdc
                        else:
                            t.pnl = 0.0
                    else:
                        t.pnl = -t.amount_usdc
                        
            await session.commit()
            logger.info("finished_resolve_trades_job", resolved=len(trades))
    except Exception as e:
        logger.exception("resolve_trades_job_error", error=str(e))

async def cleanup_job():
    logger.info("starting_cleanup_job")
    from polyflip.db.models import CollectorStatus
    from sqlalchemy import delete
    try:
        async with async_session() as session:
            threshold = datetime.now(timezone.utc) - timedelta(days=7)
            stmt = delete(CollectorStatus).where(CollectorStatus.run_at < threshold)
            result = await session.execute(stmt)
            await session.commit()
            logger.info("finished_cleanup_job", deleted_rows=result.rowcount)
    except Exception as e:
        logger.exception("cleanup_job_error", error=str(e))

async def main():
    logger.info("scheduler_starting", interval=settings.LIVE_POLL_INTERVAL_SECONDS)
    
    # Инициализируем общие клиенты для переиспользования соединений
    trader = PolyTrader()
    api_client = PolymarketClient()
    
    scheduler = AsyncIOScheduler()
    
    scheduler.add_job(
        collector_job,
        trigger=IntervalTrigger(seconds=settings.LIVE_POLL_INTERVAL_SECONDS),
        id="collector_job",
        replace_existing=True
    )
    
    # Запускаем резолвер каждые 2 минуты (120 сек)
    scheduler.add_job(
        resolver_job,
        trigger=IntervalTrigger(seconds=120),
        id="resolver_job",
        replace_existing=True
    )
    
    # Ежедневно переобучаем модели (раз в 24 часа)
    scheduler.add_job(
        retrain_job,
        trigger=IntervalTrigger(hours=settings.RETRAIN_INTERVAL_HOURS),
        id="retrain_job",
        replace_existing=True
    )
    
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
    
    # Запускаем торговый движок каждые 5 секунд с передачей общих клиентов
    scheduler.add_job(
        trade_job,
        trigger=IntervalTrigger(seconds=5),
        id="trade_job",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=3,
        kwargs={"trader": trader, "api_client": api_client}
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
