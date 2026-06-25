import asyncio
import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from polyflip.collector.parser import run_collector_cycle
from polyflip.collector.resolver import resolve_pending_markets
from polyflip.trading.engine import trade_worker_cycle
from polyflip.db.connection import async_session
from polyflip.config import settings
from polyflip.db.models import RuntimeSettings
from sqlalchemy import select

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

async def trade_job():
    async with async_session() as session:
        await trade_worker_cycle(session)

async def main():
    logger.info("scheduler_starting", interval=settings.LIVE_POLL_INTERVAL_SECONDS)
    
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
    
    # Запускаем торговый движок каждые 5 секунд
    scheduler.add_job(
        trade_job,
        trigger=IntervalTrigger(seconds=5),
        id="trade_job",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=3
    )
    
    scheduler.start()
    
    # Ждем вечно (пока процесс не убьют)
    while True:
        await asyncio.sleep(3600)

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
