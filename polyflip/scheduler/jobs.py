import asyncio
import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from polyflip.collector.parser import run_collector_cycle
from polyflip.db.connection import async_session
from polyflip.config import settings
from polyflip.db.models import RuntimeSettings
from sqlalchemy import select

logger = structlog.get_logger(__name__)

async def collector_job():
    logger.info("starting_collector_job")
    async with async_session() as session:
        # Проверяем, есть ли переопределение интервала в БД
        # (в текущей реализации APScheduler мы пока просто используем это для справки,
        # динамическое изменение триггера требует перезапуска джобы, 
        # но для Фазы 2 мы используем config)
        await run_collector_cycle(session)
    logger.info("finished_collector_job")

async def main():
    logger.info("scheduler_starting", interval=settings.LIVE_POLL_INTERVAL_SECONDS)
    
    scheduler = AsyncIOScheduler()
    
    # Запускаем сборщик рынков каждые N секунд (например, 60 секунд для LivePoll)
    # Используем LIVE_POLL_INTERVAL_SECONDS, т.к. 15-минутные рынки 
    # меняются очень быстро, и нам нужен частый поллинг для velocity
    scheduler.add_job(
        collector_job,
        trigger=IntervalTrigger(seconds=settings.LIVE_POLL_INTERVAL_SECONDS),
        id="collector_job",
        replace_existing=True
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
