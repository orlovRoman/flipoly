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
import os
import subprocess
from datetime import datetime, timezone
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

async def trade_job():
    async with async_session() as session:
        await trade_worker_cycle(session)

async def backup_job():
    logger.info("starting_backup_job")
    try:
        backup_dir = "/app/backups"
        os.makedirs(backup_dir, exist_ok=True)
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(backup_dir, f"backup_polyflip_{timestamp}.sql")
        
        db_url = os.environ.get("DATABASE_URL", "")
        pg_url = db_url.replace("+asyncpg", "")
        
        parsed = urlparse(pg_url)
        env = os.environ.copy()
        env["PGPASSWORD"] = parsed.password or ""
        
        cmd = [
            "pg_dump",
            "-h", parsed.hostname or "",
            "-U", parsed.username or "",
            "-d", (parsed.path or "").lstrip("/"),
            "-f", filepath,
            "-F", "c"
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
    
    # Ежедневный бэкап базы данных (раз в 24 часа)
    scheduler.add_job(
        backup_job,
        trigger=IntervalTrigger(hours=24),
        id="backup_job",
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
