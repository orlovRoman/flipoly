"""Durable worker for resource-intensive LightGBM training jobs.

Training must not run in the FastAPI process: a large feature build can consume
all available memory and stall Uvicorn while the host starts swapping.  The
API only creates a row in ``lgbm_training_jobs``; this process claims and runs
one job at a time.
"""

from __future__ import annotations

import asyncio
import os
import traceback
from datetime import datetime, timezone

import structlog
from sqlalchemy import select, update

from polyflip.crypto.trainer import CryptoModelTrainer
from polyflip.db.connection import async_session, engine
from polyflip.db.models import LGBMExperimentConfig, LGBMTrainingJob

logger = structlog.get_logger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _requeue_interrupted_jobs() -> None:
    async with async_session() as session:
        await session.execute(
            update(LGBMTrainingJob)
            .where(LGBMTrainingJob.status == "RUNNING")
            .values(
                status="QUEUED",
                started_at=None,
                worker_pid=None,
                error="requeued after training worker restart",
                error_traceback=None,
            )
        )
        await session.commit()


async def _claim_next_job() -> dict | None:
    async with async_session() as session:
        result = await session.execute(
            select(LGBMTrainingJob)
            .where(LGBMTrainingJob.status == "QUEUED")
            .order_by(LGBMTrainingJob.created_at, LGBMTrainingJob.id)
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        job = result.scalar_one_or_none()
        if job is None:
            return None
        job.status = "RUNNING"
        job.started_at = _now()
        job.worker_pid = os.getpid()
        job.error = None
        job.error_traceback = None
        await session.commit()
        return {
            "id": job.id,
            "symbol": job.symbol,
            "interval": job.interval,
            "feature_set": job.feature_set,
            "activate_after_train": bool(job.activate_after_train),
            "experiment_config_id": job.experiment_config_id,
        }


async def _finish_job(
    job_id: int,
    *,
    success: bool,
    result: dict | None = None,
    error: str | None = None,
    error_traceback: str | None = None,
) -> None:
    async with async_session() as session:
        await session.execute(
            update(LGBMTrainingJob)
            .where(LGBMTrainingJob.id == job_id)
            .values(
                status="SUCCESS" if success else "FAILED",
                finished_at=_now(),
                result=result,
                error=error,
                error_traceback=error_traceback,
                worker_pid=None,
            )
        )
        await session.commit()


async def _run_job(payload: dict) -> None:
    job_id = int(payload["id"])
    try:
        experiment_config = None
        config_id = payload.get("experiment_config_id")
        if config_id is not None:
            async with async_session() as session:
                config = await session.get(LGBMExperimentConfig, int(config_id))
                if config is None or config.is_archived:
                    raise ValueError("experiment config not found or archived")
                experiment_config = {
                    "feature_set": config.feature_set,
                    "model": config.model_params,
                    "calibration": config.calibration_params,
                    "thresholds": config.threshold_params,
                    "backtest": config.backtest_params,
                }

        async with async_session() as session:
            trainer = CryptoModelTrainer(session)
            ok = await trainer.train(
                payload["symbol"],
                payload["interval"],
                feature_set=(experiment_config or {}).get("feature_set", payload["feature_set"]),
                activate_after_train=payload["activate_after_train"],
                experiment_config=experiment_config,
                experiment_config_id=config_id,
            )
        await _finish_job(
            job_id,
            success=bool(ok),
            result={"success": bool(ok)},
            error=None if ok else "trainer returned unsuccessful result",
        )
        logger.info("lgbm_training_job_finished", job_id=job_id, success=bool(ok))
    except Exception as exc:
        error_traceback = traceback.format_exc()
        logger.exception("lgbm_training_job_failed", job_id=job_id, error=str(exc))
        await _finish_job(
            job_id,
            success=False,
            error=str(exc),
            error_traceback=error_traceback,
        )


async def run_worker() -> None:
    poll_interval = max(0.5, float(os.getenv("LGBM_TRAINING_POLL_INTERVAL_SEC", "2")))
    await _requeue_interrupted_jobs()
    logger.info("lgbm_training_worker_started", pid=os.getpid(), poll_interval=poll_interval)
    try:
        while True:
            payload = await _claim_next_job()
            if payload is None:
                await asyncio.sleep(poll_interval)
                continue
            await _run_job(payload)
    finally:
        await engine.dispose()


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
