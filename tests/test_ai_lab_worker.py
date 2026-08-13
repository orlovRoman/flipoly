import asyncio

import pytest

from polyflip.ai_lab import lgbm_worker


def test_lgbm_worker_rejects_unbounded_batch():
    with pytest.raises(ValueError, match="between 1 and 10"):
        asyncio.run(
            lgbm_worker.execute_lgbm_steps(
                object(),
                1,
                max_steps=lgbm_worker.MAX_LGBM_WORKER_STEPS + 1,
            )
        )


def test_lgbm_worker_rejects_zero_batch():
    with pytest.raises(ValueError, match="between 1 and 10"):
        asyncio.run(
            lgbm_worker.execute_lgbm_steps(object(), 1, max_steps=0)
        )
