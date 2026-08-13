"""Worker entry point for bounded, offline LightGBM AI Lab steps.

The caller supplies an already-scoped AsyncSession. This helper does not create
permissions, change active models or expose a live execution path.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.ai_lab.executor import ExecutionOutcome, execute_steps
from polyflip.ai_lab.lgbm_adapters import build_lgbm_adapter_registry


async def execute_lgbm_steps(
    session: AsyncSession,
    run_id: int,
    *,
    max_steps: int = 1,
) -> list[ExecutionOutcome]:
    """Execute at most max_steps queued LightGBM experiment steps."""
    registry = build_lgbm_adapter_registry(session)
    return await execute_steps(session, run_id, registry, max_steps=max_steps)
