"""PAPER overlay resolver — merges active AIConfigOverlay rows into effective settings."""
from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.db.models import AIConfigOverlay


async def get_active_paper_overlays(session: AsyncSession) -> list[AIConfigOverlay]:
    result = await session.execute(
        select(AIConfigOverlay).where(AIConfigOverlay.status == "APPLIED").order_by(AIConfigOverlay.id)
    )
    return list(result.scalars().all())


async def resolve_paper_runtime_settings(
    session: AsyncSession,
    base: dict[str, Any],
) -> tuple[dict[str, Any], list[int]]:
    effective = dict(base)
    overlays = await get_active_paper_overlays(session)
    applied_ids: list[int] = []
    for overlay in overlays:
        changes = overlay.changes if isinstance(overlay.changes, dict) else {}
        effective.update(changes)
        applied_ids.append(overlay.id)
    return effective, applied_ids
