"""Merge active AI Lab overlays into effective PAPER runtime settings."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.db.models import AIConfigOverlay

_PAPER_TARGETS = {"PAPER", "PAPER_TRADING", "SHADOW_SIMULATION"}


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


async def get_active_paper_overlays(
    session: AsyncSession,
    *,
    now: datetime | None = None,
) -> list[AIConfigOverlay]:
    cutoff = _as_utc(now or datetime.now(timezone.utc))
    result = await session.execute(
        select(AIConfigOverlay)
        .where(
            AIConfigOverlay.status == "APPLIED",
            or_(
                AIConfigOverlay.expires_at.is_(None),
                AIConfigOverlay.expires_at > cutoff,
            ),
        )
        .order_by(AIConfigOverlay.id)
    )
    overlays = []
    for overlay in result.scalars().all():
        scope = overlay.scope if isinstance(overlay.scope, dict) else {}
        target = str(scope.get("target") or "").strip().upper()
        if target and target not in _PAPER_TARGETS:
            continue
        overlays.append(overlay)
    return overlays


async def resolve_paper_runtime_settings(
    session: AsyncSession,
    base: dict[str, Any],
    *,
    now: datetime | None = None,
) -> tuple[dict[str, str], list[int]]:
    effective = {str(key): str(value) for key, value in base.items()}
    overlays = await get_active_paper_overlays(session, now=now)
    applied_ids: list[int] = []
    for overlay in overlays:
        changes = overlay.changes if isinstance(overlay.changes, dict) else {}
        effective.update({str(key): str(value) for key, value in changes.items()})
        applied_ids.append(int(overlay.id))
    return effective, applied_ids
