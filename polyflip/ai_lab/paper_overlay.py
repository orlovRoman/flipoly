"""Merge active AI Lab overlays into effective PAPER runtime settings."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.db.models import AIConfigOverlay, TradeHistory

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


def _trade_overlay_ids(trade: TradeHistory) -> set[int]:
    """Read explicit overlay IDs, with a backwards-compatible snapshot fallback."""
    raw = getattr(trade, "ai_lab_overlay_ids", None)
    if raw is None:
        snapshot = getattr(trade, "config_snapshot", None)
        try:
            payload = json.loads(snapshot) if isinstance(snapshot, str) else snapshot
        except (TypeError, ValueError):
            payload = None
        if isinstance(payload, dict):
            context = payload.get("_trade_context")
            details = context.get("decision_details") if isinstance(context, dict) else None
            if isinstance(details, dict):
                raw = details.get("ai_lab_overlay_ids", details.get("ai_overlay_ids"))
    if isinstance(raw, (str, int)):
        raw = [raw]
    if not isinstance(raw, (list, tuple, set)):
        return set()
    result: set[int] = set()
    for value in raw:
        try:
            value = int(value)
        except (TypeError, ValueError):
            continue
        if value > 0:
            result.add(value)
    return result


async def get_paper_overlay_runtime_summary(
    session: AsyncSession,
    *,
    run_id: int | None = None,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Return lifecycle and before/after PAPER metrics for overlay dashboard cards."""
    cutoff = _as_utc(now or datetime.now(timezone.utc))
    await expire_overlays(session, now=cutoff)
    query = select(AIConfigOverlay).order_by(AIConfigOverlay.id.desc())
    if run_id is not None:
        query = query.where(AIConfigOverlay.run_id == run_id)
    overlays = (await session.execute(query)).scalars().all()
    trade_rows = (
        (await session.execute(select(TradeHistory).where(TradeHistory.mode == "PAPER")))
        .scalars()
        .all()
    )
    terminal_statuses = {"FILLED", "PAPER_FILLED", "CLOSED", "RESOLVED"}

    def trade_time(trade: TradeHistory) -> datetime:
        return _as_utc(trade.created_at or trade.timestamp)

    def pnl(rows: list[TradeHistory]) -> float:
        return round(sum(float(row.pnl or 0.0) for row in rows), 8)

    result: list[dict[str, Any]] = []
    for overlay in overlays:
        scope = overlay.scope if isinstance(overlay.scope, dict) else {}
        asset = str(scope.get("asset") or "").strip().upper()
        scoped_trades = [
            row
            for row in trade_rows
            if not asset or str(row.asset or "").strip().upper() == asset
        ]
        created_at = _as_utc(overlay.created_at) if overlay.created_at else None
        before = [
            row
            for row in scoped_trades
            if (not created_at or trade_time(row) < created_at)
            and str(row.status or "").upper() in terminal_statuses
        ]
        after = [
            row
            for row in scoped_trades
            if int(overlay.id) in _trade_overlay_ids(row)
            and str(row.status or "").upper() in terminal_statuses
        ]
        total = len(scoped_trades)
        result.append(
            {
                "id": overlay.id,
                "run_id": overlay.run_id,
                "parent_overlay_id": overlay.parent_overlay_id,
                "scope": overlay.scope,
                "changes": overlay.changes,
                "status": overlay.status,
                "created_by": overlay.created_by,
                "expires_at": overlay.expires_at,
                "created_at": overlay.created_at,
                "metrics": {
                    "before": {"trade_count": len(before), "pnl": pnl(before)},
                    "after": {
                        "trade_count": len(after),
                        "pnl": pnl(after),
                        "coverage": round(len(after) / total, 6) if total else 0.0,
                    },
                    "paper_trade_count": total,
                },
            }
        )
    return result
