"""Merge active AI Lab overlays into effective PAPER runtime settings."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.ai_lab.agent_tools import expire_overlays
from polyflip.db.models import AIConfigOverlay, TradeHistory

_PAPER_TARGETS = {"PAPER", "PAPER_TRADING", "SHADOW_SIMULATION"}
_TERMINAL_TRADE_STATUSES = ("FILLED", "PAPER_FILLED", "CLOSED", "RESOLVED")


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _normalize_asset(value: Any) -> str:
    text = str(value or "").strip().upper().replace("/", "").replace("-", "")
    return text.removesuffix("USDT")


async def get_active_paper_overlays(
    session: AsyncSession,
    *,
    now: datetime | None = None,
    asset: str | None = None,
    regime: str | None = None,
) -> list[AIConfigOverlay]:
    cutoff = _as_utc(now or datetime.now(timezone.utc))
    await expire_overlays(session, now=cutoff)
    requested_asset = _normalize_asset(asset)
    requested_regime = str(regime or "").strip().lower()
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
        scoped_asset = _normalize_asset(scope.get("asset"))
        if requested_asset and scoped_asset and scoped_asset != requested_asset:
            continue
        scoped_regime = str(scope.get("regime") or "").strip().lower()
        # A regime-scoped overlay must never leak into an unclassified market;
        # callers need to provide the matching regime explicitly.
        if scoped_regime and (
            not requested_regime or scoped_regime != requested_regime
        ):
            continue
        overlays.append(overlay)
    return overlays


async def resolve_paper_runtime_settings(
    session: AsyncSession,
    base: dict[str, Any],
    *,
    now: datetime | None = None,
    asset: str | None = None,
    regime: str | None = None,
) -> tuple[dict[str, str], list[int]]:
    effective = {str(key): str(value) for key, value in base.items()}
    overlays = await get_active_paper_overlays(
        session, now=now, asset=asset, regime=regime
    )
    applied_ids: list[int] = []
    for overlay in overlays:
        changes = overlay.changes if isinstance(overlay.changes, dict) else {}
        effective.update({str(key): str(value) for key, value in changes.items()})
        applied_ids.append(int(overlay.id))
    return effective, applied_ids


def _trade_value(
    trade: TradeHistory | Mapping[str, Any], key: str, default: Any = None
) -> Any:
    """Read a field from an ORM object or a streamed row mapping."""
    if isinstance(trade, Mapping):
        return trade.get(key, default)
    mapping = getattr(trade, "_mapping", None)
    if mapping is not None:
        return mapping.get(key, default)
    return getattr(trade, key, default)


def _trade_overlay_ids(trade: TradeHistory | Mapping[str, Any]) -> set[int]:
    """Read explicit overlay IDs, with a backwards-compatible snapshot fallback."""
    raw = _trade_value(trade, "ai_lab_overlay_ids")
    if raw is None:
        snapshot = _trade_value(trade, "config_snapshot")
        try:
            payload = json.loads(snapshot) if isinstance(snapshot, str) else snapshot
        except (TypeError, ValueError):
            payload = None
        if isinstance(payload, dict):
            context = payload.get("_trade_context")
            details = (
                context.get("decision_details") if isinstance(context, dict) else None
            )
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
    if not overlays:
        return []

    trade_time_expr = func.coalesce(TradeHistory.created_at, TradeHistory.timestamp)

    def asset_values(asset: str) -> tuple[str, ...]:
        normalized = _normalize_asset(asset)
        if not normalized:
            return ()
        return tuple(
            {
                normalized,
                f"{normalized}USDT",
                f"{normalized}/USDT",
                f"{normalized}-USDT",
            }
        )

    def trade_conditions(
        *,
        asset: str,
        terminal_only: bool = False,
        before: datetime | None = None,
    ) -> list[Any]:
        conditions: list[Any] = [TradeHistory.mode == "PAPER"]
        values = asset_values(asset)
        if values:
            conditions.append(func.upper(TradeHistory.asset).in_(values))
        if terminal_only:
            conditions.append(
                func.upper(TradeHistory.status).in_(_TERMINAL_TRADE_STATUSES)
            )
        if before is not None:
            conditions.append(trade_time_expr < before)
        return conditions

    async def count_paper_trades(asset: str) -> int:
        count = await session.scalar(
            select(func.count(TradeHistory.id)).where(*trade_conditions(asset=asset))
        )
        return int(count or 0)

    async def aggregate_before(
        asset: str, created_at: datetime | None
    ) -> tuple[int, float]:
        count, total_pnl = (
            await session.execute(
                select(
                    func.count(TradeHistory.id),
                    func.coalesce(func.sum(TradeHistory.pnl), 0.0),
                ).where(
                    *trade_conditions(
                        asset=asset,
                        terminal_only=True,
                        before=created_at,
                    )
                )
            )
        ).one()
        return int(count or 0), float(total_pnl or 0.0)

    overlay_by_id = {int(overlay.id): overlay for overlay in overlays}
    overlay_assets = {
        int(overlay.id): _normalize_asset(
            (overlay.scope or {}).get("asset")
            if isinstance(overlay.scope, dict)
            else ""
        )
        for overlay in overlays
    }
    earliest_created_at = min(
        (
            _as_utc(overlay.created_at)
            for overlay in overlays
            if overlay.created_at is not None
        ),
        default=None,
    )
    after_counts: defaultdict[int, int] = defaultdict(int)
    after_pnl: defaultdict[int, float] = defaultdict(float)

    # Stream only terminal PAPER rows at or after the oldest overlay. This
    # keeps the compatibility scan for legacy config snapshots bounded and
    # avoids materialising the entire trade history in application memory.
    after_query = select(
        TradeHistory.asset.label("asset"),
        TradeHistory.status.label("status"),
        TradeHistory.pnl.label("pnl"),
        TradeHistory.created_at.label("created_at"),
        TradeHistory.timestamp.label("timestamp"),
        TradeHistory.ai_lab_overlay_ids.label("ai_lab_overlay_ids"),
        TradeHistory.config_snapshot.label("config_snapshot"),
    ).where(
        TradeHistory.mode == "PAPER",
        func.upper(TradeHistory.status).in_(_TERMINAL_TRADE_STATUSES),
        or_(
            TradeHistory.ai_lab_overlay_ids.is_not(None),
            TradeHistory.config_snapshot.is_not(None),
        ),
    )
    if earliest_created_at is not None:
        after_query = after_query.where(trade_time_expr >= earliest_created_at)
    stream = await session.stream(after_query)
    try:
        async for row in stream:
            trade = row._mapping
            trade_asset = _normalize_asset(trade.get("asset"))
            for overlay_id in _trade_overlay_ids(trade):
                overlay = overlay_by_id.get(overlay_id)
                if overlay is None:
                    continue
                scoped_asset = overlay_assets[overlay_id]
                if scoped_asset and scoped_asset != trade_asset:
                    continue
                after_counts[overlay_id] += 1
                after_pnl[overlay_id] += float(trade.get("pnl") or 0.0)
    finally:
        await stream.close()

    result: list[dict[str, Any]] = []
    for overlay in overlays:
        scope = overlay.scope if isinstance(overlay.scope, dict) else {}
        asset = _normalize_asset(scope.get("asset"))
        created_at = _as_utc(overlay.created_at) if overlay.created_at else None
        before_count, before_pnl = await aggregate_before(asset, created_at)
        total = await count_paper_trades(asset)
        overlay_id = int(overlay.id)
        after_count = after_counts[overlay_id]
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
                    "before": {
                        "trade_count": before_count,
                        "pnl": round(before_pnl, 8),
                    },
                    "after": {
                        "trade_count": after_count,
                        "pnl": round(after_pnl[overlay_id], 8),
                        "coverage": round(after_count / total, 6) if total else 0.0,
                    },
                    "paper_trade_count": total,
                },
            }
        )
    return result
