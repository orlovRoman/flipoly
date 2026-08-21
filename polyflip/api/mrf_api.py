"""
MRF status API endpoint v3.

Uses MRF telemetry columns (mrf_evaluated, mrf_mode, mrf_phase, etc.)
from DecisionFunnelLog. Per-asset PnL from TradeHistory.

Fixes:
- Uses realized_pnl_usdc with fallback to pnl for legacy rows
- Uses sqlalchemy.case (not func.case)
- Only counts completed trades (position_status=CLOSED, status=FILLED)
"""
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, func, case, desc, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.db.connection import get_db_session
from polyflip.db.models import DecisionFunnelLog, RuntimeSettings, TradeHistory
from polyflip.api.auth import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(tags=["MRF"], dependencies=[Depends(verify_api_key)])

VALID_MRF_MODES = {"OFF", "SHADOW", "ACTIVE"}
_mode_cache: dict = {"value": "OFF", "ts": 0.0}
_MODE_CACHE_TTL = 60


async def _get_mrf_mode(db: AsyncSession) -> str:
    """Read current MRF mode from RuntimeSettings, cached for 60s."""
    now = time.monotonic()
    if now - _mode_cache["ts"] < _MODE_CACHE_TTL:
        return _mode_cache["value"]

    result = await db.execute(
        select(RuntimeSettings).where(RuntimeSettings.key == "MARKET_REGIME_FILTER_MODE")
    )
    row = result.scalar_one_or_none()
    if row and row.value:
        mode = row.value.strip().upper()
        if mode not in VALID_MRF_MODES:
            mode = "OFF"
    else:
        mode = "OFF"

    _mode_cache["value"] = mode
    _mode_cache["ts"] = now
    return mode


@router.get("/api/mrf/status")
async def get_mrf_status(
    hours: int = Query(24, ge=1, le=168),
    asset: Optional[str] = Query(None),
    execution_mode: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Return current MRF regime status + statistics.
    Only counts entries where mrf_evaluated=true.
    Per-asset PnL from TradeHistory (realized_pnl_usdc, fallback to pnl).
    """
    mode = await _get_mrf_mode(db)
    since = datetime.now(timezone.utc) - timedelta(hours=hours)

    # 1. Get entries where MRF was actually evaluated
    filters = [
        DecisionFunnelLog.created_at >= since,
        DecisionFunnelLog.mrf_evaluated == True,
    ]
    if asset:
        filters.append(DecisionFunnelLog.asset == asset.upper())
    if execution_mode:
        filters.append(DecisionFunnelLog.execution_mode == execution_mode.upper())

    mrf_q = (
        select(DecisionFunnelLog)
        .where(and_(*filters))
        .order_by(desc(DecisionFunnelLog.created_at))
        .limit(500)
    )
    mrf_rows = (await db.execute(mrf_q)).scalars().all()

    # 2. Classify outcomes
    mrf_passed = 0
    mrf_blocked = 0
    mrf_reduced = 0
    mrf_not_ready = 0
    mrf_error = 0

    for row in mrf_rows:
        if row.mrf_failure_reason and "not_ready" in (row.mrf_failure_reason or "").lower():
            mrf_not_ready += 1
        elif row.mrf_failure_reason and row.mrf_failure_reason.startswith("mrf_error"):
            mrf_error += 1
        elif row.mrf_final_action == "SKIP" and row.mrf_original_action != "SKIP":
            mrf_blocked += 1
        elif row.mrf_multiplier is not None and row.mrf_multiplier < 1.0 and row.mrf_final_action != "SKIP":
            mrf_reduced += 1
        else:
            mrf_passed += 1

    total_evaluated = len(mrf_rows)
    total_blocked = mrf_blocked

    # 3. Phase distribution
    phase_counts = {}
    for row in mrf_rows:
        phase = row.mrf_phase or "UNKNOWN"
        phase_counts[phase] = phase_counts.get(phase, 0) + 1

    regime_distribution = {}
    for phase, count in phase_counts.items():
        regime_distribution[phase] = {
            "count": count,
            "pct": round(count / max(total_evaluated, 1) * 100, 1),
        }

    # 4. Per-asset stats from funnel
    asset_stats = {}
    for row in mrf_rows:
        a = row.asset or "UNKNOWN"
        if a not in asset_stats:
            asset_stats[a] = {
                "phase": row.mrf_asset_phase or "UNKNOWN",
                "strength": 0.0,
                "confidence": 0.0,
                "evaluated": 0,
                "passed": 0,
                "blocked": 0,
                "reduced": 0,
                "trades": 0,
                "wins": 0,
                "pnl": 0.0,
                "win_rate_pct": 0.0,
                "block_rate_pct": 0.0,
                "avg_multiplier": 1.0,
                "last_seen": None,
            }
        s = asset_stats[a]
        s["evaluated"] += 1
        if row.mrf_strength is not None:
            s["strength"] = row.mrf_strength
        if row.mrf_confidence is not None:
            s["confidence"] = row.mrf_confidence
        if row.mrf_asset_phase:
            s["phase"] = row.mrf_asset_phase
        if row.mrf_final_action == "SKIP" and row.mrf_original_action != "SKIP":
            s["blocked"] += 1
        elif row.mrf_multiplier is not None and row.mrf_multiplier < 1.0 and row.mrf_final_action != "SKIP":
            s["reduced"] += 1
        else:
            s["passed"] += 1
        if s["last_seen"] is None or (row.created_at and row.created_at > s["last_seen"]):
            s["last_seen"] = row.created_at

    # 5. Per-asset PnL from TradeHistory (only completed trades)
    if asset_stats:
        asset_names = list(asset_stats.keys())
        # Use realized_pnl_usdc with fallback to pnl for legacy rows.
        # Only count trades that are closed and filled.
        pnl_col = case(
            (TradeHistory.realized_pnl_usdc.isnot(None), TradeHistory.realized_pnl_usdc),
            else_=TradeHistory.pnl,
        )
        wins_col = case(
            (
                and_(
                    TradeHistory.realized_pnl_usdc.isnot(None),
                    TradeHistory.realized_pnl_usdc > 0,
                ),
                1,
            ),
            (
                and_(
                    TradeHistory.realized_pnl_usdc.is_(None),
                    TradeHistory.pnl.isnot(None),
                    TradeHistory.pnl > 0,
                ),
                1,
            ),
            else_=0,
        )
        th_q = (
            select(
                TradeHistory.asset,
                func.count(TradeHistory.id).label("trades"),
                func.sum(pnl_col).label("pnl"),
                func.sum(wins_col).label("wins"),
            )
            .where(
                and_(
                    TradeHistory.created_at >= since,
                    TradeHistory.asset.in_(asset_names),
                    TradeHistory.position_status == "CLOSED",
                    TradeHistory.status == "FILLED",
                )
            )
            .group_by(TradeHistory.asset)
        )
        th_rows = (await db.execute(th_q)).all()
        for th in th_rows:
            a = th.asset
            if a in asset_stats:
                trades = th.trades or 0
                wins = th.wins or 0
                pnl = float(th.pnl or 0.0)
                asset_stats[a]["trades"] = trades
                asset_stats[a]["wins"] = wins
                asset_stats[a]["pnl"] = round(pnl, 4)
                asset_stats[a]["win_rate_pct"] = round(wins / max(trades, 1) * 100, 1)

    # Compute derived fields
    for a, s in asset_stats.items():
        s["block_rate_pct"] = round(s["blocked"] / max(s["evaluated"], 1) * 100, 1)
        multipliers = []
        for row in mrf_rows:
            if row.asset == a and row.mrf_multiplier is not None:
                multipliers.append(row.mrf_multiplier)
        s["avg_multiplier"] = round(sum(multipliers) / len(multipliers), 4) if multipliers else 1.0
        if isinstance(s["last_seen"], datetime):
            s["last_seen"] = s["last_seen"].isoformat()

    # 6. Latest evaluation
    latest_regime = "UNKNOWN"
    latest_asset = None
    latest_strength = 0.0
    latest_confidence = 0.0
    if mrf_rows:
        latest = mrf_rows[0]
        latest_asset = latest.asset
        latest_regime = latest.mrf_phase or "UNKNOWN"
        latest_strength = latest.mrf_strength or 0.0
        latest_confidence = latest.mrf_confidence or 0.0

    # 7. Average multiplier
    multipliers = [r.mrf_multiplier for r in mrf_rows if r.mrf_multiplier is not None]
    avg_multiplier = sum(multipliers) / len(multipliers) if multipliers else 1.0

    return {
        "mode": mode,
        "latest_regime": latest_regime,
        "latest_asset": latest_asset,
        "latest_strength": round(latest_strength, 4),
        "latest_confidence": round(latest_confidence, 4),
        "total_evaluated": total_evaluated,
        "total_blocked": total_blocked,
        "total_passed": mrf_passed,
        "total_reduced": mrf_reduced,
        "total_not_ready": mrf_not_ready,
        "total_error": mrf_error,
        "block_rate_pct": round(total_blocked / max(total_evaluated, 1) * 100, 1),
        "avg_multiplier": round(avg_multiplier, 4),
        "regime_distribution": regime_distribution,
        "per_asset": asset_stats,
        "hours": hours,
    }
