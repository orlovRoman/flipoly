"""
MRF status API endpoint v2.

Uses new MRF telemetry columns (mrf_mode, mrf_phase, mrf_asset_phase, etc.)
from DecisionFunnelLog instead of parsing skip_reason strings.
"""
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, func, desc, and_, case
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.db.connection import get_db_session
from polyflip.db.models import DecisionFunnelLog, RuntimeSettings
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
    db: AsyncSession = Depends(get_db_session),
):
    """
    Return current MRF regime status + statistics.

    Uses new mrf_* columns in DecisionFunnelLog (v2).
    """
    mode = await _get_mrf_mode(db)
    since = datetime.now(timezone.utc) - timedelta(hours=hours)

    # 1. Get recent entries with MRF data (mrf_mode IS NOT NULL)
    mrf_q = (
        select(DecisionFunnelLog)
        .where(
            and_(
                DecisionFunnelLog.created_at >= since,
                DecisionFunnelLog.mrf_mode.isnot(None),
            )
        )
        .order_by(desc(DecisionFunnelLog.created_at))
        .limit(200)
    )
    mrf_rows = (await db.execute(mrf_q)).scalars().all()

    # Fallback: also get entries with old MRF: prefix in skip_reason
    old_mrf_q = (
        select(DecisionFunnelLog)
        .where(
            and_(
                DecisionFunnelLog.created_at >= since,
                DecisionFunnelLog.mrf_mode.is_(None),
                DecisionFunnelLog.skip_reason.like("MRF:%"),
            )
        )
        .order_by(desc(DecisionFunnelLog.created_at))
        .limit(100)
    )
    old_mrf_rows = (await db.execute(old_mrf_q)).scalars().all()

    # 2. Aggregate stats
    total_evaluated = len(mrf_rows) + len(old_mrf_rows)
    total_blocked = sum(1 for r in mrf_rows if r.final_action == "SKIP")
    total_blocked += sum(1 for r in old_mrf_rows if r.final_action == "SKIP")

    # Phase distribution (from new columns)
    phase_counts = {}
    for row in mrf_rows:
        phase = row.mrf_phase or "UNKNOWN"
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
    # Fallback to old format
    for row in old_mrf_rows:
        phase = (row.skip_reason or "").replace("MRF:", "")
        phase_counts[phase] = phase_counts.get(phase, 0) + 1

    # 3. Per-asset stats
    asset_stats = {}
    for row in mrf_rows:
        asset = row.asset or "UNKNOWN"
        if asset not in asset_stats:
            asset_stats[asset] = {"evaluated": 0, "blocked": 0, "phases": {}}
        asset_stats[asset]["evaluated"] += 1
        if row.final_action == "SKIP":
            asset_stats[asset]["blocked"] += 1
        phase = row.mrf_asset_phase or row.mrf_phase or "UNKNOWN"
        asset_stats[asset]["phases"][phase] = asset_stats[asset]["phases"].get(phase, 0) + 1

    # 4. Latest evaluation
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

    # 5. Average multiplier
    multipliers = [r.mrf_multiplier for r in mrf_rows if r.mrf_multiplier is not None]
    avg_multiplier = sum(multipliers) / len(multipliers) if multipliers else 1.0

    # 6. Phase distribution with percentages
    regime_distribution = {}
    for phase, count in phase_counts.items():
        regime_distribution[phase] = {
            "count": count,
            "pct": round(count / max(total_evaluated, 1) * 100, 1),
        }

    return {
        "mode": mode,
        "latest_regime": latest_regime,
        "latest_asset": latest_asset,
        "latest_strength": round(latest_strength, 4),
        "latest_confidence": round(latest_confidence, 4),
        "total_evaluated": total_evaluated,
        "total_blocked": total_blocked,
        "total_passed": total_evaluated - total_blocked,
        "block_rate_pct": round(total_blocked / max(total_evaluated, 1) * 100, 1),
        "avg_multiplier": round(avg_multiplier, 4),
        "regime_distribution": regime_distribution,
        "per_asset": asset_stats,
        "hours": hours,
    }
