"""
MRF status API endpoint v3.

Uses MRF telemetry columns (mrf_evaluated, mrf_mode, mrf_phase, etc.)
from DecisionFunnelLog. Per-asset PnL from TradeHistory.

Fixes:
- Uses realized_pnl_usdc with fallback to pnl for legacy rows
- Uses sqlalchemy.case (not func.case)
- Terminal position_status: CLOSED, RESOLVED_REDEEMABLE, RESOLVED_LOST, REDEEMED
- Executed trades: status == SUCCESS (not SKIPPED/FAILED/CANCELLED)
"""
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, func, case, desc, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.db.connection import get_db_session
from polyflip.db.models import DecisionFunnelLog, RuntimeSettings, TradeHistory
from polyflip.constants import COMBINED_MODE_SUPPORTED_ASSETS
from polyflip.api.auth import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(tags=["MRF"], dependencies=[Depends(verify_api_key)])

VALID_MRF_MODES = {"OFF", "SHADOW", "ACTIVE"}
_mode_cache: dict = {"value": "OFF", "ts": 0.0}
_version_cache: dict = {"value": 1, "ts": 0.0}

_MODE_CACHE_TTL = 60


def _parse_mrf_audit(value) -> dict:
    """Decode audit JSON defensively; old rows may contain NULL or malformed text."""
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _is_known_phase(value) -> bool:
    """Return True only for an actual classified phase, not a missing sentinel."""
    return isinstance(value, str) and value.strip().upper() not in {"", "UNKNOWN", "NONE", "NULL"}

_NOT_READY_FAILURE_PREFIXES = (
    "not_ready",
    "missing_asset",
    "insufficient_history",
    "candle_error",
    "continuity_error",
)


def _is_not_ready_failure(value) -> bool:
    reason = str(value or "").strip().lower()
    return any(
        reason == prefix or reason.startswith(prefix + ":")
        for prefix in _NOT_READY_FAILURE_PREFIXES
    )


def _is_mrf_error(value) -> bool:
    reason = str(value or "").strip().lower()
    return reason.startswith(("runtime_error", "mrf_error"))


def _extract_mrf_telemetry(row) -> dict:
    """Read current MRF telemetry, preferring the complete audit payload."""
    audit = _parse_mrf_audit(getattr(row, "mrf_audit_json", None))
    assets = audit.get("assets") if isinstance(audit.get("assets"), dict) else {}

    global_phase = (
        audit.get("global_phase")
        or audit.get("global_regime")
        or getattr(row, "mrf_phase", None)
        or "UNKNOWN"
    )
    global_strength = audit.get("global_strength")
    if global_strength is None:
        global_strength = getattr(row, "mrf_strength", None)
    global_confidence = audit.get("global_confidence")
    if global_confidence is None:
        global_confidence = getattr(row, "mrf_confidence", None)

    gate = audit.get("gate") if isinstance(audit.get("gate"), dict) else {}
    policy_version = getattr(row, "mrf_policy_version", None)
    if policy_version is None:
        policy_version = audit.get("version")
    regime_evidence = getattr(row, "mrf_regime_evidence", None)
    if regime_evidence is None:
        regime_evidence = gate.get("regime_evidence")
    gate_would_block = getattr(row, "mrf_gate_would_block", None)
    if gate_would_block is None:
        gate_would_block = gate.get("would_block")
    gate_reason = getattr(row, "mrf_gate_reason", None)
    if gate_reason is None:
        gate_reason = gate.get("reason")
    gate_threshold = getattr(row, "mrf_gate_threshold", None)
    if gate_threshold is None:
        gate_threshold = gate.get("veto_threshold")
    edge_margin = getattr(row, "mrf_edge_margin", None)
    if edge_margin is None:
        edge_margin = gate.get("edge_margin")

    return {
        "audit": audit,
        "global_phase": global_phase,
        "global_strength": global_strength,
        "global_confidence": global_confidence,
        "assets": assets,
        "policy_version": policy_version,
        "regime_evidence": regime_evidence,
        "gate_would_block": gate_would_block,
        "gate_reason": gate_reason,
        "gate_threshold": gate_threshold,
        "edge_margin": edge_margin,
    }


def _classify_mrf_row(row, telemetry: dict | None = None) -> dict[str, bool | str]:
    """Classify one row without conflating v3 evidence with v2 sizing."""
    telemetry = telemetry or _extract_mrf_telemetry(row)
    if _is_not_ready_failure(getattr(row, "mrf_failure_reason", None)):
        return {"category": "not_ready", "would_block": False, "blocked": False, "reduced": False}
    if _is_mrf_error(getattr(row, "mrf_failure_reason", None)):
        return {"category": "error", "would_block": False, "blocked": False, "reduced": False}

    policy_version = _telemetry_policy_version(telemetry)

    if policy_version == 3:
        would_block = bool(telemetry.get("gate_would_block"))
        blocked = (
            getattr(row, "mrf_final_action", None) == "SKIP"
            and getattr(row, "mrf_original_action", None) != "SKIP"
        )
        return {
            "category": "blocked" if blocked else "passed",
            "would_block": would_block,
            "blocked": blocked,
            "reduced": False,
        }

    blocked = (
        getattr(row, "mrf_final_action", None) == "SKIP"
        and getattr(row, "mrf_original_action", None) != "SKIP"
    )
    multiplier = getattr(row, "mrf_multiplier", None)
    reduced = (
        multiplier is not None
        and multiplier < 1.0
        and not blocked
    )
    return {
        "category": "blocked" if blocked else ("reduced" if reduced else "passed"),
        "would_block": False,
        "blocked": blocked,
        "reduced": reduced,
    }


def _telemetry_policy_version(telemetry: dict) -> int:
    """Normalize DB/audit policy versions before v3-specific aggregation."""
    try:
        return int(telemetry.get("policy_version"))
    except (TypeError, ValueError):
        return 0


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


async def _get_mrf_version(db: AsyncSession) -> int:
    """Read configured MRF version, defaulting to the legacy policy."""
    now = time.monotonic()
    if now - _version_cache["ts"] < _MODE_CACHE_TTL:
        return _version_cache["value"]
    result = await db.execute(
        select(RuntimeSettings).where(
            RuntimeSettings.key == "MARKET_REGIME_FILTER_VERSION"
        )
    )
    row = result.scalar_one_or_none()
    try:
        version = int(float(row.value)) if row and row.value else 1
    except (TypeError, ValueError):
        version = 1
    if version not in (1, 2, 3):
        version = 1
    _version_cache["value"] = version
    _version_cache["ts"] = now
    return version


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
    current_version = await _get_mrf_version(db)
    since = datetime.now(timezone.utc) - timedelta(hours=hours)

    # 1. Get MRF attempts, including not-ready/error rows. These rows are
    # needed to explain UNKNOWN instead of silently reporting zero failures.
    filters = [
        DecisionFunnelLog.created_at >= since,
        DecisionFunnelLog.mrf_mode.in_(("SHADOW", "ACTIVE")),
        or_(
            DecisionFunnelLog.mrf_evaluated == True,
            DecisionFunnelLog.mrf_failure_reason.isnot(None),
            DecisionFunnelLog.mrf_audit_json.isnot(None),
        ),
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
    mrf_would_block = 0
    evidence_values: list[float] = []

    for row in mrf_rows:
        classification = _classify_mrf_row(row)
        category = classification["category"]
        if category == "not_ready":
            mrf_not_ready += 1
        elif category == "error":
            mrf_error += 1
        elif category == "blocked":
            mrf_blocked += 1
        elif category == "reduced":
            mrf_reduced += 1
        else:
            mrf_passed += 1
        telemetry = _extract_mrf_telemetry(row)
        if _telemetry_policy_version(telemetry) == 3:
            if classification["would_block"]:
                mrf_would_block += 1
            evidence = telemetry.get("regime_evidence")
            if evidence is not None:
                evidence_values.append(float(evidence))

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
                "phase": "UNKNOWN",
                "strength": None,
                "confidence": None,
                "phase_available": False,
                "evaluated": 0,
                "passed": 0,
                "blocked": 0,
                "reduced": 0,
                "would_block": 0,
                "not_ready": 0,
                "trades": 0,
                "wins": 0,
                "pnl": 0.0,
                "win_rate_pct": 0.0,
                "block_rate_pct": 0.0,
                "avg_multiplier": 1.0,
                "avg_regime_evidence": 0.0,
                "_regime_evidence_values": [],
                "last_seen": None,
            }
        s = asset_stats[a]
        s["evaluated"] += 1

        telemetry = _extract_mrf_telemetry(row)
        asset_payload = telemetry["assets"].get(a)
        if not isinstance(asset_payload, dict):
            asset_payload = {}

        candidate_phase = (
            asset_payload.get("phase")
            or getattr(row, "mrf_asset_phase", None)
            or telemetry["global_phase"]
            or "UNKNOWN"
        )
        candidate_strength = asset_payload.get("strength")
        if candidate_strength is None:
            candidate_strength = telemetry["global_strength"]
        candidate_confidence = asset_payload.get("confidence")
        if candidate_confidence is None:
            candidate_confidence = telemetry["global_confidence"]

        # Rows are ordered newest first. Keep the newest valid telemetry,
        # but skip legacy UNKNOWN/zero placeholders when an older valid audit exists.
        if not s["phase_available"] and _is_known_phase(candidate_phase):
            s["phase"] = candidate_phase
            s["phase_available"] = True
        if s["strength"] is None and candidate_strength is not None:
            s["strength"] = float(candidate_strength)
        if s["confidence"] is None and candidate_confidence is not None:
            s["confidence"] = float(candidate_confidence)
        classification = _classify_mrf_row(row, telemetry)
        category = classification["category"]
        if category == "not_ready" or category == "error":
            s["not_ready"] += 1
        elif category == "blocked":
            s["blocked"] += 1
        elif category == "reduced":
            s["reduced"] += 1
        else:
            s["passed"] += 1
        if classification["would_block"]:
            s["would_block"] += 1
        if (
            _telemetry_policy_version(telemetry) == 3
            and telemetry.get("regime_evidence") is not None
        ):
            s["_regime_evidence_values"].append(float(telemetry["regime_evidence"]))
        if s["last_seen"] is None or (row.created_at and row.created_at > s["last_seen"]):
            s["last_seen"] = row.created_at

    # Always expose the configured basket, even when no recent MRF row exists.
    # This keeps PnL visible while making missing regime telemetry explicit.
    expected_dashboard_assets = set(COMBINED_MODE_SUPPORTED_ASSETS)
    if asset:
        expected_dashboard_assets.add(asset.upper())
    for expected_asset in sorted(expected_dashboard_assets):
        asset_stats.setdefault(expected_asset, {
            "phase": "UNKNOWN",
            "strength": None,
            "confidence": None,
            "phase_available": False,
            "evaluated": 0,
            "passed": 0,
            "blocked": 0,
            "reduced": 0,
            "would_block": 0,
            "not_ready": 0,
            "trades": 0,
            "wins": 0,
            "pnl": 0.0,
            "win_rate_pct": 0.0,
            "block_rate_pct": 0.0,
            "avg_multiplier": 1.0,
            "avg_regime_evidence": 0.0,
            "_regime_evidence_values": [],
            "last_seen": None,
        })

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
        TERMINAL_POSITION_STATUSES = ("CLOSED", "RESOLVED_REDEEMABLE", "RESOLVED_LOST", "REDEEMED")
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
                    TradeHistory.position_status.in_(TERMINAL_POSITION_STATUSES),
                    TradeHistory.status == "SUCCESS",
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
            telemetry = _extract_mrf_telemetry(row)
            # v3 is a binary gate; any multiplier value on a legacy row must
            # not leak into the v2 average shown by the API.
            if (
                row.asset == a
                and row.mrf_multiplier is not None
                and _telemetry_policy_version(telemetry) != 3
            ):
                multipliers.append(row.mrf_multiplier)
        s["avg_multiplier"] = round(sum(multipliers) / len(multipliers), 4) if multipliers else 1.0
        evidence = s.pop("_regime_evidence_values", [])
        s["avg_regime_evidence"] = (
            round(sum(evidence) / len(evidence), 6)
            if evidence else 0.0
        )
        if s["strength"] is None:
            s["strength"] = 0.0
        if s["confidence"] is None:
            s["confidence"] = 0.0
        if isinstance(s["last_seen"], datetime):
            s["last_seen"] = s["last_seen"].isoformat()

    # 6. Latest evaluation
    latest_regime = "UNKNOWN"
    latest_asset = None
    latest_strength = 0.0
    latest_confidence = 0.0
    latest_audit = {}
    # Prefer the newest row with an actual phase. A legacy UNKNOWN row must not
    # hide a valid evaluation that is only a few rows older.
    latest = None
    for candidate_row in mrf_rows:
        candidate_telemetry = _extract_mrf_telemetry(candidate_row)
        if _is_known_phase(candidate_telemetry["global_phase"]):
            latest = candidate_row
            latest_audit = candidate_telemetry
            break
    if latest is None and mrf_rows:
        latest = mrf_rows[0]
        latest_audit = _extract_mrf_telemetry(latest)
    if latest is not None:
        latest_asset = latest.asset
        latest_regime = latest_audit.get("global_phase") or "UNKNOWN"
        latest_strength = latest_audit.get("global_strength") or 0.0
        latest_confidence = latest_audit.get("global_confidence") or 0.0

    # 7. Average multiplier
    multipliers = [
        r.mrf_multiplier
        for r in mrf_rows
        if r.mrf_multiplier is not None
        and _telemetry_policy_version(_extract_mrf_telemetry(r)) != 3
    ]
    avg_multiplier = sum(multipliers) / len(multipliers) if multipliers else 1.0
    avg_regime_evidence = (
        sum(evidence_values) / len(evidence_values)
        if evidence_values else 0.0
    )

    return {
        "mode": mode,
        "policy_version": current_version,
        "latest_regime": latest_regime,
        "latest_asset": latest_asset,
        "latest_strength": round(latest_strength, 4),
        "latest_confidence": round(latest_confidence, 4),
        "phase_available": _is_known_phase(latest_regime),
        "total_evaluated": total_evaluated,
        "total_blocked": total_blocked,
        "total_passed": mrf_passed,
        "total_reduced": mrf_reduced,
        "total_not_ready": mrf_not_ready,
        "total_error": mrf_error,
        "total_would_block": mrf_would_block,
        "block_rate_pct": round(total_blocked / max(total_evaluated, 1) * 100, 1),
        "avg_multiplier": round(avg_multiplier, 4),
        "avg_regime_evidence": round(avg_regime_evidence, 6),
        "regime_distribution": regime_distribution,
        "per_asset": asset_stats,
        "hours": hours,
    }
