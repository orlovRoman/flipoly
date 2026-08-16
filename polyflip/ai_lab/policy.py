"""Deterministic policy engine and safety guardrail validator for AI Lab Phase 10.

Calculates candidate quality score and enforces non-negotiable boundaries on parameters and autonomy.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any
import structlog

from polyflip.ai_lab.service import AILabError

logger = structlog.get_logger("polyflip.ai_lab.policy")

MIN_MANDATORY_TRADES = 50
MIN_MANDATORY_WINDOWS = 3
MAX_ALLOWED_DRAWDOWN_USDC = 25.0
MAX_ALLOWED_DRAWDOWN = MAX_ALLOWED_DRAWDOWN_USDC


class PolicyEvaluationResult:
    """Outcome of evaluating an experiment candidate against quantitative policy rules."""

    def __init__(
        self,
        *,
        is_eligible: bool,
        gate_passed: bool,
        score: float,
        median_pnl: float,
        max_drawdown: float,
        trade_count: int,
        windows_count: int,
        rejection_reasons: list[str] | None = None,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        self.is_eligible = is_eligible
        self.gate_passed = gate_passed
        self.score = score
        self.median_pnl = median_pnl
        self.max_drawdown = max_drawdown
        self.trade_count = trade_count
        self.windows_count = windows_count
        self.rejection_reasons = list(rejection_reasons or [])
        self.diagnostics = dict(diagnostics or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_eligible": self.is_eligible,
            "gate_passed": self.gate_passed,
            "score": round(self.score, 4),
            "median_pnl": round(self.median_pnl, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "trade_count": self.trade_count,
            "windows_count": self.windows_count,
            "rejection_reasons": self.rejection_reasons,
            "diagnostics": self.diagnostics,
        }


def score_candidate(
    *,
    median_pnl: float,
    max_drawdown: float,
    trade_count: int,
    windows_count: int,
    auc: float | None = None,
    brier: float | None = None,
) -> float:
    """Calculate deterministic candidate score."""
    if not math.isfinite(median_pnl) or not math.isfinite(max_drawdown):
        return -9999.0

    # Financial score: PnL scales with independent trades and drawdown is
    # an explicit USDC penalty. AUC/ECE/Brier remain diagnostics.
    score = float(median_pnl) * max(int(trade_count), 0)
    score -= 0.5 * max(float(max_drawdown), 0.0)
    return round(score, 4)


def evaluate_candidate_policy(
    metrics: Mapping[str, Any],
    *,
    min_trades: int = MIN_MANDATORY_TRADES,
    min_windows: int = MIN_MANDATORY_WINDOWS,
    max_drawdown_limit: float = MAX_ALLOWED_DRAWDOWN,
    mode: str | None = None,
) -> PolicyEvaluationResult:
    """Evaluate metrics, optionally retaining technically valid candidates in RESEARCH mode.

    RESEARCH relaxes only evidence requirements (trade count, windows and positive
    PnL). Non-finite values, malformed counts and excessive drawdown remain hard
    failures, so this mode cannot turn an invalid result into a candidate.
    """
    rejection_reasons: list[str] = []
    from polyflip.config import settings
    research_mode = str(mode or getattr(settings, "AI_LAB_MODE", "STANDARD")).upper() == "RESEARCH"

    median_pnl = metrics.get("median_pnl", metrics.get("median_oot_pnl", 0.0))
    max_dd = metrics.get(
        "max_drawdown_usdc",
        metrics.get("max_drawdown", metrics.get("median_oot_drawdown", 0.0)),
    )
    trade_count = metrics.get("total_trades", metrics.get("trade_count", 0))
    windows_count = metrics.get(
        "window_count",
        metrics.get("windows_count", metrics.get("oot_windows", 0)),
    )
    auc = metrics.get("auc")
    brier = metrics.get("brier")

    def finite_number(value: Any, fallback: float = 0.0) -> tuple[float, bool]:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return fallback, False
        return (number, math.isfinite(number))

    pnl_num, pnl_ok = finite_number(median_pnl)
    dd_num, dd_ok = finite_number(max_dd)
    try:
        trades_num = int(trade_count)
        trades_ok = trades_num >= 0
    except (TypeError, ValueError):
        trades_num, trades_ok = 0, False
    try:
        windows_num = int(windows_count)
        windows_ok = windows_num >= 0
    except (TypeError, ValueError):
        windows_num, windows_ok = 0, False

    # 1. Finite and shape checks.
    if not pnl_ok:
        rejection_reasons.append("NON_FINITE_VALUE: median_pnl")
    if not dd_ok:
        rejection_reasons.append("NON_FINITE_VALUE: max_drawdown")
    if not trades_ok:
        rejection_reasons.append("INVALID_TRADE_COUNT")
    if not windows_ok:
        rejection_reasons.append("INVALID_WINDOW_COUNT")

    # 2. Trade volume and independent window checks.
    effective_min_trades = max(MIN_MANDATORY_TRADES, min_trades)
    if trades_num < effective_min_trades:
        rejection_reasons.append(
            f"INSUFFICIENT_TRADES: got {trade_count}, required minimum {effective_min_trades}"
        )
    effective_min_windows = max(MIN_MANDATORY_WINDOWS, min_windows)
    if windows_num < effective_min_windows:
        rejection_reasons.append(
            f"INSUFFICIENT_WINDOWS: got {windows_count}, required minimum {effective_min_windows}"
        )

    # 3. Strictly positive net PnL and drawdown cap (USDC).
    if pnl_ok and pnl_num <= 0.0:
        rejection_reasons.append(
            f"NON_POSITIVE_PNL: median net PnL is {pnl_num:.4f} <= 0.0"
        )
    if dd_ok and dd_num < 0.0:
        rejection_reasons.append("INVALID_DRAWDOWN")
    elif dd_ok and dd_num > max_drawdown_limit:
        rejection_reasons.append(
            f"EXCESSIVE_DRAWDOWN: max drawdown {dd_num:.2f} exceeds limit {max_drawdown_limit:.2f}"
        )

    hard_reasons = {
        "NON_FINITE_VALUE: median_pnl",
        "NON_FINITE_VALUE: max_drawdown",
        "INVALID_TRADE_COUNT",
        "INVALID_WINDOW_COUNT",
        "INVALID_DRAWDOWN",
    }
    if research_mode:
        gate_passed = not any(
            reason in hard_reasons or reason.startswith("EXCESSIVE_DRAWDOWN:")
            for reason in rejection_reasons
        )
    else:
        gate_passed = len(rejection_reasons) == 0
    score = score_candidate(
        median_pnl=pnl_num,
        max_drawdown=dd_num,
        trade_count=trades_num,
        windows_count=windows_num,
        auc=float(auc) if auc is not None and math.isfinite(float(auc)) else None,
        brier=float(brier) if brier is not None and math.isfinite(float(brier)) else None,
    )

    return PolicyEvaluationResult(
        is_eligible=gate_passed,
        gate_passed=gate_passed,
        score=score,
        median_pnl=pnl_num,
        max_drawdown=dd_num,
        trade_count=trades_num,
        windows_count=windows_num,
        rejection_reasons=rejection_reasons,
        diagnostics={"auc": auc, "brier": brier, "mode": "RESEARCH" if research_mode else "STANDARD", "provisional": bool(research_mode and gate_passed and rejection_reasons)},
    )


def validate_agent_action_autonomy(
    action: str,
    autonomy_level: str,
    *,
    allow_autonomous_live: bool = False,
) -> None:
    """Validate if requested action is permitted under the run's autonomy level."""
    action = action.upper()
    level = autonomy_level.upper()

    if action in {"PROPOSE_HYPOTHESIS", "EVALUATE_BASELINE"}:
        return  # Allowed in all autonomy levels

    if action in {"CREATE_CONFIG", "TRAIN_MODEL", "RUN_OOT"}:
        if level in {"OBSERVE"}:
            raise AILabError(f"Action '{action}' is prohibited in 'OBSERVE' autonomy level")
        return

    if action in {"ASSIGN_SHADOW", "APPLY_SHADOW_OVERLAY"}:
        if level in {"OBSERVE", "EXPERIMENT"}:
            raise AILabError(f"Action '{action}' requires 'AUTONOMOUS_SHADOW' or higher autonomy level")
        return

    if action in {"APPLY_CONFIG_OVERLAY"}:
        if level in {"OBSERVE", "EXPERIMENT", "AUTONOMOUS_SHADOW"}:
            raise AILabError(f"Action '{action}' requires 'AUTONOMOUS_CONFIG' or higher autonomy level")
        return

    if action in {"PROPOSE_LIVE_DEPLOYMENT"}:
        if level not in {"LIVE_PROPOSE", "AUTONOMOUS_LIVE", "DIRECTED"}:
            raise AILabError(f"Action '{action}' requires 'LIVE_PROPOSE' autonomy level")
        return

    if action in {"ACTIVATE_LIVE_DEPLOYMENT"}:
        if level != "AUTONOMOUS_LIVE" or not allow_autonomous_live:
            raise AILabError(
                "Autonomous LIVE activation is prohibited. Human-in-the-loop approval required."
            )
