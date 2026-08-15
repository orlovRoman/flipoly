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
MAX_ALLOWED_DRAWDOWN = 25.0  # USDC


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

    # Base score is the median net PnL
    score = float(median_pnl)

    # Drawdown penalty
    if max_drawdown > 15.0:
        score -= (max_drawdown - 15.0) * 0.15

    # Stability bonus for having multiple evaluated windows
    stability_bonus = min(windows_count, 5) * 0.02
    score += stability_bonus

    # Volume confidence bonus (small bonus for robust sample sizes >= 75 trades)
    if trade_count >= 75:
        score += 0.05

    # Diagnostic quality bonus if AUC is known and strong
    if auc is not None and math.isfinite(auc) and auc > 0.55:
        score += (auc - 0.55) * 0.10

    return round(score, 4)


def evaluate_candidate_policy(
    metrics: Mapping[str, Any],
    *,
    min_trades: int = MIN_MANDATORY_TRADES,
    min_windows: int = MIN_MANDATORY_WINDOWS,
    max_drawdown_limit: float = MAX_ALLOWED_DRAWDOWN,
) -> PolicyEvaluationResult:
    """Evaluate experiment metrics against non-negotiable policy constraints."""
    rejection_reasons: list[str] = []

    median_pnl = metrics.get("median_pnl", metrics.get("median_oot_pnl", 0.0))
    max_dd = metrics.get("max_drawdown", metrics.get("median_oot_drawdown", 0.0))
    trade_count = metrics.get("total_trades", metrics.get("trade_count", 0))
    windows_count = metrics.get("windows_count", metrics.get("oot_windows", 0))
    auc = metrics.get("auc")
    brier = metrics.get("brier")

    # 1. Finite checks
    for name, val in [("median_pnl", median_pnl), ("max_drawdown", max_dd)]:
        if val is None or not math.isfinite(float(val)):
            rejection_reasons.append(f"NON_FINITE_VALUE: {name}")

    # 2. Trade volume check
    effective_min_trades = max(MIN_MANDATORY_TRADES, min_trades)
    if int(trade_count) < effective_min_trades:
        rejection_reasons.append(
            f"INSUFFICIENT_TRADES: got {trade_count}, required minimum {effective_min_trades}"
        )

    # 3. Independent windows check
    effective_min_windows = max(MIN_MANDATORY_WINDOWS, min_windows)
    if int(windows_count) < effective_min_windows:
        rejection_reasons.append(
            f"INSUFFICIENT_WINDOWS: got {windows_count}, required minimum {effective_min_windows}"
        )

    # 4. Strictly positive net PnL
    if float(median_pnl) <= 0.0:
        rejection_reasons.append(
            f"NON_POSITIVE_PNL: median net PnL is {median_pnl:.4f} <= 0.0"
        )

    # 5. Drawdown limit
    if float(max_dd) > max_drawdown_limit:
        rejection_reasons.append(
            f"EXCESSIVE_DRAWDOWN: max drawdown {max_dd:.2f} exceeds limit {max_drawdown_limit:.2f}"
        )

    gate_passed = len(rejection_reasons) == 0
    score = score_candidate(
        median_pnl=float(median_pnl) if math.isfinite(float(median_pnl)) else 0.0,
        max_drawdown=float(max_dd) if math.isfinite(float(max_dd)) else 0.0,
        trade_count=int(trade_count),
        windows_count=int(windows_count),
        auc=float(auc) if auc is not None and math.isfinite(float(auc)) else None,
        brier=float(brier) if brier is not None and math.isfinite(float(brier)) else None,
    )

    return PolicyEvaluationResult(
        is_eligible=gate_passed,
        gate_passed=gate_passed,
        score=score,
        median_pnl=float(median_pnl) if math.isfinite(float(median_pnl)) else 0.0,
        max_drawdown=float(max_dd) if math.isfinite(float(max_dd)) else 0.0,
        trade_count=int(trade_count),
        windows_count=int(windows_count),
        rejection_reasons=rejection_reasons,
        diagnostics={"auc": auc, "brier": brier},
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
        # Allowed in LIVE_PROPOSE or higher
        if level in {"OBSERVE", "EXPERIMENT"}:
            raise AILabError(f"Action '{action}' requires 'LIVE_PROPOSE' autonomy level")
        return

    if action in {"ACTIVATE_LIVE_DEPLOYMENT"}:
        if level != "AUTONOMOUS_LIVE" or not allow_autonomous_live:
            raise AILabError(
                "Autonomous LIVE activation is prohibited. Human-in-the-loop approval required."
            )
