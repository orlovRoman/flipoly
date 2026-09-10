"""Steps 38-39: single report + computed final status.

Required sections: canonical data reality, exclusions, forecast quality,
challenger rule, economics + assumptions, uncertainty + concentration,
final decision. Every figure must come from final tables; legacy proxy
results live only in a separate appendix.

Status is COMPUTED from pre-set criteria, never fitted to a liked result.
"""
from __future__ import annotations


STATUSES = ("NOT_STARTED", "DATA_BLOCKED", "PENDING_NEW_PERIOD",
            "NO_INCREMENTAL_VALUE", "FORECAST_ONLY", "PROMISING_UNCERTAIN",
            "PAPER_CANDIDATE")
# Lifecycle: NOT_STARTED (final_start=null, pre-freeze) -> PENDING_NEW_PERIOD
# (frozen, awaiting new markets) -> computed status via final_status().
# final_status() itself never returns NOT_STARTED; the manifest tracks it.


def final_status(summary: dict) -> str:
    """summary keys: n_canonical, challenger_margin_ok, forecast_better,
    net_pnl, ci_lo, ci_hi, n_days, max_drawdown_ok, concentrated_ok,
    final_has_new_data."""
    if summary.get("n_canonical", 0) <= 0:
        return "DATA_BLOCKED"
    if summary.get("final_has_new_data") is False:
        return "PENDING_NEW_PERIOD"
    if not summary.get("challenger_margin_ok"):
        return "NO_INCREMENTAL_VALUE"
    if not summary.get("forecast_better"):
        return "NO_INCREMENTAL_VALUE"
    if summary.get("net_pnl", 0) <= 0:
        return "FORECAST_ONLY"
    lo, hi = summary.get("ci_lo", -1), summary.get("ci_hi", 1)
    if lo > 0 and summary.get("max_drawdown_ok") and summary.get("concentrated_ok"):
        return "PAPER_CANDIDATE"
    return "PROMISING_UNCERTAIN"


def build_report(context: dict) -> dict:
    return {
        "sections": {
            "canonical_data": context.get("canonical_data"),
            "exclusions": context.get("exclusions"),
            "forecast_quality": context.get("forecast_quality"),
            "challenger_rule": context.get("challenger_rule"),
            "economics": context.get("economics"),
            "uncertainty_concentration": context.get("uncertainty_concentration"),
        },
        "final_status": final_status(context.get("summary", {})),
        "appendix_proxy_only": context.get("proxy_appendix", "legacy proxy results (if any) live here only"),
    }
