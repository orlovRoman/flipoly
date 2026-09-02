"""Helpers for persisting weighted-policy decision telemetry.

The decision pipeline carries the canonical weighted fields in decision_details.
This module keeps the persistence mapping in one place so PAPER, SHADOW, LIVE and
SKIP rows expose the same probability and cost diagnostics.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

# These names intentionally match the SQLAlchemy columns on both telemetry tables.
# Runtime names may be more explicit than the historical database column names.
# Keep the ORM payload compatible while accepting both spellings from decisions.
WEIGHTED_TELEMETRY_ALIASES: dict[str, str] = {
    "weighted_market_contribution_logodds": "weighted_market_reference_logodds",
}

WEIGHTED_TELEMETRY_FIELDS: tuple[str, ...] = (
    "p_market_yes",
    "p_logreg_yes",
    "p_lgbm_yes",
    "weighted_policy_mode",
    "weighted_p_market_yes",
    "weighted_p_logreg_yes",
    "weighted_p_lgbm_yes",
    "weighted_p_final_yes",
    "weighted_market_weight",
    "weighted_logreg_weight",
    "weighted_lgbm_weight",
    "weighted_mrf_evidence",
    "weighted_market_contribution_logodds",
    "weighted_logreg_contribution_logodds",
    "weighted_lgbm_contribution_logodds",
    "weighted_mrf_contribution_logodds",
    "weighted_models_agree_contribution_logodds",
    "weighted_intercept_contribution_logodds",
    "weighted_models_agree",
    "weighted_selected_side",
    "weighted_yes_net_ev",
    "weighted_no_net_ev",
    "weighted_net_ev_per_share",
    "weighted_min_net_ev",
    "weighted_cost_per_share",
    "weighted_fee_rate",
    "weighted_maker_fee_rate",
    "weighted_execution_role",
    "weighted_fee_exponent",
    "weighted_fee_per_share",
    "weighted_maker_fee_per_share",
    "weighted_taker_fee_per_share",
    "weighted_slippage_per_share",
    "weighted_spread_per_share",
    "weighted_latency_buffer_per_share",
    "weighted_expected_execution_price",
    "weighted_missing_components",
    "weighted_selection_reason",
    "weighted_fee_source",
    "weighted_policy_id",
    "weighted_edge_lower_bound",
    "weighted_size_multiplier",
    "weighted_benchmark_json",
)


def _value(details: Mapping[str, Any], key: str) -> Any:
    """Read the first non-null value from canonical/weighted aliases."""
    alias = WEIGHTED_TELEMETRY_ALIASES.get(key)
    names = (key, alias, f"weighted_{key}") if alias else (key, f"weighted_{key}")
    for name in names:
        value = details.get(name)
        if value is not None and value != "":
            return value
    return None


def weighted_telemetry_from_details(
    details: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return only model-backed weighted fields safe for ORM construction.

    decision_details is also used by legacy code and may contain arbitrary JSON
    keys. Never pass that whole dictionary into a SQLAlchemy constructor.
    """
    if not isinstance(details, Mapping):
        return {}

    weighted_keys_present = any(
        key in details
        for key in (
            "weighted_policy_mode",
            "p_market_yes",
            "p_logreg_yes",
            "p_lgbm_yes",
            "weighted_p_market_yes",
            "weighted_p_logreg_yes",
            "weighted_p_lgbm_yes",
            "weighted_p_final_yes",
            "weighted_selected_side",
            "weighted_market_reference_logodds",
        )
    )
    if not weighted_keys_present:
        return {}

    result: dict[str, Any] = {}
    for key in WEIGHTED_TELEMETRY_FIELDS:
        value = _value(details, key)
        if value is not None:
            result[key] = value

    benchmark = result.get("weighted_benchmark_json")
    if benchmark is not None and not isinstance(benchmark, str):
        result["weighted_benchmark_json"] = json.dumps(
            benchmark,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return result


def weighted_telemetry_from_object(obj: Any) -> dict[str, Any]:
    """Extract the same telemetry from a result object/dataclass."""
    if obj is None:
        return {}
    details = {
        key: getattr(obj, key, None)
        for key in WEIGHTED_TELEMETRY_FIELDS
        if hasattr(obj, key)
    }
    if hasattr(obj, "weighted_market_reference_logodds"):
        details["weighted_market_reference_logodds"] = getattr(
            obj, "weighted_market_reference_logodds"
        )
    # Result objects use weighted_p_* names; preserve the presence marker even
    # when all values are null so a SKIP remains auditable.
    if not details.get("weighted_policy_mode") and hasattr(obj, "weighted_policy_mode"):
        details["weighted_policy_mode"] = getattr(obj, "weighted_policy_mode")
    return weighted_telemetry_from_details(details)
