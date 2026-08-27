"""Regression coverage for market-role outputs from free OpenCode models."""

from services.ai_research_agent.opencode_client import (
    _coerce_kv_lists,
    _hypothesis_schema,
)


def test_taker_market_role_is_normalized_to_outsider() -> None:
    assert _coerce_kv_lists({"market_role": " taker "})["market_role"] == "OUTSIDER"


def test_hypothesis_schema_restricts_market_roles() -> None:
    assert _hypothesis_schema()["properties"]["market_role"]["enum"] == ["FAVORITE", "OUTSIDER", "COMBINED", "DIRECTION_ONLY", "ALL"]
