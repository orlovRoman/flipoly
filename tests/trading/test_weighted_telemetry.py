from polyflip.trading.weighted_telemetry import weighted_telemetry_from_details


def test_weighted_telemetry_uses_weighted_probability_when_canonical_is_null():
    result = weighted_telemetry_from_details(
        {
            "weighted_policy_mode": "WEIGHTED_SHADOW",
            "p_market_yes": None,
            "weighted_p_market_yes": 0.61,
            "p_logreg_yes": None,
            "weighted_p_logreg_yes": 0.72,
        }
    )
    assert result["p_market_yes"] == 0.61
    assert result["p_logreg_yes"] == 0.72


def test_weighted_telemetry_maps_runtime_market_reference_to_legacy_column():
    result = weighted_telemetry_from_details(
        {
            "weighted_policy_mode": "WEIGHTED_ACTIVE",
            "weighted_market_reference_logodds": 0.42,
        }
    )
    assert result["weighted_market_contribution_logodds"] == 0.42
    assert "weighted_market_reference_logodds" not in result
