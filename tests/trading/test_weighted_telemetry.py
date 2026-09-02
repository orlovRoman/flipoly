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
