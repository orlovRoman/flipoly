from polyflip.trading.policy_contract import effective_policy_snapshot


def test_policy_contract_hash_is_order_independent_and_marks_legacy():
    left = effective_policy_snapshot(
        {
            "TRADING_MODE": "combined",
            "TRADING_MODE_BTC": "ct_outsider",
            "MIN_DIRECTION_PROB": "0.20",
            "COMBINED_DIR_STRONG_THRESHOLD": "0.30",
            "COMBINED_REQUIRE_CONSENSUS": "true",
            "COMBINED_FALLBACK_TO_LOGREG_ON_NONE": "true",
            "OUTSIDER_PWIN_DISCOUNT": "0.65",
        }
    )
    right = effective_policy_snapshot(
        {
            "OUTSIDER_PWIN_DISCOUNT": "0.65",
            "COMBINED_FALLBACK_TO_LOGREG_ON_NONE": "true",
            "COMBINED_REQUIRE_CONSENSUS": "true",
            "MIN_DIRECTION_PROB": "0.20",
            "COMBINED_DIR_STRONG_THRESHOLD": "0.30",
            "TRADING_MODE": "combined",
            "TRADING_MODE_BTC": "ct_outsider",
        }
    )
    assert left["policy_hash"] == right["policy_hash"]
    assert left["active"]["model_gates"]["min_direction_prob"] == 0.505
    assert left["active"]["routing"]["per_asset_trading_mode"] == {"BTC": "ct_outsider"}
    assert left["legacy"]["OUTSIDER_PWIN_DISCOUNT"]["status"] == "LEGACY"
    assert any(item["code"] == "FALLBACK_INACTIVE_UNDER_CONSENSUS" for item in left["warnings"])
    assert any(item["code"] == "STRONG_THRESHOLD_CLAMPED" for item in left["warnings"])


def test_policy_contract_changes_when_active_value_changes():
    base = effective_policy_snapshot({"MIN_DIRECTION_PROB": "0.505"})
    changed = effective_policy_snapshot({"MIN_DIRECTION_PROB": "0.60"})
    assert base["policy_hash"] != changed["policy_hash"]
