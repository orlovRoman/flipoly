from types import SimpleNamespace

from polyflip.api.mrf_api import _classify_mrf_row


def _row(**overrides):
    values = {
        "mrf_failure_reason": None,
        "mrf_policy_version": None,
        "mrf_multiplier": None,
        "mrf_final_action": "BUY_YES",
        "mrf_original_action": "BUY_YES",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_v2_multiplier_is_classified_as_reduced():
    result = _classify_mrf_row(
        _row(mrf_policy_version=2, mrf_multiplier=0.5),
        telemetry={"policy_version": 2},
    )

    assert result == {
        "category": "reduced",
        "would_block": False,
        "blocked": False,
        "reduced": True,
    }


def test_v3_negative_evidence_without_veto_is_not_reduced():
    result = _classify_mrf_row(
        _row(mrf_policy_version="3", mrf_multiplier=None),
        telemetry={
            "policy_version": "3",
            "regime_evidence": -0.10,
            "gate_would_block": False,
        },
    )

    assert result["category"] == "passed"
    assert result["would_block"] is False
    assert result["blocked"] is False
    assert result["reduced"] is False


def test_v3_shadow_would_block_is_not_an_actual_block():
    result = _classify_mrf_row(
        _row(mrf_policy_version=3),
        telemetry={"policy_version": 3, "gate_would_block": True},
    )

    assert result["category"] == "passed"
    assert result["would_block"] is True
    assert result["blocked"] is False
    assert result["reduced"] is False


def test_v3_active_veto_counts_as_would_block_and_blocked():
    result = _classify_mrf_row(
        _row(
            mrf_policy_version=3,
            mrf_final_action="SKIP",
            mrf_original_action="BUY_NO",
        ),
        telemetry={"policy_version": 3, "gate_would_block": True},
    )

    assert result["category"] == "blocked"
    assert result["would_block"] is True
    assert result["blocked"] is True
    assert result["reduced"] is False
