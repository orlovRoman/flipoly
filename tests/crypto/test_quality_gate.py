"""Behavioral tests for Trainer Quality Gate validation."""
import pickle
import copy

import numpy as np
import pytest

from polyflip.crypto.trainer import CRYPTO_FEATURES, _evaluate_quality_gate


class _SmokeModel:
    n_features_in_ = len(CRYPTO_FEATURES)

    def __init__(self, probabilities=(0.4, 0.6)):
        self.probabilities = probabilities

    def predict_proba(self, rows):
        return np.asarray([self.probabilities] * len(rows), dtype=float)


def _evaluate(**overrides):
    base_backtest = {
        "COMBINED": {
            "net_profit": 100.0,
            "n_trades": 60,
            "max_drawdown_usdc": 10.0,
            "coverage_reasons": {"missing_close_time": 0},
            "oot_windows": [{"net_profit": 10.0, "n_trades": 10}, {"net_profit": 20.0, "n_trades": 15}],
        },
        "FAVORITE_ONLY": {
            "net_profit": 50.0,
        },
        "OUTSIDER_ONLY": {
            "net_profit": 50.0,
        }
    }

    values = {
        "model_bytes": pickle.dumps(_SmokeModel()),
        "val_auc": 0.62,
        "baseline_auc": 0.55,
        "ece": 0.08,
        "threshold": 0.55,
        "threshold_down": 0.53,
        "active_accuracy": 0.61,
        "active_version": 4,
        "backtest_variants": base_backtest,
    }

    # Handle deep update for backtest_variants
    if "backtest_variants" in overrides:
        merged_backtest = copy.deepcopy(base_backtest)
        for k, v in overrides["backtest_variants"].items():
            if k in merged_backtest:
                merged_backtest[k].update(v)
        values["backtest_variants"] = merged_backtest
        del overrides["backtest_variants"]

    values.update(overrides)
    return _evaluate_quality_gate(**values)


def test_valid_model_passes_quality_gate():
    passed, reasons, threshold_up, threshold_down = _evaluate()

    assert passed is True
    assert reasons == []
    assert threshold_up == pytest.approx(0.55)
    assert threshold_down == pytest.approx(0.53)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"ece": 0.16}, "ECE_TOO_HIGH"),
        ({"backtest_variants": {"COMBINED": {"net_profit": -5.0}}}, "PNL_NEGATIVE"),
        ({"backtest_variants": {"FAVORITE_ONLY": {"net_profit": -15.0}}}, "PNL_NEGATIVE"),
        ({"backtest_variants": {"COMBINED": {"n_trades": 40}}}, "INSUFFICIENT_TRADES"),
        ({"backtest_variants": {"COMBINED": {"max_drawdown_usdc": 25.0}}}, "MAX_DRAWDOWN_TOO_HIGH"),
        ({"backtest_variants": {"COMBINED": {"oot_windows": [{"net_profit": -5.0, "n_trades": 12}, {"net_profit": 10.0, "n_trades": 15}]}}}, "OOT_WINDOW_UNSTABLE"),
        ({"backtest_variants": {"COMBINED": {"coverage_reasons": {"missing_close_time": 5}}}}, "MISSING_CLOSE_TIME"),
        # New missing metric cases
        ({"backtest_variants": {"COMBINED": {"net_profit": None}}}, "PNL_NEGATIVE: COMBINED net_profit is missing"),
        ({"backtest_variants": {"FAVORITE_ONLY": {"net_profit": None}}}, "PNL_NEGATIVE: FAVORITE_ONLY net_profit is missing"),
        ({"backtest_variants": {"OUTSIDER_ONLY": {"net_profit": None}}}, "PNL_NEGATIVE: OUTSIDER_ONLY net_profit is missing"),
        ({"backtest_variants": {"COMBINED": {"n_trades": None}}}, "INSUFFICIENT_TRADES: missing < 50"),
        ({"backtest_variants": {"COMBINED": {"max_drawdown_usdc": None}}}, "MAX_DRAWDOWN_TOO_HIGH: missing > 20.00"),
        ({"backtest_variants": {"COMBINED": {"oot_windows": []}}}, "OOT_WINDOW_UNSTABLE: 0 positive OOT windows out of 0"),
        ({"backtest_variants": {"COMBINED": {"oot_windows": [{"net_profit": 10.0, "n_trades": 10}, {"n_trades": 1}]}}}, "OOT_WINDOW_UNSTABLE: 1 positive OOT windows out of 2"),
    ],
)
def test_metric_failures_are_audited(overrides, reason):
    passed, reasons, _, _ = _evaluate(**overrides)

    assert passed is False
    assert any(reason in item for item in reasons)


def test_both_decision_thresholds_are_validated_and_sanitized():
    passed, reasons, threshold_up, threshold_down = _evaluate(
        threshold=-0.2,
        threshold_down=1.2,
    )

    assert passed is False
    assert threshold_up == pytest.approx(0.0)
    assert threshold_down == pytest.approx(1.0)
    assert any("UP threshold" in item for item in reasons)
    assert any("DOWN threshold" in item for item in reasons)


@pytest.mark.parametrize(
    "probabilities",
    [
        (float("nan"), 0.5),
        (-0.1, 1.1),
        (0.2, 0.2),
    ],
)
def test_smoke_test_rejects_invalid_probabilities(probabilities):
    passed, reasons, _, _ = _evaluate(
        model_bytes=pickle.dumps(_SmokeModel(probabilities)),
    )

    assert passed is False
    assert any("invalid predict_proba result" in item for item in reasons)
