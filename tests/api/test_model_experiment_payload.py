from datetime import datetime, timezone
from types import SimpleNamespace

from polyflip.api.analytics import _model_experiment_payload


def _model(**overrides):
    values = {
        "asset": "BTC",
        "version": 7,
        "accuracy": 0.57,
        "baseline": 0.50,
        "ece": 0.04,
        "brier_score": 0.23,
        "features": "mid_price,direction_lag_1",
        "training_params": {
            "target_source": "POLYMARKET_FLIP_VS_FINAL_OUTCOME",
            "validation_scheme": "GROUPED_WALK_FORWARD",
            "feature_set_version": "B-direction-sequence-v1",
            "backtest_strategy_branch": "OUTSIDER_ONLY",
            "model_config": {"C": 0.1, "penalty": "l2"},
            "log_loss": 0.66,
            "oot_markets": 42,
        },
        "quality_gate_reasons": {"backtest": {}},
        "training_window_start": datetime(2025, 1, 1, tzinfo=timezone.utc),
        "training_window_end": datetime(2025, 2, 1, tzinfo=timezone.utc),
        "backtest_pnl": 12.5,
        "backtest_trades": 30,
        "backtest_wr": 0.6,
        "is_active": False,
        "trained_at": datetime(2025, 2, 2, tzinfo=timezone.utc),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_model_payload_exposes_persisted_oot_experiment_metadata():
    payload = _model_experiment_payload(_model())

    assert payload["feature_set_version"] == "B-direction-sequence-v1"
    assert payload["validation_scheme"] == "GROUPED_WALK_FORWARD"
    assert payload["target_source"] == "POLYMARKET_FLIP_VS_FINAL_OUTCOME"
    assert payload["strategy_branch"] == "OUTSIDER_ONLY"
    assert payload["oot_markets"] == 42
    assert payload["backtest_pnl"] == 12.5
    assert payload["log_loss"] == 0.66


def test_comparison_key_changes_when_oot_window_changes():
    original = _model_experiment_payload(_model())
    changed = _model_experiment_payload(
        _model(training_window_end=datetime(2025, 2, 2, tzinfo=timezone.utc))
    )

    assert original["comparison_key"] != changed["comparison_key"]


def test_feature_set_does_not_change_comparison_key():
    original = _model_experiment_payload(_model())
    params = dict(_model().training_params, feature_set_version="C-candle-structure-v1")
    candidate = _model_experiment_payload(_model(training_params=params))

    assert original["comparison_key"] == candidate["comparison_key"]
