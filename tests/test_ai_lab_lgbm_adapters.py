import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from polyflip.ai_lab.executor import StepContext
from polyflip.ai_lab import lgbm_adapters


def _context(**overrides):
    payload = {
        "run_id": 1,
        "step_id": 2,
        "action": "RUN_POLYMARKET_OOT",
        "config_id": 9,
        "config_hash": "hash",
        "objective": "compare LightGBM candidates",
        "scope": {},
        "input_payload": {},
        "model_family": "LIGHTGBM",
        "feature_set": "B",
        "asset": "BTC",
        "regime": None,
        "interval": "15m",
        "model_params": {},
        "calibration_params": {},
        "strategy_params": {"strategy_branch": "COMBINED"},
        "backtest_params": {},
    }
    payload.update(overrides)
    return StepContext(**payload)


def test_registry_contains_only_offline_lgbm_actions():
    registry = lgbm_adapters.build_lgbm_adapter_registry(object())
    assert registry.actions() == (
        "RUN_OOT_BACKTEST",
        "RUN_POLYMARKET_OOT",
        "TRAIN_MODEL",
    )


def test_normalized_config_uses_saved_strategy_calibration():
    context = _context(
        feature_set="C",
        model_params={"n_estimators": 100},
        strategy_params={"calibration": {"method": "PLATT"}},
        backtest_params={"min_edge": 0.05},
    )

    config = lgbm_adapters._normalized_config(context)

    assert config["feature_set"] == "C"
    assert config["model"]["n_estimators"] == 100
    assert config["calibration"]["method"] == "PLATT"
    assert config["backtest"]["min_edge"] == 0.05


def test_polymarket_oot_replays_saved_canonical_variant(monkeypatch):
    row = SimpleNamespace(
        id=41,
        asset="BTCUSDT_low_vol",
        training_params={
            "target_source": "POLYMARKET_FINAL_OUTCOME",
            "backtest_pnl_mode": "POLYMARKET_OOF",
            "backtest_variants": {
                "COMBINED": {
                    "n_markets": 10,
                    "n_trades": 2,
                    "net_profit": 1.25,
                    "max_drawdown_usdc": 0.2,
                    "total_invested": 2.0,
                    "win_rate": 0.5,
                    "slices": [],
                }
            },
        },
        dataset_fingerprint="fp-41",
        training_window_start=datetime(2026, 8, 1, tzinfo=timezone.utc),
        training_window_end=datetime(2026, 8, 2, tzinfo=timezone.utc),
        accuracy=0.74,
        ece=0.02,
        brier_score=0.24,
        train_samples=100,
        validation_samples=20,
    )

    async def load_rows(_session, _context):
        return [row]

    def aggregate(_results, *, strategy_branch):
        assert strategy_branch == "COMBINED"
        return {
            "strategy_branch": strategy_branch,
            "n_markets": 10,
            "n_quotes": 10,
            "n_oof": 10,
            "n_eligible": 2,
            "n_trades": 2,
            "net_profit": 1.25,
            "max_drawdown_usdc": 0.2,
            "roi_pct": 62.5,
            "win_rate": 0.5,
            "avg_edge": 0.1,
            "avg_net_edge": 0.08,
            "avg_entry_price": 0.3,
            "coverage_pct": 100.0,
            "sharpe_ratio": 1.0,
            "profit_factor": 2.0,
            "slices": [{"dimension": "PRICE", "bucket": "0.20-0.35"}],
        }

    monkeypatch.setattr(lgbm_adapters, "_training_rows", load_rows)
    monkeypatch.setattr(lgbm_adapters, "aggregate_stored_polymarket_backtests", aggregate)

    result = asyncio.run(
        lgbm_adapters.run_lgbm_polymarket_oot(_context(), object())
    )

    assert result.status == "SUCCEEDED"
    assert result.trade_count == 2
    assert result.net_pnl == pytest.approx(1.25)
    assert result.max_drawdown == pytest.approx(0.2)
    assert result.metrics["strategy_branch"] == "COMBINED"
    assert result.slices["slices"][0]["bucket"] == "0.20-0.35"


def test_polymarket_oot_rejects_noncanonical_training_metadata(monkeypatch):
    row = SimpleNamespace(
        id=41,
        asset="BTCUSDT_low_vol",
        training_params={
            "target_source": "BINANCE_NEXT_RETURN",
            "backtest_pnl_mode": "BINANCE",
            "backtest_variants": {"COMBINED": {"n_markets": 10}},
        },
        dataset_fingerprint="legacy",
        training_window_start=None,
        training_window_end=None,
    )

    async def load_rows(_session, _context):
        return [row]

    monkeypatch.setattr(lgbm_adapters, "_training_rows", load_rows)

    result = asyncio.run(
        lgbm_adapters.run_lgbm_polymarket_oot(_context(), object())
    )

    assert result.status == "INSUFFICIENT_DATA"
    assert result.error_code == "POLYMARKET_OOT_MISSING"


class _ScalarResult:
    def __init__(self, values):
        self.values = values

    def scalars(self):
        return self

    def all(self):
        return list(self.values)


class _TrainingSession:
    def __init__(self, rows):
        self.rows = rows
        self.calls = 0
        self.rollback_count = 0

    async def execute(self, _statement):
        self.calls += 1
        return _ScalarResult([] if self.calls == 1 else self.rows)

    def add(self, _value):
        return None

    async def flush(self):
        return None

    async def rollback(self):
        self.rollback_count += 1


def test_training_adapter_disables_activation_and_runtime_writes(monkeypatch):
    row = SimpleNamespace(
        id=55,
        asset="BTCUSDT_low_vol",
        version=1,
        model_blob=b"model",
        is_active=False,
        dataset_fingerprint="fp",
        training_window_start=None,
        training_window_end=None,
        accuracy=0.71,
        ece=0.03,
        brier_score=0.24,
        train_samples=80,
        validation_samples=20,
        training_params={"feature_set": "B"},
    )
    session = _TrainingSession([row])
    calls = {}

    class FakeTrainer:
        def __init__(self, db):
            calls["session"] = db

        async def train(self, *args, **kwargs):
            calls["args"] = args
            calls["kwargs"] = kwargs
            return True

    async def fake_artifact(_session, _context, _rows, _config):
        return SimpleNamespace(id=700)

    monkeypatch.setattr(lgbm_adapters, "CryptoModelTrainer", FakeTrainer)
    monkeypatch.setattr(lgbm_adapters, "_create_bundle_artifact", fake_artifact)

    result = asyncio.run(
        lgbm_adapters.train_lgbm(
            _context(action="TRAIN_MODEL", regime="low_vol"),
            session,
        )
    )

    assert result.status == "SUCCEEDED"
    assert calls["kwargs"]["save_settings"] is False
    assert calls["kwargs"]["activate_after_train"] is False
    assert calls["kwargs"]["experiment_config_id"] is None


def test_training_adapter_rolls_back_active_candidate_safety_violation(monkeypatch):
    row = SimpleNamespace(
        id=56,
        asset="BTCUSDT_low_vol",
        version=1,
        model_blob=b"active-model",
        is_active=True,
        dataset_fingerprint="fp-active",
        training_window_start=None,
        training_window_end=None,
        accuracy=0.71,
        ece=0.03,
        brier_score=0.24,
        train_samples=80,
        validation_samples=20,
        training_params={"feature_set": "B"},
    )
    session = _TrainingSession([row])

    class FakeTrainer:
        def __init__(self, _db):
            pass

        async def train(self, *_args, **_kwargs):
            return True

    monkeypatch.setattr(lgbm_adapters, "CryptoModelTrainer", FakeTrainer)

    with pytest.raises(RuntimeError, match="active LightGBM"):
        asyncio.run(
            lgbm_adapters.train_lgbm(
                _context(action="TRAIN_MODEL", regime="low_vol"),
                session,
            )
        )

    assert session.rollback_count == 1
