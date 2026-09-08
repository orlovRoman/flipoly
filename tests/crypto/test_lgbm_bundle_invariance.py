"""Test regime bundle invariance (Item 1.11).

Ensures that training a candidate regime bundle does not alter active predictor
volatility tertile boundaries or predictions of already loaded active models,
and verifies multi-instance isolation across CryptoPredictor instances.
"""
import pytest
import numpy as np
import pickle
from unittest.mock import AsyncMock, MagicMock
from polyflip.crypto.predictor import CryptoPredictor
from polyflip.crypto.volatility import VolatilityRegimePolicy
from polyflip.crypto.feature_sets import CONTROL_FEATURES
from polyflip.db.models import ModelRegistry, RuntimeSettings


class RegimeModel:
    def __init__(self, proba_up: float):
        self.proba_up = proba_up
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        p = self.proba_up
        n = len(X) if hasattr(X, "__len__") else 1
        return np.tile(np.array([1.0 - p, p]), (n, 1))


def _build_model_row(symbol: str, regime: str, proba_up: float, p33: float, p67: float, bundle_id: str):
    r = MagicMock(spec=ModelRegistry)
    r.asset = f"{symbol}_{regime}"
    r.version = 1
    r.is_active = True
    r.model_type = "LightGBM"
    r.model_blob = pickle.dumps(RegimeModel(proba_up))
    r.features = ",".join(CONTROL_FEATURES)
    r.ece = 0.03
    r.decision_threshold = 0.55
    r.decision_threshold_down = 0.45
    r.interval = "15m"
    r.training_params = {
        "target_source": "POLYMARKET_FINAL_OUTCOME",
        "vol_p33": p33,
        "vol_p67": p67,
        "bundle_id": bundle_id,
        "bundle_version": "1.0.0",
    }
    return r


@pytest.mark.asyncio
async def test_candidate_train_does_not_mutate_active_predictor():
    """Active predictor (pred_active) is unaffected by candidate bundle training in separate instance."""
    pred_active = CryptoPredictor()
    pred_worker = CryptoPredictor()

    # Active bundle (v1: p33=0.50, p67=1.50)
    active_rows = {
        "low_vol": _build_model_row("BTCUSDT", "low_vol", 0.40, 0.50, 1.50, "bundle_v1"),
        "mid_vol": _build_model_row("BTCUSDT", "mid_vol", 0.60, 0.50, 1.50, "bundle_v1"),
        "high_vol": _build_model_row("BTCUSDT", "high_vol", 0.80, 0.50, 1.50, "bundle_v1"),
    }

    def make_db_mock(rows_by_regime, p33, p67):
        db = AsyncMock()
        async def mock_exec(stmt):
            stmt_str = str(stmt)
            res = MagicMock()
            if "RuntimeSettings" in stmt_str or "runtime_settings" in stmt_str:
                mock_row = MagicMock()
                if "P33" in stmt_str:
                    mock_row.value = str(p33)
                elif "P67" in stmt_str:
                    mock_row.value = str(p67)
                else:
                    mock_row.value = "0.0"
                res.scalar_one_or_none.return_value = mock_row
                return res

            if "modelregistry.asset, model_registry.version" in stmt_str.lower() or "model_registry.asset in" in stmt_str.lower():
                res.all.return_value = [
                    MagicMock(asset=f"BTCUSDT_{reg}", version=1)
                    for reg in ["low_vol", "mid_vol", "high_vol"]
                ]
                return res

            asset_val = None
            for crit in getattr(stmt, "_where_criteria", ()):
                left = getattr(crit, "left", None)
                if getattr(left, "name", None) == "asset":
                    right = getattr(crit, "right", None)
                    if hasattr(right, "value"):
                        asset_val = right.value
                        break

            matched_row = rows_by_regime.get("mid_vol")
            if asset_val:
                for reg in ["low_vol", "mid_vol", "high_vol"]:
                    if asset_val == f"BTCUSDT_{reg}":
                        matched_row = rows_by_regime[reg]
                        break

            mock_sc = MagicMock()
            mock_sc.first.return_value = matched_row
            res.scalars.return_value = mock_sc
            res.scalar_one_or_none.return_value = None
            return res

        db.execute = AsyncMock(side_effect=mock_exec)
        return db

    db_active = make_db_mock(active_rows, 0.50, 1.50)
    await pred_active.load(db_active, "BTCUSDT")

    # Initial active state
    assert pred_active._vol_p33s["BTCUSDT"] == 0.50
    assert pred_active._vol_p67s["BTCUSDT"] == 1.50
    mid_model_active = pred_active._models["BTCUSDT"]["mid_vol"]
    pred_before = mid_model_active.predict_proba(np.zeros((1, len(CONTROL_FEATURES))))[0, 1]
    assert pred_before == 0.60

    # Candidate bundle trained/loaded in worker (v2: p33=0.70, p67=2.10, proba=0.75)
    candidate_rows = {
        "low_vol": _build_model_row("BTCUSDT", "low_vol", 0.45, 0.70, 2.10, "bundle_v2_cand"),
        "mid_vol": _build_model_row("BTCUSDT", "mid_vol", 0.75, 0.70, 2.10, "bundle_v2_cand"),
        "high_vol": _build_model_row("BTCUSDT", "high_vol", 0.90, 0.70, 2.10, "bundle_v2_cand"),
    }
    db_candidate = make_db_mock(candidate_rows, 0.70, 2.10)
    await pred_worker.load(db_candidate, "BTCUSDT")

    # Worker instance reflects candidate parameters
    assert pred_worker._vol_p33s["BTCUSDT"] == 0.70
    assert pred_worker._vol_p67s["BTCUSDT"] == 2.10
    mid_model_worker = pred_worker._models["BTCUSDT"]["mid_vol"]
    pred_worker_out = mid_model_worker.predict_proba(np.zeros((1, len(CONTROL_FEATURES))))[0, 1]
    assert pred_worker_out == 0.75

    # INVARIANCE: Active predictor instance is completely unmodified
    assert pred_active._vol_p33s["BTCUSDT"] == 0.50
    assert pred_active._vol_p67s["BTCUSDT"] == 1.50
    assert pred_active._models["BTCUSDT"]["mid_vol"].predict_proba(np.zeros((1, len(CONTROL_FEATURES))))[0, 1] == 0.60

    # Invalidate all instances
    CryptoPredictor.invalidate_all("BTCUSDT")
    assert "BTCUSDT" not in pred_active._loaded_symbols
    assert "BTCUSDT" not in pred_worker._loaded_symbols

