"""
tests/crypto/test_quality_gate.py

Тесты Quality Gate проверки в trainer.py и ручного обхода (quality_override) в crypto_dashboard.py.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone

from polyflip.db.models import ModelRegistry


def test_quality_gate_passed_audit_fields():
    model = ModelRegistry(
        asset="BTCUSDT_low_vol",
        version=5,
        model_type="lgbm",
        accuracy=0.62,
        baseline=0.51,
        ece=0.08,
        quality_gate_passed=True,
        quality_gate_reasons={"reasons": [], "auc": 0.62, "ece": 0.08},
        activation_source="TRAINER",
        quality_override=False,
        activated_at=datetime.now(timezone.utc),
        activated_by="trainer",
        is_active=True,
    )
    assert model.quality_gate_passed is True
    assert model.activation_source == "TRAINER"
    assert model.quality_override is False
    assert model.is_active is True


def test_quality_gate_failed_audit_fields():
    model = ModelRegistry(
        asset="ETHUSDT_high_vol",
        version=3,
        model_type="lgbm",
        accuracy=0.48,
        baseline=0.52,
        ece=0.22,
        quality_gate_passed=False,
        quality_gate_reasons={"reasons": ["Negative lift vs baseline", "Excessive ECE: 0.22 > 0.15"], "auc": 0.48, "ece": 0.22},
        activation_source=None,
        quality_override=False,
        activated_at=None,
        activated_by=None,
        is_active=False,
    )
    assert model.quality_gate_passed is False
    assert model.activation_source is None
    assert model.quality_override is False
    assert model.is_active is False
    assert len(model.quality_gate_reasons["reasons"]) == 2
