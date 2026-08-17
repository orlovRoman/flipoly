import pytest
from unittest.mock import MagicMock
from polyflip.trading.decision_runners import resolve_logreg_model


def test_resolve_logreg_model_primary_phase():
    """Deployable phase model is selected as PRIMARY."""
    cache = MagicMock()
    cache.models = {"ETH_decided": object(), "ETH": object()}
    cache.versions = {"ETH_decided": 13, "ETH": 47}
    cache.features = {"ETH_decided": ["f1"], "ETH": ["f1"]}
    cache.deployable = {"ETH_decided": True, "ETH": True}

    model, status, reason, model_key, ver, feats = resolve_logreg_model(cache, "ETH", "decided")
    assert status == "PRIMARY"
    assert model_key == "ETH_decided"
    assert ver == 13
    assert reason is None


def test_resolve_logreg_model_fallback_to_base():
    """Undeployable phase model falls back to deployable base model."""
    cache = MagicMock()
    cache.models = {"ETH_decided": object(), "ETH": object()}
    cache.versions = {"ETH_decided": 13, "ETH": 47}
    cache.features = {"ETH_decided": ["f1"], "ETH": ["f1"]}
    cache.deployable = {"ETH_decided": False, "ETH": True}

    model, status, reason, model_key, ver, feats = resolve_logreg_model(cache, "ETH", "decided")
    assert status == "FALLBACK_BASE"
    assert model_key == "ETH"
    assert ver == 47
    assert reason == "PHASE_MODEL_NOT_DEPLOYABLE"


def test_resolve_logreg_model_abstains_when_both_undeployable():
    """Undeployable phase model and undeployable base model results in ABSTAIN."""
    cache = MagicMock()
    cache.models = {"ETH_decided": object(), "ETH": object()}
    cache.versions = {"ETH_decided": 13, "ETH": 47}
    cache.features = {"ETH_decided": ["f1"], "ETH": ["f1"]}
    cache.deployable = {"ETH_decided": False, "ETH": False}

    model, status, reason, model_key, ver, feats = resolve_logreg_model(cache, "ETH", "decided")
    assert model is None
    assert status == "ABSTAIN"
    assert reason == "BASE_AND_PHASE_NOT_DEPLOYABLE"
    assert model_key is None


def test_resolve_logreg_model_never_hops_cross_phase():
    """Cross-phase fallback (e.g. decided -> leaning) is strictly forbidden."""
    cache = MagicMock()
    cache.models = {"ETH_leaning": object()}
    cache.versions = {"ETH_leaning": 11}
    cache.features = {"ETH_leaning": ["f1"]}
    cache.deployable = {"ETH_leaning": True}

    # Requesting decided when only leaning exists -> must ABSTAIN, never use leaning
    model, status, reason, model_key, ver, feats = resolve_logreg_model(cache, "ETH", "decided")
    assert model is None
    assert status == "ABSTAIN"
