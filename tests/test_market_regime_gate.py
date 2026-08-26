import math

import pytest

from polyflip.crypto.market_regime_classifier import (
    MarketPhase,
    RegimeClassification,
)
from polyflip.crypto.market_regime_policy import (
    VetoGateConfig,
    compute_veto_gate,
)


def classification(
    phase: MarketPhase,
    strength: float = 0.8,
    confidence: float = 0.8,
) -> RegimeClassification:
    direction = {
        MarketPhase.STRONG_UP: 1.0,
        MarketPhase.WEAK_UP: 1.0,
        MarketPhase.STRONG_DOWN: -1.0,
        MarketPhase.WEAK_DOWN: -1.0,
    }.get(phase, 0.0)
    return RegimeClassification(
        phase=phase,
        strength=strength,
        confidence=confidence,
        direction=direction,
        reason_codes=[],
    )


def run_gate(
    asset_phase: MarketPhase,
    global_phase: MarketPhase,
    candidate_direction: float,
    *,
    asset_strength: float = 0.8,
    asset_confidence: float = 0.8,
    global_strength: float = 0.8,
    global_confidence: float = 0.8,
    net_edge: float = 0.05,
    min_edge_used: float = 0.04,
    config: VetoGateConfig | None = None,
):
    return compute_veto_gate(
        asset_classification=classification(
            asset_phase, asset_strength, asset_confidence,
        ),
        global_classification=classification(
            global_phase, global_strength, global_confidence,
        ),
        candidate_direction=candidate_direction,
        net_edge=net_edge,
        min_edge_used=min_edge_used,
        config=config or VetoGateConfig(),
    )


def test_supporting_asset_and_global_regimes_pass():
    result = run_gate(
        MarketPhase.STRONG_UP,
        MarketPhase.WEAK_UP,
        1.0,
    )
    assert result.would_block is False
    assert result.reason == "regime_supports_candidate"
    assert result.regime_evidence > 0


def test_both_regimes_against_buy_yes_block():
    result = run_gate(
        MarketPhase.STRONG_DOWN,
        MarketPhase.WEAK_DOWN,
        1.0,
    )
    assert result.would_block is True
    assert result.reason == "regime_veto"


def test_both_regimes_against_buy_no_block():
    result = run_gate(
        MarketPhase.STRONG_UP,
        MarketPhase.WEAK_UP,
        -1.0,
    )
    assert result.would_block is True


def test_mixed_weighted_evidence_can_still_veto():
    result = run_gate(
        MarketPhase.STRONG_DOWN,
        MarketPhase.STRONG_UP,
        1.0,
        asset_strength=0.8,
        asset_confidence=0.8,
        global_strength=0.2,
        global_confidence=0.2,
    )
    assert result.regime_evidence < 0
    assert result.would_block is True


def test_neutral_phases_have_no_evidence_and_pass():
    result = run_gate(
        MarketPhase.MIXED,
        MarketPhase.SIDEWAYS,
        1.0,
    )
    assert result.regime_evidence == 0
    assert result.would_block is False
    assert result.reason == "no_negative_regime_evidence"


def test_neutral_phases_pass_even_with_zero_veto_threshold():
    result = run_gate(
        MarketPhase.MIXED,
        MarketPhase.SIDEWAYS,
        1.0,
        config=VetoGateConfig(veto_threshold=0.0),
    )
    assert result.regime_evidence == 0
    assert result.would_block is False


def test_large_edge_margin_overrides_negative_regime():
    result = run_gate(
        MarketPhase.STRONG_DOWN,
        MarketPhase.STRONG_DOWN,
        1.0,
        net_edge=0.20,
        min_edge_used=0.04,
    )
    assert result.would_block is False
    assert result.reason == "strong_edge_override"


def test_veto_threshold_boundary_is_inclusive():
    result = run_gate(
        MarketPhase.WEAK_DOWN,
        MarketPhase.WEAK_DOWN,
        1.0,
        asset_strength=0.5,
        asset_confidence=0.6,
        global_strength=0.5,
        global_confidence=0.6,
        config=VetoGateConfig(veto_threshold=0.30),
    )
    assert result.regime_evidence == pytest.approx(-0.30)
    assert result.would_block is True


def test_edge_override_boundary_is_inclusive():
    result = run_gate(
        MarketPhase.STRONG_DOWN,
        MarketPhase.STRONG_DOWN,
        1.0,
        net_edge=0.09,
        min_edge_used=0.04,
        config=VetoGateConfig(edge_override_margin=0.05),
    )
    assert result.edge_margin == pytest.approx(0.05)
    assert result.would_block is False
    assert result.reason == "strong_edge_override"


def test_invalid_gate_config_and_inputs_are_rejected():
    with pytest.raises(ValueError, match="weight sum"):
        VetoGateConfig(asset_weight=0.0, global_weight=0.0)
    with pytest.raises(ValueError, match="non-finite"):
        VetoGateConfig(veto_threshold=math.nan)
    with pytest.raises(ValueError, match="candidate_direction"):
        run_gate(MarketPhase.SIDEWAYS, MarketPhase.SIDEWAYS, 0.0)
    with pytest.raises(ValueError, match="non-finite"):
        run_gate(MarketPhase.SIDEWAYS, MarketPhase.SIDEWAYS, 1.0, net_edge=math.inf)
    with pytest.raises(ValueError, match="classification"):
        run_gate(
            MarketPhase.STRONG_UP,
            MarketPhase.STRONG_UP,
            1.0,
            asset_strength=math.nan,
        )


def test_missing_classification_value_is_rejected_as_non_finite():
    asset = classification(MarketPhase.STRONG_UP, strength=None)
    with pytest.raises(ValueError, match="classification contains non-finite"):
        compute_veto_gate(
            asset_classification=asset,
            global_classification=classification(MarketPhase.STRONG_UP),
            candidate_direction=1.0,
            net_edge=0.10,
            min_edge_used=0.05,
            config=VetoGateConfig(),
        )
