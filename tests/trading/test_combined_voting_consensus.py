import pytest
from polyflip.trading.combined_voting import resolve_direction_consensus, DirectionConsensus

def test_agree_yes_yes():
    res = resolve_direction_consensus(lgbm_vote="UP", lr_vote="BUY_YES", require_consensus=True, fallback_to_logreg_on_none=True)
    assert res.final_side == "BUY_YES"
    assert res.consensus_type == "AGREE"

def test_agree_no_no():
    res = resolve_direction_consensus(lgbm_vote="DOWN", lr_vote="BUY_NO", require_consensus=True, fallback_to_logreg_on_none=True)
    assert res.final_side == "BUY_NO"
    assert res.consensus_type == "AGREE"

def test_conflict_skip():
    res = resolve_direction_consensus(lgbm_vote="UP", lr_vote="BUY_NO", require_consensus=True, fallback_to_logreg_on_none=True)
    assert res.final_side == "SKIP"
    assert res.consensus_type == "CONFLICT"

def test_conflict_lgbm_wins():
    res = resolve_direction_consensus(lgbm_vote="UP", lr_vote="BUY_NO", require_consensus=False, fallback_to_logreg_on_none=True)
    assert res.final_side == "BUY_YES"
    assert res.consensus_type == "CONFLICT"

def test_lgbm_none_logreg_wins():
    res = resolve_direction_consensus(lgbm_vote="NONE", lr_vote="BUY_YES", require_consensus=True, fallback_to_logreg_on_none=True)
    assert res.final_side == "BUY_YES"
    assert res.consensus_type == "PARTIAL_LR"

def test_lgbm_none_no_fallback():
    res = resolve_direction_consensus(lgbm_vote="NONE", lr_vote="BUY_YES", require_consensus=True, fallback_to_logreg_on_none=False)
    assert res.final_side == "SKIP"
    assert res.consensus_type == "PARTIAL_LR"

def test_logreg_abstain_lgbm_ok():
    res = resolve_direction_consensus(lgbm_vote="UP", lr_vote="ABSTAIN", require_consensus=True, fallback_to_logreg_on_none=True)
    assert res.final_side == "BUY_YES"
    assert res.consensus_type == "PARTIAL_LGBM"

def test_both_abstain():
    res = resolve_direction_consensus(lgbm_vote="NONE", lr_vote="ABSTAIN", require_consensus=True, fallback_to_logreg_on_none=True)
    assert res.final_side == "SKIP"
    assert res.consensus_type == "BOTH_ABSTAIN"
