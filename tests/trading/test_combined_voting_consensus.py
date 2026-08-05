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

def test_logreg_direction_vote_abstain_band():
    from polyflip.trading.combined_voting import logreg_direction_vote
    # With default 0.05 band: |0.52 - 0.50| = 0.02 < 0.05 -> ABSTAIN
    assert logreg_direction_vote(p_flip=0.52, fresh_yes_price=0.60) == "ABSTAIN"
    # With tight 0.01 band: |0.52 - 0.50| = 0.02 >= 0.01 -> BUY_NO (since yes_fav and p_flip > 0.5)
    assert logreg_direction_vote(p_flip=0.52, fresh_yes_price=0.60, abstain_band=0.01) == "BUY_NO"
    # With wider 0.10 band: |0.58 - 0.50| = 0.08 < 0.10 -> ABSTAIN
    assert logreg_direction_vote(p_flip=0.58, fresh_yes_price=0.60, abstain_band=0.10) == "ABSTAIN"

def test_combine_votes_docstring_present():
    from polyflip.trading.combined_voting import combine_votes
    assert combine_votes.__doc__ is not None
    assert len(combine_votes.__doc__.strip()) > 0
