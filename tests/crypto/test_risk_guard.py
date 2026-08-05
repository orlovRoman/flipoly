import pytest
from polyflip.crypto.risk_guard import check_funding_veto, FUNDING_EXTREME_THRESHOLD

class TestFundingVeto:

    def test_normal_funding_no_veto(self):
        v = check_funding_veto(0.0001, 'UP')
        assert not v.vetoed
        assert v.stake_multiplier == 1.0

    def test_extreme_positive_with_crowd_vetoed(self):
        v = check_funding_veto(0.0006, 'UP')
        assert v.vetoed
        assert v.stake_multiplier == 0.0

    def test_extreme_positive_against_crowd_allowed(self):
        v = check_funding_veto(0.0006, 'DOWN')
        assert not v.vetoed
        assert v.stake_multiplier == 0.75

    def test_extreme_negative_with_crowd_vetoed(self):
        v = check_funding_veto(-0.0006, 'DOWN')
        assert v.vetoed

    def test_extreme_negative_against_crowd_allowed(self):
        v = check_funding_veto(-0.0006, 'UP')
        assert not v.vetoed

    def test_boundary_exactly_at_threshold(self):
        v = check_funding_veto(FUNDING_EXTREME_THRESHOLD, 'UP')
        assert v.vetoed

    def test_none_direction_does_not_crash(self):
        v = check_funding_veto(0.001, 'NONE')
        assert not v.vetoed