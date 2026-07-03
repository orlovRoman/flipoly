import pytest
from polyflip.trading.position_sizing import (
    compute_edge, compute_bet_size_edge_scaled,
    compute_bet_size_with_liquidity, is_in_dead_zone, apply_polymarket_fee,
)
from polyflip.constants import INVALID_EDGE_SENTINEL, FLIP_MIDPOINT


class TestComputeEdge:
    def test_positive_edge(self):
        # win_prob=0.80, buy_price=0.70 → edge=0.80/0.70-1≈0.143
        assert compute_edge(0.80, 0.70) == pytest.approx(0.143, abs=1e-3)

    def test_zero_edge_at_fair_price(self):
        # win_prob=buy_price → edge=0
        assert compute_edge(0.75, 0.75) == pytest.approx(0.0, abs=1e-4)

    def test_negative_edge(self):
        assert compute_edge(0.60, 0.70) < 0

    def test_returns_sentinel_on_zero_price(self):
        assert compute_edge(0.80, 0.0) == INVALID_EDGE_SENTINEL

    def test_returns_sentinel_on_negative_price(self):
        assert compute_edge(0.80, -0.1) == INVALID_EDGE_SENTINEL


class TestBetSizing:
    def test_min_bet_at_min_edge(self):
        result = compute_bet_size_edge_scaled(
            edge=0.05, min_bet_usdc=5.0, max_bet_usdc=50.0,
            min_edge=0.05, max_edge=0.40
        )
        assert result == pytest.approx(5.0)

    def test_max_bet_at_max_edge(self):
        result = compute_bet_size_edge_scaled(
            edge=0.40, min_bet_usdc=5.0, max_bet_usdc=50.0,
            min_edge=0.05, max_edge=0.40
        )
        assert result == pytest.approx(50.0)

    def test_zero_on_negative_edge(self):
        result = compute_bet_size_edge_scaled(
            edge=-0.01, min_bet_usdc=5.0, max_bet_usdc=50.0
        )
        assert result == 0.0

    def test_linear_midpoint(self):
        # edge посередине → ставка посередине
        result = compute_bet_size_edge_scaled(
            edge=0.225, min_bet_usdc=5.0, max_bet_usdc=55.0,
            min_edge=0.05, max_edge=0.40
        )
        assert result == pytest.approx(30.0, abs=0.1)

    def test_liquidity_cap_applied(self):
        # volume_5min=10, liquidity_fraction=0.05 → cap=0.5 → min_bet=5 → cap=max(0.5,5)=5
        result = compute_bet_size_with_liquidity(
            edge=0.30, volume_5min=10.0,
            min_bet_usdc=5.0, max_bet_usdc=50.0,
            liquidity_fraction=0.05,
        )
        assert result == pytest.approx(5.0)

    def test_liquidity_cap_not_applied_on_high_volume(self):
        # volume_5min=10000 → cap=500 >> bet → не обрезается
        result = compute_bet_size_with_liquidity(
            edge=0.10, volume_5min=10_000.0,
            min_bet_usdc=5.0, max_bet_usdc=50.0,
        )
        assert result > 5.0


class TestDeadZone:
    def test_midpoint_is_dead_zone(self):
        assert is_in_dead_zone(FLIP_MIDPOINT, dead_zone_width=0.10) is True

    def test_above_dead_zone(self):
        assert is_in_dead_zone(0.56, dead_zone_width=0.10) is False

    def test_below_dead_zone(self):
        assert is_in_dead_zone(0.44, dead_zone_width=0.10) is False

    def test_boundary_edge(self):
        # abs(0.55 - 0.5) = 0.05 = 0.10/2 → NOT in dead zone (strict <)
        assert is_in_dead_zone(0.55, dead_zone_width=0.10) is False


class TestPolymarketFee:
    def test_fee_reduces_pnl(self):
        result = apply_polymarket_fee(100.0)
        assert result == pytest.approx(99.8)

    def test_custom_fee_rate(self):
        result = apply_polymarket_fee(100.0, fee_rate=0.01)
        assert result == pytest.approx(99.0)

    def test_zero_pnl(self):
        assert apply_polymarket_fee(0.0) == 0.0
