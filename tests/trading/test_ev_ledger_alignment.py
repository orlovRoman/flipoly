import numpy as np
import pytest

from polyflip.crypto.edge import (
    compute_economic_edge,
    compute_net_ev_per_share,
    compute_roi,
)


def test_ev_alignment_with_settled_win_trade():
    """Verify compute_net_ev_per_share matches exact realized ledger PnL for a WIN trade."""
    shares = 100.0
    ask_price = 0.55
    fee_rate = 0.015       # 1.5% taker fee
    slippage_rate = 0.005  # 0.5% slippage

    fee_per_share = ask_price * fee_rate
    slippage_per_share = ask_price * slippage_rate

    # Realized hold-to-settlement accounting for a WIN outcome (payout = $1.00 per share)
    entry_cost = shares * ask_price
    total_fee = shares * fee_per_share
    total_slippage = shares * slippage_per_share
    settlement_payout = shares * 1.00
    realized_pnl = settlement_payout - entry_cost - total_fee - total_slippage

    # Under p_win = 1.0, EV per share * shares must equal realized_pnl exactly
    ev_per_share = compute_net_ev_per_share(
        p_win=1.0,
        executable_ask=ask_price,
        fee_per_share=fee_per_share,
        slippage_per_share=slippage_per_share,
    )
    expected_pnl = ev_per_share * shares

    assert np.isclose(expected_pnl, realized_pnl, atol=1e-5)
    assert np.isclose(ev_per_share, 1.00 - ask_price - fee_per_share - slippage_per_share, atol=1e-6)


def test_ev_alignment_with_settled_loss_trade():
    """Verify compute_net_ev_per_share matches exact realized ledger PnL for a LOSS trade."""
    shares = 200.0
    ask_price = 0.40
    fee_rate = 0.02
    slippage_rate = 0.01

    fee_per_share = ask_price * fee_rate
    slippage_per_share = ask_price * slippage_rate

    # Realized hold-to-settlement accounting for a LOSS outcome (payout = $0.00 per share)
    entry_cost = shares * ask_price
    total_fee = shares * fee_per_share
    total_slippage = shares * slippage_per_share
    settlement_payout = 0.0
    realized_pnl = settlement_payout - entry_cost - total_fee - total_slippage

    # Under p_win = 0.0, EV per share * shares must equal realized loss exactly
    ev_per_share = compute_net_ev_per_share(
        p_win=0.0,
        executable_ask=ask_price,
        fee_per_share=fee_per_share,
        slippage_per_share=slippage_per_share,
    )
    expected_pnl = ev_per_share * shares

    assert np.isclose(expected_pnl, realized_pnl, atol=1e-5)
    assert np.isclose(ev_per_share, -ask_price - fee_per_share - slippage_per_share, atol=1e-6)


def test_expected_value_linearity_over_replay_ledger():
    """Verify mathematical expectation theorem: mean ledger PnL converges to theoretical EV sum."""
    rng = np.random.default_rng(42)
    n_trades = 1000
    shares_per_trade = 50.0
    fee_rate = 0.01
    slippage_rate = 0.005

    p_wins = rng.uniform(0.40, 0.75, size=n_trades)
    asks = rng.uniform(0.35, 0.65, size=n_trades)

    theoretical_ev_sum = 0.0
    for p_w, ask in zip(p_wins, asks):
        net_ev = compute_economic_edge(
            p_win=p_w,
            executable_ask=ask,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
        )
        theoretical_ev_sum += net_ev * shares_per_trade

    # Simulate 500 Monte Carlo replays of the ledger outcomes
    n_sims = 500
    realized_pnls = []
    for _ in range(n_sims):
        outcomes = rng.binomial(n=1, p=p_wins)
        total_pnl = 0.0
        for y, ask in zip(outcomes, asks):
            payout = shares_per_trade * 1.0 if y == 1 else 0.0
            cost = shares_per_trade * ask * (1.0 + fee_rate + slippage_rate)
            total_pnl += payout - cost
        realized_pnls.append(total_pnl)

    mean_realized_pnl = float(np.mean(realized_pnls))

    # Expectation convergence within 2 standard errors
    se = float(np.std(realized_pnls) / np.sqrt(n_sims))
    assert abs(mean_realized_pnl - theoretical_ev_sum) < 2.5 * se


def test_no_spread_double_counting_in_ev():
    """Verify executable ask already includes spread half-cost without subtracting spread again."""
    mid = 0.50
    half_spread = 0.02
    ask = mid + half_spread  # 0.52

    ev = compute_net_ev_per_share(p_win=0.60, executable_ask=ask)
    # Correct net EV is p_win - ask = 0.60 - 0.52 = 0.08
    assert np.isclose(ev, 0.08)

    # If someone erroneously subtracted the full spread (0.04) again:
    erroneous_ev = 0.60 - ask - 2 * half_spread  # 0.04
    assert not np.isclose(ev, erroneous_ev)


def test_compute_roi_and_boundary_guards():
    """Verify return on capital ROI calculation and un-executable price boundaries."""
    # Normal ROI
    net_ev = compute_net_ev_per_share(p_win=0.60, executable_ask=0.50)
    roi = compute_roi(net_ev, executable_ask=0.50)
    assert np.isclose(roi, 0.10 / 0.50)  # +20% expected return on capital

    # Boundary guards
    assert compute_net_ev_per_share(p_win=0.8, executable_ask=1.0) == 0.0
    assert compute_net_ev_per_share(p_win=0.8, executable_ask=1.05) == 0.0
    assert compute_net_ev_per_share(p_win=0.8, executable_ask=0.0) == 0.0
    assert compute_net_ev_per_share(p_win=0.8, executable_ask=-0.1) == 0.0

    # Economic edge boundary guards
    assert compute_economic_edge(0.7, 1.0, fee_rate=0.01, slippage_rate=0.01) == 0.0
    assert compute_economic_edge(0.7, -0.5, fee_rate=0.01, slippage_rate=0.01) == 0.0
    assert compute_roi(0.05, 0.0) == 0.0
