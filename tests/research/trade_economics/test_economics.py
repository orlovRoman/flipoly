import pytest
from polyflip.research.trade_economics.commissions import calculate_commission
from polyflip.research.trade_economics.pnl import apply_fee_to_budget, calculate_net_pnl

def test_calculate_commission_fixed():
    scheme = {
        'evidence_status': 'HISTORICAL',
        'formula_id': 'FIXED_PERCENTAGE',
        'parameters': {'rate': 0.001}
    }
    assert calculate_commission(scheme, 'taker', 0.5, 100) == 0.05
    
def test_calculate_commission_unknown():
    assert calculate_commission(None, 'taker', 0.5, 100) is None
    assert calculate_commission({'evidence_status': 'UNKNOWN'}, 'taker', 0.5, 100) is None
    
def test_calculate_commission_current_only():
    scheme = {'evidence_status': 'CURRENT_ONLY', 'formula_id': 'ZERO_FEE'}
    # Historical date provided
    assert calculate_commission(scheme, 'taker', 0.5, 100, transaction_date='2020-01-01') is None
    # No historical date
    assert calculate_commission(scheme, 'taker', 0.5, 100) == 0.0
    
def test_calculate_commission_maker_taker():
    scheme = {
        'evidence_status': 'HISTORICAL',
        'formula_id': 'MAKER_TAKER',
        'parameters': {'maker_rate': 0.0005, 'taker_rate': 0.002}
    }
    assert calculate_commission(scheme, 'maker', 0.5, 100) == 0.025
    assert calculate_commission(scheme, 'taker', 0.5, 100) == 0.1
    
def test_calculate_commission_nonlinear():
    scheme = {
        'evidence_status': 'HISTORICAL',
        'formula_id': 'POLYMARKET_NONLINEAR',
        'parameters': {'rate': 0.02}
    }
    # shares * price * (1-price) * rate
    # 100 * 0.4 * 0.6 * 0.02 = 0.48
    assert abs(calculate_commission(scheme, 'taker', 0.4, 100) - 0.48) < 1e-6
    
def test_calculate_commission_missing_params():
    scheme = {'evidence_status': 'HISTORICAL', 'formula_id': 'FIXED_PERCENTAGE', 'parameters': {}}
    assert calculate_commission(scheme, 'taker', 0.5, 100) is None
    
def test_calculate_commission_rounding():
    scheme = {
        'evidence_status': 'HISTORICAL',
        'formula_id': 'FIXED_PERCENTAGE',
        'parameters': {'rate': 0.001},
        'round_method': 'ROUND_DOWN',
        'round_decimals': 4
    }
    # 100 * 0.55555 * 0.001 = 0.055555
    # Rounded down to 4 decimals -> 0.0555
    assert calculate_commission(scheme, 'taker', 0.55555, 100) == 0.0555

def test_apply_fee_to_budget():
    # exclusive: 100 budget, price 0.5, rate 0.01 -> shares = 200, fee = 200*0.5*0.01 = 1, cost = 101
    cost, shares, fee = apply_fee_to_budget(100, 0.01, 0.5, mode='exclusive')
    assert cost == 101.0
    assert shares == 200.0
    assert fee == 1.0
    
    # inclusive: 100 budget, price 0.5, rate 0.01 -> shares = 100 / (0.5 * 1.01) = 198.0198...
    # fee = shares * 0.5 * 0.01 = 0.990099
    cost, shares, fee = apply_fee_to_budget(100, 0.01, 0.5, mode='inclusive')
    assert cost == 100.0
    assert abs(shares - 198.0198) < 1e-3
    assert abs(fee - 0.990099) < 1e-3
    
    with pytest.raises(ValueError):
        apply_fee_to_budget(100, 0.01, 0.5, mode='unknown')

def test_calculate_net_pnl():
    # Fully closed, no fee uncertainty
    assert calculate_net_pnl(100, 0, 50, 1, 0, 0, is_fully_closed=True) == 49.0
    
    # Not fully closed
    assert calculate_net_pnl(100, 0, 50, 1, 0, 0, is_fully_closed=False) is None
    
    # Uncertain fee
    assert calculate_net_pnl(100, 0, 50, 1, 0, 0, is_fully_closed=True, fee_is_uncertain=True) is None
