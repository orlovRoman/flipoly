"""
Accounting registry and PnL logic with multi-fill VWAP and settlement support.
"""
from typing import Optional, List, Dict, Any
from decimal import Decimal

def calculate_net_pnl(
    sale_proceeds: float,
    settlement_proceeds: float,
    purchase_cash: float,
    platform_fees: float,
    attributable_network_costs: float,
    confirmed_rebates: float,
    is_fully_closed: bool,
    fee_is_uncertain: bool = False
) -> Optional[float]:
    """
    Calculate PnL for closed position.
    net_pnl = sale_proceeds + settlement_proceeds - purchase_cash - platform_fees - attributable_network_costs + confirmed_rebates
    If position is not fully closed, returns None.
    If fee is uncertain, returns None.
    """
    if not is_fully_closed:
        return None
        
    if fee_is_uncertain:
        return None
        
    return sale_proceeds + settlement_proceeds - purchase_cash - platform_fees - attributable_network_costs + confirmed_rebates

def calculate_fill_position(fills: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Aggregates multiple execution fills into a single position:
    filled_shares = sum(shares)
    purchase_cash = sum(price * shares)
    VWAP = purchase_cash / filled_shares
    total_fee_usdc = sum(fee_usdc)
    """
    if not fills:
        return {
            "filled_shares": 0.0,
            "purchase_cash": 0.0,
            "vwap": 0.0,
            "total_fee_usdc": 0.0,
            "fill_count": 0
        }
        
    d_shares = Decimal('0.0')
    d_cash = Decimal('0.0')
    d_fee = Decimal('0.0')
    
    for f in fills:
        sh = Decimal(str(f.get("shares", 0.0)))
        pr = Decimal(str(f.get("price", 0.0)))
        fe = Decimal(str(f.get("fee_usdc", 0.0) or 0.0))
        d_shares += sh
        d_cash += (pr * sh)
        d_fee += fe
        
    f_shares = float(d_shares)
    f_cash = float(d_cash)
    f_vwap = float(d_cash / d_shares) if d_shares > 0 else 0.0
    f_fee = float(d_fee)
    
    return {
        "filled_shares": f_shares,
        "purchase_cash": f_cash,
        "vwap": f_vwap,
        "total_fee_usdc": f_fee,
        "fill_count": len(fills)
    }

def apply_fee_to_budget(
    total_budget: float,
    commission_rate: float,
    price: float,
    mode: str = 'exclusive'
):
    """
    mode = 'exclusive' means "$1 on purchase PLUS commission" (cost = shares*price + commission)
    mode = 'inclusive' means "$1 INCLUDING commission" (cost = total_budget, shares adjusted)
    """
    if mode == 'exclusive':
        shares = total_budget / price
        commission = shares * price * commission_rate
        actual_cost = total_budget + commission
        return actual_cost, shares, commission
    elif mode == 'inclusive':
        shares = total_budget / (price * (1 + commission_rate))
        commission = shares * price * commission_rate
        actual_cost = total_budget
        return actual_cost, shares, commission
    raise ValueError(f"Unknown mode: {mode}")
