"""
Accounting registry and PnL logic.
"""
from typing import Optional

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
    Calculate PnL. For a fully closed position:
    net_pnl = sale_proceeds + settlement_proceeds - purchase_cash - platform_fees - attributable_network_costs + confirmed_rebates
    If position is not fully closed, returns None.
    If fee is uncertain, returns None.
    """
    if not is_fully_closed:
        return None
        
    if fee_is_uncertain:
        return None
        
    return sale_proceeds + settlement_proceeds - purchase_cash - platform_fees - attributable_network_costs + confirmed_rebates

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
