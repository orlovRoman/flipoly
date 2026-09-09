import math
from typing import Dict, Any, Optional

def calculate_commission(
    scheme: Optional[Dict[str, Any]], 
    role: str, 
    price: float, 
    shares: float,
    transaction_date: Optional[str] = None
) -> Optional[float]:
    """
    Calculate commission based on the applicable scheme.
    If scheme is None, returns None (uncertainty).
    If zero commission, returns 0.0.
    """
    if scheme is None or scheme.get('evidence_status') == 'UNKNOWN':
        return None
        
    if scheme.get('evidence_status') == 'CURRENT_ONLY' and transaction_date:
        # If the scheme is only for current trades, but we have a historical date, we can't apply it to the past
        return None
        
    formula_id = scheme.get('formula_id')
    params = scheme.get('parameters', {})
    
    if formula_id == 'FIXED_PERCENTAGE':
        if 'rate' not in params:
            return None
        rate = params.get('rate')
        fee = price * shares * rate
    elif formula_id == 'MAKER_TAKER':
        if role == 'taker' and 'taker_rate' not in params:
            return None
        if role == 'maker' and 'maker_rate' not in params:
            return None
        rate = params.get('taker_rate') if role == 'taker' else params.get('maker_rate')
        fee = price * shares * rate
    elif formula_id == 'POLYMARKET_NONLINEAR':
        if 'rate' not in params:
            return None
        rate = params.get('rate')
        fee = shares * price * (1.0 - price) * rate
    elif formula_id == 'ZERO_FEE':
        fee = 0.0
    else:
        return None
        
    rounding = scheme.get('rounding_rule')
    if rounding == 'ROUND_DOWN':
        charged_asset = scheme.get('charged_asset', 'USD')
        decimals = 6 if charged_asset == 'USDC' else 2
        multiplier = 10 ** decimals
        fee = math.floor(fee * multiplier) / multiplier
            
    return fee
