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
        
    evidence_status = scheme.get('evidence_status')
    valid_from = scheme.get('valid_from')
    valid_to = scheme.get('valid_to')
    
    if evidence_status == 'CURRENT_ONLY':
        return None
        
    if transaction_date:
        tx_date_str = str(transaction_date)[:10]
        if valid_from and tx_date_str < str(valid_from)[:10]:
            return None
        if valid_to and tx_date_str > str(valid_to)[:10]:
            return None
    else:
        if valid_from or valid_to:
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
        
    round_method = scheme.get('round_method') or scheme.get('rounding_rule')
    if round_method == 'ROUND_DOWN':
        decimals = scheme.get('round_decimals')
        if decimals is not None:
            multiplier = 10 ** decimals
            fee = math.floor(fee * multiplier) / multiplier
            
    return fee
