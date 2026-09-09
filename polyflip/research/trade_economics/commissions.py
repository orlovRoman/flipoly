"""
Pure functions for commission calculation.
"""
from typing import Dict, Any, Optional

def calculate_commission(
    scheme: Optional[Dict[str, Any]], 
    role: str, 
    price: float, 
    shares: float
) -> Optional[float]:
    """
    Calculate commission based on the applicable scheme.
    If scheme is None, returns None (uncertainty).
    If zero commission, returns 0.0.
    """
    if scheme is None or scheme.get('evidence_status') == 'UNKNOWN':
        return None
        
    formula_id = scheme.get('formula_id')
    params = scheme.get('parameters', {})
    
    if formula_id == 'FIXED_PERCENTAGE':
        rate = params.get('rate', 0.0)
        return price * shares * rate
    elif formula_id == 'MAKER_TAKER':
        rate = params.get('taker_rate', 0.0) if role == 'taker' else params.get('maker_rate', 0.0)
        return price * shares * rate
    elif formula_id == 'ZERO_FEE':
        return 0.0
    else:
        return None
