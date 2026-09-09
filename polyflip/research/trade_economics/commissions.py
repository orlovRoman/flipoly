"""
Pure functions and Decimal-based precision accounting for commission calculation.
"""
import math
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
from typing import Dict, Any, Optional, Union
import datetime

def _parse_date(d: Any) -> Optional[datetime.date]:
    if d is None:
        return None
    if isinstance(d, float) and math.isnan(d):
        return None
    if isinstance(d, str):
        if d.lower() == 'nan' or d.strip() == '':
            return None
        try:
            s = d[:10]
            if len(s) == 10 and s[2] == '.' and s[5] == '.':
                return datetime.datetime.strptime(s, "%d.%m.%Y").date()
            if len(s) == 10 and s[2] == '-' and s[5] == '-':
                return datetime.datetime.strptime(s, "%d-%m-%Y").date()
            return datetime.datetime.fromisoformat(s).date()
        except ValueError:
            pass
    if hasattr(d, 'date') and callable(d.date):
        return d.date()
    if isinstance(d, datetime.date):
        return d
    return None

def calculate_commission_decimal(
    scheme: Optional[Dict[str, Any]], 
    role: str, 
    price: Union[float, Decimal, str], 
    shares: Union[float, Decimal, str],
    transaction_date: Optional[Any] = None
) -> Optional[Decimal]:
    """
    Calculate commission in Decimal precision based on the applicable scheme.
    If scheme is None, returns None (uncertainty).
    If zero commission, returns Decimal('0.0').
    """
    if scheme is None or scheme.get('evidence_status') == 'UNKNOWN':
        return None
        
    evidence_status = scheme.get('evidence_status')
    valid_from = _parse_date(scheme.get('valid_from'))
    valid_to = _parse_date(scheme.get('valid_to'))
    
    if evidence_status == 'CURRENT_ONLY':
        return None
        
    tx_date = _parse_date(transaction_date)
    if tx_date is not None:
        if valid_from and tx_date < valid_from:
            return None
        if valid_to and tx_date > valid_to:
            return None
    else:
        if valid_from or valid_to:
            return None

    formula_id = scheme.get('formula_id')
    params = scheme.get('parameters', {})
    
    d_price = Decimal(str(price))
    d_shares = Decimal(str(shares))

    if formula_id == 'FIXED_PERCENTAGE':
        if 'rate' not in params:
            return None
        rate = Decimal(str(params.get('rate')))
        fee = d_price * d_shares * rate
    elif formula_id == 'MAKER_TAKER':
        if role == 'taker' and 'taker_rate' not in params:
            return None
        if role == 'maker' and 'maker_rate' not in params:
            return None
        rate = Decimal(str(params.get('taker_rate') if role == 'taker' else params.get('maker_rate')))
        fee = d_price * d_shares * rate
    elif formula_id == 'POLYMARKET_NONLINEAR':
        if 'rate' not in params:
            return None
        rate = Decimal(str(params.get('rate')))
        fee = d_shares * d_price * (Decimal('1.0') - d_price) * rate
    elif formula_id == 'ZERO_FEE':
        fee = Decimal('0.0')
    else:
        return None
        
    round_method = scheme.get('round_method') or scheme.get('rounding_rule')
    decimals = scheme.get('round_decimals')
    
    if decimals is not None and round_method:
        quant = Decimal('10') ** -int(decimals)
        if round_method == 'ROUND_DOWN':
            fee = fee.quantize(quant, rounding=ROUND_DOWN)
        elif round_method in ('ROUND_HALF_UP', 'ROUND_NEAREST'):
            fee = fee.quantize(quant, rounding=ROUND_HALF_UP)
            
    return fee

def calculate_commission(
    scheme: Optional[Dict[str, Any]], 
    role: str, 
    price: float, 
    shares: float,
    transaction_date: Optional[Any] = None
) -> Optional[float]:
    """
    Float-compatible wrapper over calculate_commission_decimal.
    """
    d_fee = calculate_commission_decimal(scheme, role, price, shares, transaction_date)
    return float(d_fee) if d_fee is not None else None
