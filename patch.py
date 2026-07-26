import re

content = open('polyflip/trading/decision_logic.py', 'r', encoding='utf-8').read()

# Add decision_details to TradeDecision
content = content.replace(
    'p_win_raw: Optional[float] = None\n    probability_adjustment: Optional[str] = None',
    'p_win_raw: Optional[float] = None\n    probability_adjustment: Optional[str] = None\n    decision_details: Optional[dict] = None'
)

# Fix decide_ml_trend
old_trend = '''    no_flip_thresh = float(config.get("NO_FLIP_THRESHOLD", 0.35))

    p_flip_calibrated = apply_ece_correction(p_flip, ece)
    p_win = 1.0 - p_flip_calibrated

    # 1. Проверяем dead zone'''
new_trend = '''    no_flip_thresh = float(config.get("NO_FLIP_THRESHOLD", 0.35))

    p_flip_calibrated = apply_ece_correction(p_flip, ece)
    p_flip_effective = max(p_flip, p_flip_calibrated)
    p_win = 1.0 - p_flip_effective

    # 1. Проверяем dead zone'''
content = content.replace(old_trend, new_trend)

content = content.replace(
    '    if p_flip_calibrated >= no_flip_thresh:\n        return TradeDecision("SKIP", 0, 0,\n            f"p_flip_calibrated={p_flip_calibrated:.3f} >= threshold={no_flip_thresh:.3f}", "SKIP",',
    '    if p_flip_effective >= no_flip_thresh:\n        return TradeDecision("SKIP", 0, 0,\n            f"p_flip_effective={p_flip_effective:.3f} >= threshold={no_flip_thresh:.3f}", "SKIP",'
)

old_trend_return = '''    return TradeDecision(
        action, buy_price, bet,
        f"ML_TREND p_flip={p_flip:.3f} < {no_flip_thresh:.3f}",
        "ML_TREND",
        p_flip=p_flip, edge=edge,
        p_win_effective=p_win, p_win_raw=p_win
    )'''
new_trend_return = '''    decision_details = {
        "p_flip_raw": round(p_flip, 4),
        "p_flip_effective": round(p_flip_effective, 4),
        "ece_used": round(ece, 4),
        "threshold_upper_applied": round(no_flip_thresh, 4),
        "bet_size_before_multiplier": round(bet, 4),
    }

    return TradeDecision(
        action, buy_price, bet,
        f"ML_TREND p_flip_effective={p_flip_effective:.3f} < {no_flip_thresh:.3f}",
        "ML_TREND",
        p_flip=p_flip, edge=edge,
        p_win_effective=p_win, p_win_raw=1.0 - p_flip,
        decision_details=decision_details
    )'''
content = content.replace(old_trend_return, new_trend_return)


# Fix decide_outsider
content = content.replace(
    '    p_flip_calibrated = apply_ece_correction(p_flip, ece)\n\n    # 1. Сначала проверяем dead zone',
    '    p_flip_calibrated = apply_ece_correction(p_flip, ece)\n    p_flip_effective = min(p_flip, p_flip_calibrated)\n\n    # 1. Сначала проверяем dead zone'
)

content = content.replace(
    '    p_win_outsider = p_flip_calibrated * outsider_pwin_discount',
    '    p_win_outsider = p_flip_effective * outsider_pwin_discount'
)

content = content.replace(
    '        p_flip_calibrated=round(p_flip_calibrated, 4),',
    '        p_flip_effective=round(p_flip_effective, 4),'
)

content = content.replace(
    '    if p_flip_calibrated < flip_thresh:\n        return TradeDecision("SKIP", 0, 0,\n            f"p_flip_calibrated={p_flip_calibrated:.3f} < threshold={flip_thresh:.3f}", "SKIP",',
    '    if p_flip_effective < flip_thresh:\n        return TradeDecision("SKIP", 0, 0,\n            f"p_flip_effective={p_flip_effective:.3f} < threshold={flip_thresh:.3f}", "SKIP",'
)

old_outsider_return = '''    return TradeDecision(
        outsider_action, outsider_ask, bet,
        f"outsider {outsider_action.split('_')[1]}, p_flip={p_flip:.3f}", "OUTSIDER",
        p_flip=p_flip, edge=edge,
        p_win_effective=p_win_outsider, p_win_raw=p_win_outsider
    )'''

new_outsider_return = '''    decision_details = {
        "p_flip_raw": round(p_flip, 4),
        "p_flip_effective": round(p_flip_effective, 4),
        "ece_used": round(ece, 4),
        "threshold_upper_applied": round(flip_thresh, 4),
        "bet_size_before_multiplier": round(bet, 4),
    }

    return TradeDecision(
        outsider_action, outsider_ask, bet,
        f"OUTSIDER p_flip_effective={p_flip_effective:.3f} >= {flip_thresh:.3f}",
        "TRADE_ON_FLIP",
        p_flip=p_flip, edge=edge,
        p_win_effective=p_win_outsider, p_win_raw=p_flip * outsider_pwin_discount,
        decision_details=decision_details
    )'''
content = content.replace(old_outsider_return, new_outsider_return)

with open('polyflip/trading/decision_logic.py', 'w', encoding='utf-8') as f:
    f.write(content)
