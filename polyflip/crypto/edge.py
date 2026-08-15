"""
Модуль вычисления силы сигнала для крипто-модели LightGBM и экономического преимущества.
"""

def compute_crypto_signal_strength(
    p_up: float,
    threshold_up: float,
    threshold_down: float,
) -> tuple[float, str]:
    """
    Вычисляет силу крипто-сигнала с учетом асимметричных порогов.

    Returns:
        (signal_strength, direction) где direction: "UP" | "DOWN" | "NONE".
        Если p_up попадает в мертвую зону (threshold_down, threshold_up),
        возвращается signal_strength=0.0 и direction="NONE".
    """
    if not 0.0 <= threshold_down < threshold_up <= 1.0:
        return 0.0, "NONE"
    # ``threshold_down`` is a lower bound on p_up, not a threshold on p_down.
    # Keeping both thresholds in the same coordinate system makes the dead
    # zone explicit and prevents overlapping UP/DOWN ranges.
    if p_up >= threshold_up:
        return round(p_up - threshold_up, 4), "UP"
    if p_up <= threshold_down:
        return round(threshold_down - p_up, 4), "DOWN"
    return 0.0, "NONE"

def compute_economic_edge(
    p_win: float,
    executable_ask: float,
    fee_rate: float,
    slippage_rate: float,
) -> float:
    """
    Рассчитывает реальное экономическое преимущество (edge) сделки.
    """
    if executable_ask <= 0.0:
        return 0.0
    effective_cost = executable_ask * (1 + slippage_rate)
    gross_edge = p_win / effective_cost - 1
    return gross_edge - fee_rate
