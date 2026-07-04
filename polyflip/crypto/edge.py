"""
Модуль вычисления edge для крипто-модели LightGBM.
Изолирован от Polymarket-логики.
"""

def compute_crypto_edge(
    p_up: float,
    threshold_up: float,
    threshold_down: float,
) -> tuple[float, str]:
    """
    Вычисляет математическое преимущество крипто-модели с учетом асимметричных порогов.

    Returns:
        (edge, direction) где direction: "UP" | "DOWN" | "NONE".
        Если p_up попадает в мертвую зону (threshold_down, threshold_up),
        возвращается edge=0.0 и direction="NONE".
    """
    if p_up >= threshold_up:
        return round(p_up - threshold_up, 4), "UP"
    if p_up <= threshold_down:
        return round(threshold_down - p_up, 4), "DOWN"
    return 0.0, "NONE"
