def compute_kelly_multiplier(
    p_win: float,
    buy_price: float,
    max_fraction: float = 0.10,
) -> tuple[float, float]:
    """
    Вычисляет Kelly fraction и Kelly multiplier для размера ставки на бинарном рынке.
    p_win  — вероятность выигрыша нашей ставки.
    buy_price — цена покупки токена (от 0 до 1).
    max_fraction — максимальный риск на одну сделку (10% по умолчанию).
    
    Возвращает:
        tuple[float, float]: (kelly_fraction, kelly_multiplier)
        kelly_fraction: от 0.0 до max_fraction
        kelly_multiplier: от 0.5 до 2.0
    """
    if buy_price <= 0.0 or buy_price >= 1.0:
        return 0.0, 0.5
    b = (1.0 - buy_price) / buy_price
    f = (p_win * (b + 1.0) - 1.0) / b
    f = max(0.0, min(f, max_fraction))
    multiplier = 0.5 + (f / max_fraction) * 1.5
    return round(f, 4), round(multiplier, 2)
