def compute_kelly_fraction(
    p_win: float,
    buy_price: float,
    max_fraction: float = 0.10,
) -> float:
    """
    Вычисляет Kelly fraction (долю капитала) для ставки на бинарном рынке.
    p_win  — вероятность выигрыша нашей ставки.
    buy_price — цена покупки токена (от 0 до 1).
    max_fraction — максимальный риск на одну сделку (10% по умолчанию).
    
    Возвращает:
        float: kelly_fraction (от 0.0 до max_fraction). Если edge <= 0, возвращает 0.0.
    """
    if buy_price <= 0.0 or buy_price >= 1.0:
        return 0.0
    
    # Kelly Formula: f = (bp - q) / b, where b is decimal odds - 1.
    # On Polymarket, decimal odds = 1 / buy_price. So b = (1 - buy_price) / buy_price.
    # Wait, simple way for binary markets: f = (p_win - buy_price) / (1 - buy_price)
    edge = p_win - buy_price
    if edge <= 0:
        return 0.0
        
    f = edge / (1.0 - buy_price)
    f = max(0.0, min(f, max_fraction))
    return round(f, 4)
