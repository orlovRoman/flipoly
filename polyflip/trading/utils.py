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

def compute_dead_zone(
    flip_threshold: float,
    dead_zone_width: float,
    auto_mode: bool = True,
) -> tuple[float, float]:
    """
    Возвращает (no_flip_upper, flip_lower) — границы мёртвой зоны.
    
    В авторежиме: зона симметрична вокруг середины между flip_threshold 
    и (1 - flip_threshold).
    
    Пример: flip_threshold=0.70, width=0.10
        lower = 0.70 - 0.05 = 0.65
        upper = 0.70 + 0.05 = 0.75
        → мёртвая зона: [0.65, 0.75]
        → BUY YES если p_flip < 0.65
        → BUY NO если p_flip > 0.75
    """
    if auto_mode:
        half = dead_zone_width / 2.0
        lower = round(flip_threshold - half, 4)
        upper = round(flip_threshold + half, 4)
    else:
        # Ручной режим: upper = flip_threshold, lower = flip_threshold - width
        lower = round(flip_threshold - dead_zone_width, 4)
        upper = flip_threshold
    return lower, upper
