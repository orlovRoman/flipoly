

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
