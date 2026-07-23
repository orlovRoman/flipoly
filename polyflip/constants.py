"""
polyflip/constants.py

ОПИСЫВАЕТ СТРУКТУРУ АЛГОРИТМА И СТАТИЧЕСКИЕ МАППИНГИ.
Здесь содержатся только структурные константы, маркеры и алгоритмические параметры кода.

Все динамические / настраиваемые параметры хранятся в RuntimeSettings БД
и читаются исключительно через polyflip.services.settings_service.
"""

# --- Режимы торговли (строковые идентификаторы) ---
TRADING_MODE_ML       = "ml"
TRADING_MODE_FAVORITE = "favorite"
TRADING_MODE_LIGHTGBM = "lightgbm"
TRADING_MODE_CRYPTO   = TRADING_MODE_LIGHTGBM
TRADING_MODE_COMBINED = "combined"
DEFAULT_TRADING_MODE  = TRADING_MODE_ML

# --- Маппинги активов и поддерживаемые рынки ---
ASSET_TO_BINANCE_SYMBOL = {
    "BTC":  "BTCUSDT",
    "ETH":  "ETHUSDT",
    "DOGE": "DOGEUSDT",
    "XRP":  "XRPUSDT",
    "SOL":  "SOLUSDT",
}
COMBINED_BINANCE_SYMBOLS = ASSET_TO_BINANCE_SYMBOL
COMBINED_MODE_SUPPORTED_ASSETS = frozenset({"BTC", "ETH", "DOGE", "XRP", "SOL"})

# --- Price-Phase Split Boundaries ---
PRICE_PHASE_BOUNDARIES: dict[str, tuple[float, float]] = {
    "contested": (0.00, 0.10),  # mid_price 0.40–0.60: рынок не решён
    "leaning":   (0.10, 0.25),  # mid_price 0.25–0.75: есть склонение
    "decided":   (0.25, 0.50),  # mid_price < 0.25 или > 0.75: рынок решён
}


def get_price_phase(mid_price: float) -> str:
    """Определяет фазу по mid_price. Единое место истины для trainer и engine."""
    dev = round(abs(mid_price - 0.5), 4)
    for phase, (lo, hi) in PRICE_PHASE_BOUNDARIES.items():
        if lo <= dev < hi:
            return phase
    return "decided"  # fallback: dev >= 0.50


# --- Математические sentinels ---
FLIP_MIDPOINT         = 0.5    # нейтральная вероятность
INVALID_EDGE_SENTINEL = -1.0   # возврат compute_edge при buy_price <= 0

# Порог предупреждения о плохой калибровке модели (ECE)
# Если ECE > этого значения — вероятности модели ненадёжны
ECE_WARN_THRESHOLD: float = 0.07

# --- Кросс-валидация (алгоритмические параметры) ---
CV_N_SPLITS     = 5            # кол-во фолдов кросс-валидации
CV_RANDOM_STATE = 42           # seed для воспроизводимости

# --- Таймеры / интервалы входа ---
FAVORITE_MODE_ENTRY_SEC        = 180   # дефолт: 3 минуты до закрытия
FAVORITE_MODE_ENTRY_WINDOW_SEC = 60    # окно входа ±60 сек от точки входа

# --- API / сеть / инфраструктура ---
HTTP_TIMEOUT_SEC           = 10.0      # таймаут CLOB/Gamma API
VOLUME_WINDOW_MIN          = 5         # окно для volume_5min
TRADE_CHECK_LIMIT          = 5         # кол-во последних записей для проверки дублей
LIVE_POLL_INTERVAL_SECONDS = 10        # интервал опроса коллектора по умолчанию
