# --- Торговые константы ---
FAVORITE_THRESHOLD = 0.55          # граница фаворит/аутсайдер
DEAD_ZONE_WIDTH = 0.10            # ширина мёртвой зоны (единый параметр для обоих режимов)
AUTO_DEAD_ZONE = True        # по умолчанию включён авто-режим
# AUTO_DEAD_ZONE_WIDTH удалена — движок теперь всегда читает DEAD_ZONE_WIDTH
DAILY_LOSS_LIMIT_USDC = -100.0    # стоп-лосс на день

# Режим ставки на аутсайдера (NO при флипе)
TRADE_ON_FAVORITE = True       # по умолчанию включен
TRADE_ON_FLIP = False          # по умолчанию выключен
FLIP_THRESHOLD = 0.60         # p_flip >= 0.60 → рассматриваем аутсайдера
OUTSIDER_MAX_PRICE = 0.45     # не покупать аутсайдера дороже (замена NO_MAX_PRICE)
NO_MIN_EDGE = 0.04            # минимальный edge для NO-ставки (ВНИМАНИЕ: если менять, обновите дефолты и учтите существующую БД, где может быть 0.05)

# --- Размер ставок ---
TRADE_BET_SIZE_USDC   = 5.0    # минимальная ставка
MAX_BET_SIZE_USDC     = 50.0   # максимальная ставка
LIQUIDITY_FRACTION    = 0.05   # не более 5% от volume_5min

# --- Границы цены входа ---
FAVORITE_MIN_PRICE    = 0.55   # не покупать фаворита дешевле
FAVORITE_MAX_PRICE    = 0.95   # не покупать фаворита дороже
FAVORITE_MIN_EDGE     = -0.01  # мин. edge для pure-favorite (мягче чем ML)

# --- ML пороги ---
NO_FLIP_THRESHOLD     = 0.35   # p_flip < этого → ML_TREND покупает фаворита

# --- Комиссии ---
POLYMARKET_FEE_RATE   = 0.002  # 0.2% от выплаты

# --- Математические sentinels ---
FLIP_MIDPOINT         = 0.5    # нейтральная вероятность
INVALID_EDGE_SENTINEL = -1.0   # возврат compute_edge при buy_price <= 0

# --- Edge ---
MIN_EDGE = 0.05                     # ROI-based: (win_prob/buy_price) - 1. 0.05 = 5% ROI
MAX_EDGE_SCALING      = 0.40   # верхняя граница масштабирования ставки
MAX_EDGE_FILTER       = 0.20   # выше этого edge считается аномальным (торговый фильтр)

# --- Режимы торговли ---
TRADING_MODE_ML = "ml"
TRADING_MODE_FAVORITE = "favorite"
DEFAULT_TRADING_MODE = TRADING_MODE_ML
FAVORITE_MODE_ENTRY_SEC = 180        # дефолт: 3 минуты до закрытия
FAVORITE_MODE_ENTRY_WINDOW_SEC = 60  # окно входа ±60 сек от точки входа

# --- Калибровка модели ---
MIN_PRECISION_FOR_THRESHOLD = 0.60  # минимальная precision при поиске порога
MAX_SUSPICIOUS_THRESHOLD = 0.95     # выше этого — подозрение на leakage
CV_N_SPLITS = 5                     # кол-во фолдов кросс-валидации
CV_RANDOM_STATE = 42                # seed для воспроизводимости

# --- Отбор фич LogisticRegression ---
LR_COEF_THRESHOLD = 0.005  # для 15m рынков порог ниже
LR_MIN_FEATURES   = 4
LR_MIN_AUC_FOR_DEPLOY = 0.53  # 15m рынки сложнее

# --- API / сеть ---
HTTP_TIMEOUT_SEC = 10.0             # таймаут CLOB/Gamma API
VOLUME_WINDOW_MIN = 5               # окно для volume_5min
TRADE_CHECK_LIMIT = 5               # кол-во последних записей для проверки дублей
LIVE_POLL_INTERVAL_SECONDS = 10     # интервал опроса коллектора по умолчанию

# ─── LightGBM / Crypto trainer ────────────────────────────────────────────
LGBM_N_ESTIMATORS      = 300
LGBM_LEARNING_RATE     = 0.05
LGBM_NUM_LEAVES        = 31
LGBM_MAX_DEPTH         = 5
LGBM_MIN_CHILD_SAMPLES = 30
LGBM_SUBSAMPLE         = 0.8
LGBM_COLSAMPLE_BYTREE  = 0.8
LGBM_REG_ALPHA         = 0.1
LGBM_REG_LAMBDA        = 1.0


# Backtester
BACKTEST_MIN_EDGE         = 0.04    # модель даёт сигнал только если prob - 0.5 >= MIN_EDGE
BACKTEST_COMMISSION       = 0.001   # 0.1% на сторону (Binance taker)
BACKTEST_TRAIN_RATIO      = 0.70    # 70% train / 30% test (walk-forward)
BACKTEST_SHARPE_ANNUALIZE = 252 * 96  # 15m свечей в году (96 свечей × 252 торг. дня)

# --- Крипто-торговля ---
TRADING_MODE_LIGHTGBM = "lightgbm"
TRADING_MODE_CRYPTO = TRADING_MODE_LIGHTGBM
TRADING_MODE_COMBINED = "combined"

COMBINED_MODE_SUPPORTED_ASSETS = frozenset({"BTC", "ETH", "DOGE", "XRP", "SOL"})
COMBINED_BINANCE_SYMBOLS: dict[str, str] = {
    "BTC":  "BTCUSDT",
    "ETH":  "ETHUSDT",
    "DOGE": "DOGEUSDT",
    "XRP":  "XRPUSDT",
    "SOL":  "SOLUSDT",
}

CRYPTO_MIN_EDGE = 0.05
USE_CRYPTO_CONFIRM = False
CRYPTO_STANDALONE = False
ASSET_TO_BINANCE_SYMBOL = {
    "BTC":  "BTCUSDT",
    "ETH":  "ETHUSDT",
    "DOGE": "DOGEUSDT",
    "XRP":  "XRPUSDT",
    "SOL":  "SOLUSDT",
}

# --- Price-Phase Split Boundaries ---
PRICE_PHASE_BOUNDARIES: dict[str, tuple[float, float]] = {
    "contested": (0.00, 0.10),  # mid_price 0.40–0.60: рынок не решён
    "leaning":   (0.10, 0.25),  # mid_price 0.25–0.75: есть склонение
    "decided":   (0.25, 0.50),  # mid_price < 0.25 или > 0.75: рынок решён
}

MIN_SAMPLES_FOR_PHASE_MODEL = 150  # порог для фазовой модели

def get_price_phase(mid_price: float) -> str:
    """Определяет фазу по mid_price. Единое место истины для trainer и engine."""
    dev = round(abs(mid_price - 0.5), 4)
    for phase, (lo, hi) in PRICE_PHASE_BOUNDARIES.items():
        if lo <= dev < hi:
            return phase
    return "decided"  # fallback: dev >= 0.50

# --- Combined Mode ---
COMBINED_NONE_BET_MULTIPLIER = 0.5

