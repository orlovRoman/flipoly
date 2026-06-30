# --- Торговые константы ---
FAVORITE_THRESHOLD = 0.55          # граница фаворит/аутсайдер
DEAD_ZONE_WIDTH = 0.15            # ширина мёртвой зоны вокруг flip-порога
AUTO_DEAD_ZONE = True        # по умолчанию включён
AUTO_DEAD_ZONE_WIDTH = 0.10 # ширина нейтральной полосы (10 пп)
DAILY_LOSS_LIMIT_USDC = -100.0    # стоп-лосс на день

# Режим ставки на аутсайдера (NO при флипе)
TRADE_ON_FLIP = False          # по умолчанию выключен
FLIP_THRESHOLD = 0.70         # p_flip >= 0.70 → рассматриваем NO
NO_MAX_PRICE = 0.60           # не покупать NO дороже 0.60
NO_MIN_EDGE = 0.04            # минимальный edge для NO-ставки

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

# --- API / сеть ---
HTTP_TIMEOUT_SEC = 10.0             # таймаут CLOB/Gamma API
VOLUME_WINDOW_MIN = 5               # окно для volume_5min
TRADE_CHECK_LIMIT = 5               # кол-во последних записей для проверки дублей
LIVE_POLL_INTERVAL_SECONDS = 10     # интервал опроса коллектора по умолчанию
MIN_EDGE = 0.05                     # минимальный edge (разница вероятностей) по умолчанию
MAX_EDGE = 0.10                     # максимальный edge по умолчанию

