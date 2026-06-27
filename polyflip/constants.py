# --- Торговые константы ---
FAVORITE_THRESHOLD = 0.5          # граница фаворит/аутсайдер
KELLY_MAX_FRACTION = 0.10         # максимальная Kelly-доля от капитала
DEAD_ZONE_WIDTH = 0.15            # ширина мёртвой зоны вокруг flip-порога
DAILY_LOSS_LIMIT_USDC = -100.0    # стоп-лосс на день

# --- Калибровка модели ---
MIN_PRECISION_FOR_THRESHOLD = 0.60  # минимальная precision при поиске порога
MAX_SUSPICIOUS_THRESHOLD = 0.95     # выше этого — подозрение на leakage
CV_N_SPLITS = 5                     # кол-во фолдов кросс-валидации
CV_RANDOM_STATE = 42                # seed для воспроизводимости

# --- API / сеть ---
HTTP_TIMEOUT_SEC = 10.0             # таймаут CLOB/Gamma API
VOLUME_WINDOW_MIN = 5               # окно для volume_5min
TRADE_CHECK_LIMIT = 5               # кол-во последних записей для проверки дублей
