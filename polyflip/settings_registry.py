"""
polyflip/settings_registry.py

Единый реестр runtime-настроек.
Каждый параметр описан ровно один раз — с явно заданным дефолтным значением.

Использование:
  - settings_service.py:      _DEFAULTS = registry_defaults()
  - init_runtime_settings.py:  DEFAULTS = registry_defaults()
  - api/settings.py:           valid_keys = list(editable_keys())
  - trading/engine.py:         settings_keys = list(registry_keys())

Это гарантирует: если добавили ключ в реестр — он автоматически появляется
в сидере, в API, в сервисе настроек и в движке.
"""
from dataclasses import dataclass

from polyflip.constants import (
    DEFAULT_TRADING_MODE,
    LIVE_POLL_INTERVAL_SECONDS,
)


@dataclass(frozen=True)
class SettingDef:
    """Описание одного runtime-параметра."""
    key: str
    default: str
    editable: bool = True   # False — только для внутренних ключей (напр. TRADING_ENABLED)
    description: str = ""


# ── Реестр ───────────────────────────────────────────────────────────────────
REGISTRY: list[SettingDef] = [
    # --- Мёртвая зона ---
    SettingDef("DEAD_ZONE_WIDTH", "0.10",
               description="Ширина мёртвой зоны вокруг flip-порога (единый параметр)"),

    # --- Финансы / потери ---
    SettingDef("DAILY_LOSS_LIMIT_USDC", "-100.0",
               description="Дневной стоп-лосс в USDC (отрицательное число)"),
    SettingDef("MAX_OPEN_POSITIONS", "5",
               description="Максимальное количество одновременно открытых позиций"),
    SettingDef("MAX_TOTAL_EXPOSURE_USDC", "50.0",
               description="Общий лимит риска по всем позициям и заявкам"),
    SettingDef("MAX_SINGLE_ORDER_USDC", "1.0",
               description="Максимальная сумма одной сделки"),
    SettingDef("CONFIRM_THRESHOLD_USDC", "5.0",
               description="Порог ручного подтверждения крупной ставки"),
    SettingDef("EXECUTION_COOLDOWN_SEC", "10",
               description="Задержка перед новой попыткой при ошибке (сек)"),
    SettingDef("TRADE_EXECUTION_TIME_SEC", "30",
               description="Максимальное время на исполнение сделки"),
    SettingDef("INITIAL_CAPITAL", "1000.0",
               description="Начальный капитал для расчёта финрезультата"),

    # --- Стоп-лосс позиции ---
    SettingDef("STOP_LOSS_ENABLED", "false",
               description="Включить стоп-лосс открытых позиций"),
    SettingDef("STOP_LOSS_PCT_FAVORITE", "40.0",
               description="Стоп-лосс % для ставок на фаворита (1–99)"),
    SettingDef("STOP_LOSS_PCT_OUTSIDER", "60.0",
               description="Стоп-лосс % для ставок на аутсайдера (1–99)"),
    SettingDef("STOP_LOSS_CHECK_SEC", "30",
               description="Интервал проверки открытых позиций воркером стоп-лосса (сек)"),

    # --- Тейк-профит позиции ---
    SettingDef("TAKE_PROFIT_ENABLED", "false",
               description="Включить тейк-профит открытых позиций"),
    SettingDef("TAKE_PROFIT_MULTIPLIER", "2.0",
               description="Мультипликатор тейк-профита от цены входа"),
    SettingDef("TAKE_PROFIT_CHECK_INTERVAL_SEC", "30",
               description="Интервал проверки открытых позиций воркером тейк-профита (сек)"),

    # --- Режим торговли ---
    SettingDef("TRADING_MODE", DEFAULT_TRADING_MODE,
               description="Режим торговли: combined (единственный активный режим)"),
    SettingDef("TRADING_ENABLED", "false", editable=False,
               description="Управляется только через UI toggle, не через general settings API"),
    SettingDef("LIVE_TRADING_ENABLED", "false", editable=False,
               description="Управляется только через специальный API (kill-switch)"),

    # --- Таймеры / опрос ---
    SettingDef("LIVE_POLL_INTERVAL_SECONDS", str(LIVE_POLL_INTERVAL_SECONDS)),
    SettingDef("FAVOR_MIN_TIME_LEFT_SEC", "60",
               description="Мин. время до конца для фаворита (сек)"),
    SettingDef("FAVOR_MAX_TIME_LEFT_SEC", "600",
               description="Макс. время до конца для фаворита (сек)"),
    SettingDef("OUTS_MIN_TIME_LEFT_SEC", "30",
               description="Мин. время до конца для аутсайдера (сек)"),
    SettingDef("OUTS_MAX_TIME_LEFT_SEC", "300",
               description="Макс. время до конца для аутсайдера (сек)"),
    SettingDef("TRADE_BET_SIZE_USDC", "10.0"),

    # --- Сайзинг ---
    SettingDef("MAX_BET_SIZE_USDC", "50.0",
               description="Максимальная ставка (USDC)"),
    SettingDef("BET_SIZING_MODE", "scaled",
               description="Режим расчёта размера: scaled | fixed"),
    SettingDef("LIQUIDITY_FRACTION", "0.05",
               description="Макс. доля от volume_5min на одну ставку"),

    # --- Edge ---
    SettingDef("OUTS_MIN_EDGE", "0.04",
               description="Мин. edge для ставок против толпы (в долях)"),
    SettingDef("MAX_BET_EDGE", "0.40",
               description="Edge при котором достигается макс. размер ставки при scaled-режиме"),

    # --- Фаворит ---
    SettingDef("FAVORITE_THRESHOLD", "0.55",
               description="Граница фаворит/аутсайдер по mid_price"),
    SettingDef("FAVORITE_MIN_EDGE", "-0.01",
               description="Мин. edge для ставок на фаворита в Combined"),
    SettingDef("FAVORITE_MIN_PRICE", "0.55",
               description="Мин. цена для входа в фаворита"),
    SettingDef("FAVORITE_MAX_PRICE", "0.95",
               description="Макс. цена для входа в фаворита"),

    # --- Аутсайдер / NO ---
    SettingDef("OUTSIDER_MAX_PRICE", "0.45",
               description="Макс. цена покупки аутсайдера"),
    SettingDef("TRADE_ON_FAVORITE", "true",
               description="Если включено, бот делает ставки по тренду на фаворита в Combined."),
    SettingDef("TRADE_ON_FLIP", "false",
               description="Торговать на флип (стратегия аутсайдера)"),
    SettingDef("FLIP_THRESHOLD", "0.60",
               description="Порог p_flip для входа в аутсайдера"),
    SettingDef("OUTSIDER_PWIN_DISCOUNT", "0.65",
               description="Хранится как коэффициент 0.0-1.0; в UI отображается как проценты (0-100). Дисконт вероятности выигрыша аутсайдера."),
    SettingDef("MAX_SPREAD_PCT", "0.08",
               description="Макс. спред как доля от mid_price. Шире — сделка не создаётся"),
    SettingDef("TRADE_FLIP_THRESHOLD", "0.85",
               description="Глобальный порог ПРОТИВ ТОЛПЫ (если нет индивидуального)"),

    SettingDef("COMBINED_COST_BUFFER", "0.020",
               description="Буфер транзакционных издержек в Combined-режиме (в USD). Вычитается из gross_edge при расчёте net_edge."),

    SettingDef("MAX_PRICE_DRIFT", "0.10",
               description="Макс. дрейф цены от момента сигнала до исполнения"),

    SettingDef("MIN_DIRECTION_PROB", "0.505",
               description="Мин. уверенность LGBM в направлении (UP/DOWN) для входа в Combined (пол как fallback)"),
    SettingDef("MIN_WIN_PROB", "0.51",
               description="Мин. вероятность победы кандидата (LGBM + LogReg) для входа"),
    SettingDef("COMBINED_DIR_DISCOUNT_WEIGHT", "0.0",
               description="Вес дисконта к p_candidate_win за неуверенность LightGBM (0.0 = выкл, 0.3 = до -30% при dir_prob=min_dir_prob)"),
    SettingDef("COMBINED_DIR_STRONG_THRESHOLD", "0.65",
               description="Порог dir_prob, выше которого сигнал LightGBM считается сильным и дисконт равен 0 (0.50-1.0)"),
    SettingDef("COMBINED_REQUIRE_CONSENSUS", "true",
               description="Обе модели (LightGBM и LogReg) должны проголосовать за одно направление. Иначе - SKIP."),
    SettingDef("COMBINED_FALLBACK_TO_LOGREG_ON_NONE", "true",
               description="Если LightGBM выдает NONE, то используется голос LogReg. Если false, то NONE ведет к SKIP."),
    SettingDef("COMBINED_LOGREG_ABSTAIN_BAND", "0.05",
               description="Ширина коридора нерешительности LogReg вокруг 0.50 (|p_flip - 0.50| < band -> ABSTAIN)"),
    SettingDef(
        key="INVERT_LGBM_SIGNAL",
        default="false",
        description="Инвертировать сигнал LightGBM: если true, p_up↔p_down меняются местами. "
                    "Используется, когда LGBM является контриндикатором (WinRate < 40%).",
    ),

    # --- Обучение LogReg / Phase models ---
    SettingDef("MIN_SAMPLES_FOR_PHASE_MODEL", "150",
               description="Мин. количество сэмплов для обучения фазовой модели (contested/leaning/decided)"),
    SettingDef("LR_MIN_AUC_FOR_DEPLOY", "0.53",
               description="Мин. AUC LogReg-модели для деплоя. Ниже — модель не сохраняется"),
    SettingDef("LR_COEF_THRESHOLD", "0.005",
               description="Порог коэффициента LogReg для отсева слабых фич"),
    SettingDef("LR_MIN_FEATURES", "4",
               description="Мин. кол-во фич после отсева слабых"),
    SettingDef("LR_TRAIN_MAX_TIME_LEFT_MIN", "15.0",
               description="Верхняя граница time_left (мин) для обучающей выборки LogReg. "
                           "Для 15-минутных рынков = 15.0. Отсекает данные с нерелевантным горизонтом."),
    SettingDef("LR_TRAIN_MIN_TIME_LEFT_MIN", "0.5",
               description="Нижняя граница time_left (мин) для обучающей выборки. "
                           "Исключает снапшоты последних 30 сек (шум исполнения)."),
    SettingDef("LR_SAMPLE_WEIGHT_MODE", "time_decay",
               description="Режим взвешивания сэмплов при обучении: "
                           "'uniform' — без весов, "
                           "'time_decay' — вес = 1/(time_left+1), "
                           "'exp_decay' — вес = exp(-time_left/tau)."),
    SettingDef("LR_SAMPLE_WEIGHT_TAU", "5.0",
               description="Параметр tau для exp_decay взвешивания (в минутах). "
                           "Сэмплы старше tau минут получают вес < 0.37."),

    # --- Цена входа ---
    SettingDef("TRADE_MIN_PRICE", "0.05",
               description="Мин. цена YES/NO для входа"),
    SettingDef("TRADE_MAX_PRICE", "0.95",
               description="Макс. цена YES/NO для входа"),

    # --- Активы ---
    SettingDef("TRADE_ASSETS", "BTC,ETH",
               description="Список торгуемых активов (через запятую)"),
    SettingDef("ACTIVE_FEATURES", "mid_price,spread_pct,volume_5min,log_time_left,is_final_phase,price_deviation,high_price_final,price_velocity,price_momentum,spread_trend,volume_trend,price_distance_from_max,day_of_week",
               description="Список признаков для ML-модели"),

    # --- Крипто ---

    # --- Прочее ---
    SettingDef("BYPASS_BET_SIZE_CHECK", "false", editable=False,
               description="Debug-only. Не открывать через API."),

    SettingDef("ENTRY_STRATEGY", "first",
               description="Стратегия входа: first | best_edge | confirmed"),

    # --- Валидация порогов LightGBM ---
    SettingDef("LGBM_MIN_VALID_THRESHOLD", "0.30",
               description="Минимально допустимый порог LightGBM (ниже → fallback). Ниже 0.30 = модель всегда даёт сигнал"),
    SettingDef("LGBM_MAX_VALID_THRESHOLD", "0.75",
               description="Максимально допустимый порог LightGBM (выше → fallback). Выше 0.75 = модель никогда не сигналит"),
    SettingDef("LGBM_THRESHOLD_FALLBACK", "0.55",
               description="Нейтральный порог при некорректном автоматическом значении"),
    SettingDef("LGBM_MIN_PRECISION_FOR_THRESHOLD", "0.52",
               description="Мин. precision при поиске оптимального порога. 0.52 для крипто, 0.60 для строгого режима"),

    # --- CV / обучение ---
    SettingDef("LGBM_CV_N_SPLITS", "5",
               description="Кол-во фолдов TimeSeriesSplit при обучении LightGBM"),
    SettingDef("LGBM_MAX_SUSPICIOUS_THRESHOLD", "0.95",
               description="Порог подозрения на data leakage при обучении (обычно 0.95)"),

    # --- ML пороги ---

    # --- Комиссии ---
    SettingDef("POLYMARKET_FEE_RATE", "0.002",
               description="Комиссия Polymarket (0.002 = 0.2%). Влияет на расчёт PnL в takeprofit/stoploss workers"),

    # --- Бэктест ---
    SettingDef("BACKTEST_MIN_EDGE", "0.04",
               description="Мин. edge для сигнала в бэктесте"),
    SettingDef("BACKTEST_FEE_PER_TRADE", "0.02",
               description="Комиссия за сделку в бэктесте (0.02 = 2%)"),
    SettingDef("BACKTEST_MIN_TRADES", "10",
               description="Мин. кол-во сделок в бэктесте для успешного обучения"),
    SettingDef("BACKTEST_MIN_PNL", "0.0",
               description="Мин. PnL в бэктесте для успешного обучения"),

    # --- Hyperparameters LightGBM Crypto ---
    SettingDef("CRYPTO_LGBM_N_ESTIMATORS", "300",
               description="Количество деревьев LightGBM для крипто-моделей"),
    SettingDef("CRYPTO_LGBM_LEARNING_RATE", "0.05",
               description="Скорость обучения LightGBM"),
    SettingDef("CRYPTO_LGBM_NUM_LEAVES", "15",
               description="Максимальное число листьев в дереве LightGBM"),
    SettingDef("CRYPTO_LGBM_MAX_DEPTH", "4",
               description="Максимальная глубина дерева LightGBM"),
    SettingDef("CRYPTO_LGBM_MIN_CHILD_SAMPLES", "50",
               description="Мин. количество образцов в листе"),
    SettingDef("CRYPTO_LGBM_SUBSAMPLE", "0.8",
               description="Доля сэмплов для бутстрапа"),
    SettingDef("CRYPTO_LGBM_COLSAMPLE_BYTREE", "1.0",
               description="Доля признаков при сплите дерева"),
    SettingDef("CRYPTO_LGBM_REG_ALPHA", "0.1",
               description="L1 регуляризация LightGBM"),
    SettingDef("CRYPTO_LGBM_REG_LAMBDA", "1.0",
               description="L2 регуляризация LightGBM"),
    SettingDef("LGBM_EPSILON_QUANTILE", "0.70",
               description="Квантиль epsilon-фильтра таргета. 0.70 = учимся на топ-30% движений"),
]


ALL_SETTINGS = REGISTRY


# ── Удобные геттеры ───────────────────────────────────────────────────────────

def registry_keys() -> set[str]:
    """Все ключи реестра (для settings_keys в движке)."""
    return {s.key for s in REGISTRY}


def registry_defaults() -> dict[str, str]:
    """Дефолты для сидера (заменяет жёсткий DEFAULTS-словарь)."""
    return {s.key: s.default for s in REGISTRY}


def editable_keys() -> set[str]:
    """Только ключи, которые можно редактировать через API (без скрытых)."""
    return {s.key for s in REGISTRY if s.editable}

REQUIRED_SETTINGS_KEYS = frozenset([
    "INVERT_LGBM_SIGNAL",
    "FAVOR_MIN_TIME_LEFT_SEC",
    "FAVOR_MAX_TIME_LEFT_SEC",
    "OUTS_MIN_TIME_LEFT_SEC",
    "OUTS_MAX_TIME_LEFT_SEC",
    "TRADE_BET_SIZE_USDC",
    "FLIP_THRESHOLD",
    "TRADE_MAX_PRICE",
    "TRADE_MIN_PRICE",
    "FAVORITE_MAX_PRICE",
    "FAVORITE_MIN_PRICE",
    "MIN_DIRECTION_PROB",
    "MIN_WIN_PROB",
    "COMBINED_DIR_DISCOUNT_WEIGHT",
    "COMBINED_DIR_STRONG_THRESHOLD",
    "MAX_SPREAD_PCT",
    "OUTSIDER_PWIN_DISCOUNT",
    "OUTS_MIN_EDGE",
    "FAVORITE_MIN_EDGE",
    "COMBINED_REQUIRE_CONSENSUS",
    "COMBINED_FALLBACK_TO_LOGREG_ON_NONE",
    "COMBINED_LOGREG_ABSTAIN_BAND",
    "COMBINED_COST_BUFFER",
])

def validate_required_keys():
    """Проверяет наличие обязательных ключей в реестре (для CI)."""
    current_keys = {s.key for s in REGISTRY}
    missing = REQUIRED_SETTINGS_KEYS - current_keys
    if missing:
        raise ValueError(f"Missing required settings in REGISTRY: {missing}")
