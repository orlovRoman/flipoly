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
    FAVORITE_MODE_ENTRY_SEC,
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
    SettingDef("AUTO_DEAD_ZONE", "true",
               description="Авто-расчёт границ зоны по калибровке модели"),

    # --- Финансы / потери ---
    SettingDef("DAILY_LOSS_LIMIT_USDC", "-100.0",
               description="Дневной стоп-лосс в USDC (отрицательное число)"),
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
               description="Режим: ml | favorite | lightgbm | combined"),
    SettingDef("TRADING_ENABLED", "false", editable=False,
               description="Управляется только через UI toggle, не через general settings API"),

    # --- Таймеры / опрос ---
    SettingDef("FAVORITE_MODE_ENTRY_SEC", str(FAVORITE_MODE_ENTRY_SEC)),
    SettingDef("LIVE_POLL_INTERVAL_SECONDS", str(LIVE_POLL_INTERVAL_SECONDS)),
    SettingDef("TRADE_MIN_TIME_LEFT_SEC", "10"),
    SettingDef("TRADE_MAX_TIME_LEFT_SEC", "360"),
    SettingDef("TRADE_EXECUTION_TIME_SEC", "30"),

    # --- Сайзинг ---
    SettingDef("TRADE_BET_SIZE_USDC", "5.0",
               description="Минимальная ставка (USDC)"),
    SettingDef("MAX_BET_SIZE_USDC", "50.0",
               description="Максимальная ставка (USDC)"),
    SettingDef("BET_SIZING_MODE", "scaled",
               description="Режим расчёта размера: scaled | fixed"),
    SettingDef("LIQUIDITY_FRACTION", "0.05",
               description="Макс. доля от volume_5min на одну ставку"),

    # --- Edge ---
    SettingDef("MIN_EDGE", "0.05",
               description="Мин. edge для входа в сделку"),
    SettingDef("MAX_EDGE_FILTER", "0.20",
               description="Фильтр аномального edge: SKIP если edge > этого значения"),

    # --- Фаворит ---
    SettingDef("FAVORITE_THRESHOLD", "0.55",
               description="Граница фаворит/аутсайдер по mid_price"),
    SettingDef("FAVORITE_MIN_EDGE", "-0.01",
               description="Мин. edge для PURE_FAVORITE (мягче чем ML)"),
    SettingDef("FAVORITE_MIN_PRICE", "0.55",
               description="Мин. цена для входа в фаворита"),
    SettingDef("FAVORITE_MAX_PRICE", "0.95",
               description="Макс. цена для входа в фаворита"),

    # --- Аутсайдер / NO ---
    SettingDef("NO_MIN_EDGE", "0.04",
               description="Мин. edge для ставки на аутсайдера (NO)"),
    SettingDef("OUTSIDER_MAX_PRICE", "0.45",
               description="Макс. цена покупки аутсайдера"),
    SettingDef("TRADE_ON_FAVORITE", "true",
               description="Если включено, бот делает ставки по тренду на фаворита (Pure Favorite/ML Trend/Crypto Trend)."),
    SettingDef("TRADE_ON_FLIP", "false",
               description="Торговать на флип (стратегия аутсайдера)"),
    SettingDef("FLIP_THRESHOLD", "0.60",
               description="Порог p_flip для входа в аутсайдера"),
    SettingDef("TRADE_FLIP_THRESHOLD", "0.85",
               description="Глобальный порог ПРОТИВ ТОЛПЫ (если нет индивидуального)"),

    # --- ML ---
    SettingDef("TRADE_NO_FLIP_THRESHOLD", "0.35",
               description="p_flip < этого → торгуем фаворита (ML режим)"),
    SettingDef("COMBINED_NONE_BET_MULTIPLIER", "0.5",
               description="Множитель размера ставки при неопределенности (LGBM=NONE) в Combined-режиме (0.0 - 1.0)"),
    SettingDef("MAX_PRICE_DRIFT", "0.10",
               description="Макс. дрейф цены от момента сигнала до исполнения"),

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
    SettingDef("ACTIVE_FEATURES", "time_left_min,mid_price,spread,volume_5min,price_velocity,hour_of_day",
               description="Список признаков для ML-модели"),

    # --- Крипто ---
    SettingDef("CRYPTO_MIN_EDGE", "0.05",
               description="Мин. edge для стратегии Crypto Trend"),
    SettingDef("USE_CRYPTO_CONFIRM", "false",
               description="Требовать подтверждение сигнала для крипто"),
    SettingDef("CRYPTO_STANDALONE", "false",
               description="Крипто-режим без привязки к Polymarket-рынку"),

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
    SettingDef("NO_FLIP_THRESHOLD", "0.35",
               description="p_flip < этого → ML_TREND покупает фаворита (не аутсайдера)"),

    # --- Комиссии ---
    SettingDef("POLYMARKET_FEE_RATE", "0.002",
               description="Комиссия Polymarket (0.002 = 0.2%). Влияет на расчёт PnL в takeprofit/stoploss workers"),

    # --- Бэктест ---
    SettingDef("BACKTEST_MIN_EDGE", "0.04",
               description="Мин. edge для сигнала в бэктесте"),
    SettingDef("BACKTEST_TRAIN_RATIO", "0.70",
               description="Доля обучающей выборки при walk-forward бэктесте (0.70 = 70%)"),

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


# ── Удобные геттеры ───────────────────────────────────────────────────────────

def registry_keys() -> set[str]:
    """Все ключи реестра (для settings_keys в движке)."""
    return {s.key for s in REGISTRY}


def registry_defaults() -> dict[str, str]:
    """Дефолты для сидера (заменяет жёсткий DEFAULTS-словарь)."""
    return {s.key: s.default for s in REGISTRY}


def editable_keys() -> set[str]:
    """Ключи, разрешённые для изменения через API."""
    return {s.key for s in REGISTRY if s.editable}
