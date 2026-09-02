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
               description="Ширина мёртвой зоны вокруг flip-порога. Чем больше значение, тем МЕНЬШЕ сделок."),

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
    SettingDef("TAKE_PROFIT_ORDER_MODE", "GTD",
               description="Take-profit order mode: GTD (resting until market end) | TRIGGERED (wait for bid target)"),

    # --- Режим торговли ---
    SettingDef("TRADING_MODE", DEFAULT_TRADING_MODE,
               description="Режим торговли: favorite | combined"),
    SettingDef("TRADING_ENABLED", "false", editable=False,
               description="Управляется только через UI toggle, не через general settings API"),
    SettingDef("LIVE_TRADING_ENABLED", "false", editable=False,
               description="Управляется только через специальный API (kill-switch)"),

    # --- Таймеры / опрос ---
    SettingDef("LIVE_POLL_INTERVAL_SECONDS", str(LIVE_POLL_INTERVAL_SECONDS)),
    SettingDef("LIVE_ORDER_MODE", "MAKER_TTL", description="LIVE order policy: maker-only TTL by default | LIMIT_TTL | FAK | FAK_RETRY"),
    SettingDef("LIVE_GTC_TTL_SECONDS", "10.0", description="Resting order lifetime in seconds before cancellation"),
    SettingDef("LIVE_MAKER_REPRICE_ON_CROSS", "true", description="Reprice a crossed post-only GTC/GTD order once using a fresh book"),
    SettingDef("LIVE_MAKER_REPRICE_MAX_RETRIES", "1", description="Maximum maker reprice attempts after a cross (hard capped at 1)"),
    SettingDef("LIVE_MAKER_TICK_SIZE", "0.01", description="Price tick used for passive maker repricing"),
    SettingDef("LIVE_FAK_RETRY_MAX_ATTEMPTS", "3", description="Maximum retries for transient FAK errors"),
    SettingDef("LIVE_FAK_RETRY_DELAY_SEC", "0.75", description="Delay between transient order retries"),
    SettingDef("PAPER_EXECUTION_PROFILE", "LIVE_PARITY", description="PAPER execution: LIVE_PARITY models delay, depth, slippage, fees and minimum size; INSTANT is test-only"),
    SettingDef("PAPER_LIVE_DELAY_SEC", "2.0", description="Simulated PAPER to LIVE submission delay"),
    SettingDef("PAPER_SLIPPAGE_PCT", "0.5", description="PAPER adverse execution slippage in percent"),
    SettingDef("PAPER_FEE_MODEL", "POLYMARKET_PRICE_DEPENDENT", description="PAPER fee model: POLYMARKET_PRICE_DEPENDENT | FLAT_NOTIONAL"),
    SettingDef("PAPER_FEE_RATE", "0.07", description="PAPER fee-rate parameter; with price-dependent model fee/share = rate × p × (1-p)"),
    SettingDef("PAPER_FEE_EXPONENT", "1.0", description="PAPER fee-curve exponent; in CLOB V2 sourced from fd.e"),
    SettingDef("PAPER_MIN_ORDER_SHARES", "5", description="PAPER minimum outcome-token size"),
    SettingDef("MIRROR_MAX_BACKOFF_SECONDS", "30", description="Maximum retry backoff for the PAPER to LIVE mirror worker"),

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
               description="Мин. edge для ставок против толпы (в долях). Чем больше значение, тем строже фильтр и МЕНЬШЕ сделок."),
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
               description="Порог p_flip для входа в аутсайдера. Чем больше значение, тем МЕНЬШЕ сделок."),
    SettingDef("OUTSIDER_PWIN_DISCOUNT", "0.65",
               description="Множитель вероятности выигрыша аутсайдера (в UI %). Чем больше значение (ближе к 100%), тем слабее штраф и БОЛЬШЕ сделок."),
    SettingDef("MAX_SPREAD_PCT", "0.08",
               description="Макс. спред как доля от mid_price. Чем больше значение, тем БОЛЬШЕ сделок (допускается широкий спред)."),
    SettingDef("TRADE_FLIP_THRESHOLD", "0.85",
               description="Глобальный порог ПРОТИВ ТОЛПЫ (если нет индивидуального)"),

    SettingDef("COMBINED_COST_BUFFER", "0.020",
               description="Буфер транзакционных издержек в Combined-режиме (в USD). Вычитается из gross_edge. Чем больше значение, тем МЕНЬШЕ сделок."),

    SettingDef("MAX_PRICE_DRIFT", "0.10",
               description="Макс. дрейф цены от момента сигнала до исполнения. Чем больше значение, тем БОЛЬШЕ сделок."),

    SettingDef("MIN_DIRECTION_PROB", "0.505",
               description="Мин. уверенность LGBM в направлении (UP/DOWN). Чем больше значение, тем МЕНЬШЕ сделок."),
    SettingDef("MIN_WIN_PROB", "0.51",
               description="Мин. вероятность победы кандидата (LGBM + LogReg) для входа. Чем больше значение, тем МЕНЬШЕ сделок."),
    SettingDef("COMBINED_DIR_DISCOUNT_WEIGHT", "0.0",
               description="Вес дисконта к p_candidate_win за неуверенность LightGBM. Чем больше значение, тем сильнее штраф и МЕНЬШЕ сделок."),
    SettingDef("COMBINED_DIR_STRONG_THRESHOLD", "0.65",
               description="Порог dir_prob, выше которого дисконт равен 0. Чем больше значение, тем сложнее снять штраф и МЕНЬШЕ сделок."),
    SettingDef("COMBINED_REQUIRE_CONSENSUS", "true",
               description="Обе модели (LightGBM и LogReg) должны проголосовать за одно направление. Иначе - SKIP."),
    SettingDef("COMBINED_FALLBACK_TO_LOGREG_ON_NONE", "true",
               description="Если LightGBM выдает NONE, то используется голос LogReg. Если false, то NONE ведет к SKIP."),
    SettingDef("COMBINED_LGBM_UNAVAILABLE_POLICY", "SKIP",
               description="Поведение при недоступности LightGBM: LOGREG_FALLBACK | SKIP"),
    SettingDef("COMBINED_LOGREG_ABSTAIN_BAND", "0.05",
               description="Ширина коридора нерешительности LogReg вокруг 0.50. Чем больше значение, тем чаще LogReg воздерживается (МЕНЬШЕ сделок)."),
    SettingDef(
        key="INVERT_LGBM_SIGNAL",
        default="false",
        description="Инвертировать сигнал LightGBM: если true, p_up↔p_down меняются местами. "
                    "Используется, когда LGBM является контриндикатором (WinRate < 40%).",
    ),
    SettingDef(
        key="ENABLE_ECE_CORRECTION",
        default="true",
        description="Коррекция вероятности p_flip по показателю ECE (Expected Calibration Error).",
    ),
    SettingDef(
        key="LIGHTGBM_DECISION_MODE",
        default="SHADOW",
        description="Режим применения прогнозов LightGBM: SHADOW (теневой аналитический) | ACTIVE (торговый).",
    ),
    SettingDef(
        key="TRADING_POLICY_MODE",
        default="LEGACY",
        description="Политика решения: LEGACY | WEIGHTED_SHADOW | WEIGHTED_ACTIVE. Shadow только считает новую политику.",
    ),
    SettingDef("WEIGHTED_POLICY_ID", "UNVERSIONED",
               description="Immutable identifier of the calibrated weighted-policy artifact."),
    SettingDef("WEIGHTED_MARKET_WEIGHT", "0.90",
               description="Вес текущей рыночной вероятности YES в weighted policy."),
    SettingDef("WEIGHTED_LOGREG_WEIGHT", "0.05",
               description="Вес вероятности LogReg в weighted policy."),
    SettingDef("WEIGHTED_LGBM_WEIGHT", "0.05",
               description="Вес вероятности LightGBM в weighted policy."),
    SettingDef("WEIGHTED_MRF_BETA", "0.0",
               description="Поправка MRF в log-odds; 0 отключает влияние до валидации режима."),
    SettingDef("WEIGHTED_FEE_RATE", "0.07",
               description="Конфигурационный fallback ставки taker-комиссии для weighted policy; фактическая комиссия fill главнее."),
    SettingDef("WEIGHTED_FEE_EXPONENT", "1.0",
               description="Fallback exponent fee curve для weighted policy; в CLOB V2 берётся из fd.e."),
    SettingDef("WEIGHTED_SLIPPAGE_RATE", "0.005",
               description="Оценка проскальзывания weighted policy как доля цены (0.005 = 0.5%)."),
    SettingDef("WEIGHTED_EXECUTION_ROLE", "TAKER",
               description="Роль для оценки затрат weighted policy: TAKER | MAKER."),
    SettingDef("WEIGHTED_INTERCEPT", "0.0",
               description="Калибровочный intercept в log-odds weighted policy."),
    SettingDef("WEIGHTED_MAKER_FEE_RATE", "0.0",
               description="Fallback maker fee rate для weighted policy."),
    SettingDef("WEIGHTED_LATENCY_BUFFER", "0.0",
               description="Дополнительная стоимость задержки, USDC на одну share."),
    SettingDef("WEIGHTED_MIN_NET_EV_FAVORITE", "0.03",
               description="Минимальный net EV фаворита, USDC на одну share."),
    SettingDef("WEIGHTED_MIN_NET_EV_OUTSIDER", "0.03",
               description="Минимальный net EV аутсайдера, USDC на одну share."),
    SettingDef("WEIGHTED_FIXED_BET_USDC", "1.0",
               description="Фиксированный бюджет weighted active до валидации sizing."),
    SettingDef("WEIGHTED_MRF_EXTREME_VETO_THRESHOLD", "-1.0",
               description="Экспериментальный extreme MRF veto; -1.0 отключает veto."),
    SettingDef("WEIGHTED_MODELS_AGREE_BETA", "0.0",
               description="Поправка за согласие LogReg и LightGBM в log-odds."),
    SettingDef("WEIGHTED_MRF_APPLICATION", "PROBABILITY",
               description="Применение MRF: PROBABILITY или STAKE."),
    SettingDef("WEIGHTED_MRF_SIZING_GAMMA", "0.0",
               description="Множитель MRF для размера ставки в STAKE-режиме."),
    SettingDef("WEIGHTED_POLICY_ARTIFACT_PATH", "",
               description="Путь к immutable weighted-policy artifact внутри API-контейнера."),
    SettingDef("WEIGHTED_SIZING_MODE", "FIXED",
               description="Sizing: FIXED | LOWER_BOUND_KELLY | STEPPED_EDGE."),
    SettingDef("WEIGHTED_STANDARD_ERROR", "0.0",
               description="OOF uncertainty для conservative lower-bound sizing."),
    SettingDef("WEIGHTED_KELLY_FRACTION", "0.025",
               description="Доля Kelly для LOWER_BOUND_KELLY sizing."),
    SettingDef("WEIGHTED_SIZE_CAP_USDC", "3.0",
               description="Верхний cap размера weighted ставки в USDC."),


    # --- Обучение LogReg / Phase models ---
    SettingDef("TRAIN_MAX_PARALLEL_JOBS", "2",
               description="Максимальное количество параллельных задач обучения моделей"),
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
    SettingDef("LGBM_TARGET_COVERAGE", "0.40",
               description="Целевое покрытие направленных сигналов LightGBM (0.20 / 0.40 / 0.60 / 0.80)"),
    SettingDef("LGBM_CALIBRATION_METHOD", "PLATT",
               description="Калибровка p_win: PLATT, AUTO или ISOTONIC; AUTO допускает isotonic только при OOT-улучшении"),

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
    SettingDef("CRYPTO_LGBM_N_JOBS", "2",
               description="LightGBM worker threads per model"),
    SettingDef("CRYPTO_LGBM_EARLY_STOPPING_ROUNDS", "30",
               description="Early stopping rounds for LightGBM"),
    SettingDef("CRYPTO_LGBM_HYPERPARAM_SEARCH_TRIALS", "1",
               description="Bounded hyperparameter search trials"),
    SettingDef("CRYPTO_LGBM_REG_LAMBDA", "1.0",
               description="L2 регуляризация LightGBM"),
    SettingDef("LGBM_EPSILON_QUANTILE", "0.70",
               description="Квантиль epsilon-фильтра таргета. 0.70 = учимся на топ-30% движений"),

    # --- Market Regime Filter (MRF) ---
    SettingDef("MARKET_REGIME_FILTER_MODE", "OFF",
               description="Режим MRF: OFF (выключен) | SHADOW (запись логов) | ACTIVE (торговый фильтр)"),
    SettingDef("MARKET_REGIME_FILTER_VERSION", "1",
               description="Версия MRF для сопоставления телеметрии с экспериментами"),
    SettingDef("MARKET_REGIME_MIN_HISTORY", "97",
               description="Минимальное количество 15-мин свечей для построения снапшота (≈24ч)"),
    SettingDef("MARKET_REGIME_OUTSIDER_TREND_MULTIPLIER", "0.0",
               description="Мультипликатор размера ставки на аутсайдера в TREND (0.0 = не менять)"),
    SettingDef("MARKET_REGIME_UNKNOWN_MULTIPLIER", "0.8",
               description="Мультипликатор размера ставки при UNKNOWN регионе (0.8 = снижение на 20%)"),
    SettingDef("MARKET_REGIME_BREADTH_THRESHOLD", "0.65",
               description="Порог breadth (доля активов с ret>0) для определения TREND (0.65 = 65% активов растут)"),
    SettingDef("MARKET_REGIME_EFFICIENCY_THRESHOLD", "0.40",
               description="Порог рыночной эффективности (efficiency = mean_abs_ret / volatility). 0.40 = 40% эффективности"),
    SettingDef("MARKET_REGIME_VETO_THRESHOLD", "0.15",
               description="MRF v3: порог отрицательного regime evidence для veto"),
    SettingDef("MARKET_REGIME_EDGE_OVERRIDE_MARGIN", "0.05",
               description="MRF v3: минимальный запас edge для override veto"),
    SettingDef("MARKET_REGIME_ASSET_WEIGHT", "0.70",
               description="MRF v3: вес режима отдельного актива"),
    SettingDef("MARKET_REGIME_GLOBAL_WEIGHT", "0.30",
               description="MRF v3: вес глобального режима рынка"),
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
    "ENABLE_ECE_CORRECTION",
    "LIGHTGBM_DECISION_MODE",
    "TRADING_POLICY_MODE",
    "WEIGHTED_POLICY_ID",
    "WEIGHTED_MARKET_WEIGHT",
    "WEIGHTED_LOGREG_WEIGHT",
    "WEIGHTED_LGBM_WEIGHT",
    "WEIGHTED_MRF_BETA",
    "WEIGHTED_FEE_RATE",
    "WEIGHTED_FEE_EXPONENT",
    "WEIGHTED_SLIPPAGE_RATE",
    "WEIGHTED_EXECUTION_ROLE",
    "WEIGHTED_INTERCEPT",
    "WEIGHTED_MAKER_FEE_RATE",
    "WEIGHTED_LATENCY_BUFFER",
    "WEIGHTED_MIN_NET_EV_FAVORITE",
    "WEIGHTED_MIN_NET_EV_OUTSIDER",
    "WEIGHTED_FIXED_BET_USDC",
    "WEIGHTED_MRF_EXTREME_VETO_THRESHOLD",
])

def validate_required_keys():
    """Проверяет наличие обязательных ключей в реестре (для CI)."""
    current_keys = {s.key for s in REGISTRY}
    missing = REQUIRED_SETTINGS_KEYS - current_keys
    if missing:
        raise ValueError(f"Missing required settings in REGISTRY: {missing}")
