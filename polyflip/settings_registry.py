"""
polyflip/settings_registry.py

Единый реестр runtime-настроек.
Каждый параметр описан ровно один раз — дефолт берётся из constants.py.

Использование:
  - init_runtime_settings.py:  DEFAULTS = registry_defaults()
  - api/settings.py:           valid_keys = list(editable_keys())
  - trading/engine.py:         settings_keys = list(registry_keys())

Это гарантирует: если добавили ключ в реестр — он автоматически появляется
в сидере, в API и в движке. Рассинхронизация становится структурно невозможной.
"""
from dataclasses import dataclass
from typing import Optional, Callable

from polyflip.constants import (
    DEAD_ZONE_WIDTH,
    DAILY_LOSS_LIMIT_USDC,
    DEFAULT_TRADING_MODE,
    FAVORITE_MODE_ENTRY_SEC,
    LIVE_POLL_INTERVAL_SECONDS,
    MIN_EDGE,
    MAX_EDGE_SCALING,    # потолок масштабирования ставки → MAX_BET_EDGE
    MAX_EDGE_FILTER,     # фильтр аномального edge
    FAVORITE_THRESHOLD,
    FAVORITE_MIN_EDGE,
    NO_MIN_EDGE,
    CRYPTO_MIN_EDGE,
    FLIP_THRESHOLD,
    OUTSIDER_MAX_PRICE,
    LIQUIDITY_FRACTION,
    FAVORITE_MIN_PRICE,
    FAVORITE_MAX_PRICE,
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
    SettingDef("DEAD_ZONE_WIDTH", str(DEAD_ZONE_WIDTH),
               description="Ширина мёртвой зоны вокруг flip-порога (единый параметр)"),
    SettingDef("AUTO_DEAD_ZONE", "true",
               description="Авто-расчёт границ зоны по калибровке модели"),

    # --- Финансы / потери ---
    SettingDef("DAILY_LOSS_LIMIT_USDC", str(DAILY_LOSS_LIMIT_USDC),
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
    SettingDef("LIQUIDITY_FRACTION", str(LIQUIDITY_FRACTION),
               description="Макс. доля от volume_5min на одну ставку"),

    # --- Edge ---
    SettingDef("MIN_EDGE", str(MIN_EDGE),
               description="Мин. edge для входа в сделку"),
    SettingDef("MAX_BET_EDGE", str(MAX_EDGE_SCALING),
               description="Потолок масштабирования ставки (при edge=MAX_BET_EDGE → макс. ставка)"),
    SettingDef("MAX_EDGE_FILTER", str(MAX_EDGE_FILTER),
               description="Фильтр аномального edge: SKIP если edge > этого значения"),

    # --- Фаворит ---
    SettingDef("FAVORITE_THRESHOLD", str(FAVORITE_THRESHOLD),
               description="Граница фаворит/аутсайдер по mid_price"),
    SettingDef("FAVORITE_MIN_EDGE", str(FAVORITE_MIN_EDGE),
               description="Мин. edge для PURE_FAVORITE (мягче чем ML)"),
    SettingDef("FAVORITE_MIN_PRICE", str(FAVORITE_MIN_PRICE),
               description="Мин. цена для входа в фаворита"),
    SettingDef("FAVORITE_MAX_PRICE", str(FAVORITE_MAX_PRICE),
               description="Макс. цена для входа в фаворита"),

    # --- Аутсайдер / NO ---
    SettingDef("NO_MIN_EDGE", str(NO_MIN_EDGE),
               description="Мин. edge для ставки на аутсайдера (NO)"),
    SettingDef("OUTSIDER_MAX_PRICE", str(OUTSIDER_MAX_PRICE),
               description="Макс. цена покупки аутсайдера"),
    SettingDef("TRADE_ON_FAVORITE", "true",
               description="Если включено, бот делает ставки по тренду на фаворита (Pure Favorite/ML Trend/Crypto Trend)."),
    SettingDef("TRADE_ON_FLIP", "false",
               description="Торговать на флип (стратегия аутсайдера)"),
    SettingDef("FLIP_THRESHOLD", str(FLIP_THRESHOLD),
               description="Порог p_flip для входа в аутсайдера"),
    SettingDef("TRADE_FLIP_THRESHOLD", "0.85",
               description="Глобальный порог ПРОТИВ ТОЛПЫ (если нет индивидуального)"),

    # --- ML ---
    SettingDef("TRADE_NO_FLIP_THRESHOLD", "0.15",
               description="p_flip < этого → торгуем фаворита (ML режим)"),
    SettingDef("MAX_PRICE_DRIFT", "0.10",
               description="Макс. дрейф цены от момента сигнала до исполнения"),

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
    SettingDef("CRYPTO_MIN_EDGE", str(CRYPTO_MIN_EDGE),
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
