# polyflip/api/backtest_schemas.py
from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional
from datetime import datetime


# ──────────────────────────────────────────
# INPUT: параметры запуска бэктеста
# ──────────────────────────────────────────

class BacktestConfig(BaseModel):
    """Параметры одного прогона бэктеста. Все поля с дефолтами → можно запустить без настроек."""

    # Фильтрация данных
    assets: list[str] = Field(default=["BTC", "ETH"], description="Список ассетов для теста")
    date_from: Optional[datetime] = Field(default=None, description="Начало периода (UTC)")
    date_to: Optional[datetime] = Field(default=None, description="Конец периода (UTC)")
    min_snapshots_per_market: int = Field(default=3, ge=1, le=50)

    # Торговое окно
    min_time_left_min: float = Field(default=1.0, ge=0.1, le=1440.0)
    max_time_left_min: float = Field(default=60.0, ge=1.0, le=1440.0)

    # Стратегия
    strategy_mode: Literal["ML", "PURE_FAVORITE"] = Field(default="ML")
    trade_on_flip: bool = Field(default=False, description="Включить OUTSIDER стратегию")

    # Пороги ML
    no_flip_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    flip_threshold: float = Field(default=0.60, ge=0.0, le=1.0)

    # Пороги PURE_FAVORITE
    favorite_threshold: float = Field(default=0.65, ge=0.5, le=0.99)
    yes_min_price: float = Field(default=0.55, ge=0.01, le=0.99)
    yes_max_price: float = Field(default=0.95, ge=0.01, le=0.99)
    no_min_price: float = Field(default=0.55, ge=0.01, le=0.99)
    no_max_price: float = Field(default=0.95, ge=0.01, le=0.99)

    # Dead zone
    auto_dead_zone_width: float = Field(default=0.10, ge=0.0, le=0.5)

    # Размер ставки
    kelly_enabled: bool = Field(default=True)
    kelly_multiplier: float = Field(default=0.25, ge=0.01, le=1.0)
    initial_capital: float = Field(default=1000.0, ge=10.0, le=1_000_000.0)
    trade_bet_size_usdc: float = Field(default=5.0, ge=1.0)
    max_bet_size_usdc: float = Field(default=50.0, ge=1.0)
    min_edge: float = Field(default=-0.05, ge=-1.0, le=1.0)
    max_edge: float = Field(default=0.50, ge=-1.0, le=1.0)

    # Исполнение
    slippage_pct: float = Field(default=0.005, ge=0.0, le=0.10)

    # Модель (None = брать активную из ModelRegistry)
    model_id: Optional[int] = Field(default=None)

    @field_validator("max_time_left_min")
    @classmethod
    def max_gt_min(cls, v, info):
        if "min_time_left_min" in info.data and v <= info.data["min_time_left_min"]:
            raise ValueError("max_time_left_min must be > min_time_left_min")
        return v

    def to_runner_config(self) -> dict:
        """Конвертирует в dict для BacktestRunner (ключи = SCREAMING_SNAKE)."""
        return {
            "MIN_TIME_LEFT_MIN": self.min_time_left_min,
            "MAX_TIME_LEFT_MIN": self.max_time_left_min,
            "STRATEGY_MODE": self.strategy_mode,
            "TRADE_ON_FLIP": str(self.trade_on_flip).lower(),
            "NO_FLIP_THRESHOLD": self.no_flip_threshold,
            "FLIP_THRESHOLD": self.flip_threshold,
            "FAVORITE_THRESHOLD": self.favorite_threshold,
            "YES_MIN_PRICE": self.yes_min_price,
            "YES_MAX_PRICE": self.yes_max_price,
            "NO_MIN_PRICE": self.no_min_price,
            "NO_MAX_PRICE": self.no_max_price,
            "AUTO_DEAD_ZONE_WIDTH": self.auto_dead_zone_width,
            "KELLY_ENABLED": str(self.kelly_enabled).lower(),
            "KELLY_MULTIPLIER": self.kelly_multiplier,
            "INITIAL_CAPITAL": self.initial_capital,
            "TRADE_BET_SIZE_USDC": self.trade_bet_size_usdc,
            "MAX_BET_SIZE_USDC": self.max_bet_size_usdc,
            "MIN_EDGE": self.min_edge,
            "MAX_EDGE": self.max_edge,
            "SLIPPAGE_PCT": self.slippage_pct,
        }


# ──────────────────────────────────────────
# OUTPUT: результаты прогона
# ──────────────────────────────────────────

class StrategyBreakdown(BaseModel):
    strategy: str
    trades: int
    net_pnl: float
    win_rate_pct: float
    avg_edge: Optional[float]
    avg_kelly: Optional[float]


class AssetBreakdown(BaseModel):
    asset: str
    trades: int
    net_pnl: float
    win_rate_pct: float


class EquityCurvePoint(BaseModel):
    trade_index: int
    cumulative_pnl: float
    trade_pnl: float
    market_id: str
    asset: str
    strategy: str
    outcome: str          # "WIN" / "LOSS"
    p_flip: Optional[float]
    edge: Optional[float]
    bet_size: float
    executed_price: float


class BacktestResult(BaseModel):
    # Мета
    run_id: str                   # uuid4 для кэша
    config: BacktestConfig
    started_at: datetime
    finished_at: datetime
    duration_sec: float

    # Датасет
    total_markets_loaded: int
    tradeable_markets: int
    skipped_markets: int

    # Сводные метрики
    total_trades: int
    total_invested: float
    net_profit: float
    roi_pct: float
    win_rate_pct: float
    avg_trade_pnl: float
    max_drawdown_pct: float       # максимальная просадка в %
    sharpe_ratio: Optional[float]
    profit_factor: float          # gross_profit / gross_loss

    # Детализация
    strategies: list[StrategyBreakdown]
    assets: list[AssetBreakdown]
    equity_curve: list[EquityCurvePoint]  # для графика

    # Топ-10 лучших и худших сделок
    top_trades: list[EquityCurvePoint]
    worst_trades: list[EquityCurvePoint]


class BacktestRunResponse(BaseModel):
    run_id: str
    status: Literal["running", "completed", "error"]
    message: str
    result: Optional[BacktestResult] = None
