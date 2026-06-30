"""
MarketReplay: загружает снимки одного рынка и предоставляет интерфейс реплея.
Не зависит от БД напрямую — получает уже загруженные ORM-объекты.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from polyflip.trading.feature_builder import MarketSignal, signal_from_snapshot_row


@dataclass(frozen=True)
class MarketTick:
    """Один момент времени в рынке."""
    market_id: str
    asset: str
    time_left_min: float
    mid_price: float
    spread: float
    volume_5min: float
    price_velocity: float
    hour_of_day: int
    final_outcome: str       # "YES" / "NO" — известен только постфактум
    recorded_at: datetime

    def to_signal(self) -> MarketSignal:
        return MarketSignal(
            asset=self.asset,
            mid_price=self.mid_price,
            spread=self.spread,
            volume_5min=self.volume_5min,
            price_velocity=self.price_velocity,
            hour_of_day=self.hour_of_day,
            time_left_min=self.time_left_min,
        )


class MarketReplay:
    """
    Реплей одного рынка по его историческим снимкам.
    
    Тики отсортированы от начала рынка к концу:
      ticks[0] → самый ранний снимок (max time_left_min)
      ticks[-1] → самый поздний снимок (min time_left_min)
    """

    def __init__(self, snapshots: list) -> None:
        if not snapshots:
            raise ValueError("Cannot create MarketReplay with empty snapshots")

        self.ticks: list[MarketTick] = sorted(
            [self._to_tick(s) for s in snapshots],
            key=lambda t: -t.time_left_min   # убывающий time_left = от начала к концу
        )
        self.market_id: str = snapshots[0].market_id
        self.asset: str = snapshots[0].asset
        self.final_outcome: str = snapshots[0].final_outcome  # YES / NO / INVALID

    def _to_tick(self, snapshot) -> MarketTick:
        return MarketTick(
            market_id=snapshot.market_id,
            asset=snapshot.asset,
            time_left_min=float(snapshot.time_left_min or 0),
            mid_price=float(snapshot.mid_price or 0.5),
            spread=float(snapshot.spread or 0.02),
            volume_5min=float(snapshot.volume_5min or 0),
            price_velocity=float(snapshot.price_velocity or 0),
            hour_of_day=int(snapshot.hour_of_day or 0),
            final_outcome=snapshot.final_outcome or "INVALID",
            recorded_at=snapshot.recorded_at or datetime.utcnow(),
        )

    def get_entry_tick(
        self,
        min_time_min: float,
        max_time_min: float,
    ) -> Optional[MarketTick]:
        """
        Возвращает первый тик в торговом окне [min_time_min, max_time_min].
        «Первый» = самый ранний момент когда рынок был в окне (max time_left).
        Имитирует поведение реального движка: вход при первой возможности.
        """
        for tick in self.ticks:
            if min_time_min <= tick.time_left_min <= max_time_min:
                return tick
        return None

    def get_ticks_in_window(
        self,
        min_time_min: float,
        max_time_min: float,
    ) -> list[MarketTick]:
        return [t for t in self.ticks if min_time_min <= t.time_left_min <= max_time_min]

    @property
    def is_tradeable(self) -> bool:
        """Рынок пригоден для бэктеста (resolved, не INVALID)."""
        return self.final_outcome in ("YES", "NO")

    @property
    def snapshot_count(self) -> int:
        return len(self.ticks)


def group_snapshots_into_replays(snapshots: list) -> dict[str, MarketReplay]:
    """
    Группирует список MarketSnapshot по market_id.
    Возвращает dict: market_id → MarketReplay.
    Фильтрует рынки с < 3 снимками (недостаточно данных).
    """
    groups: dict[str, list] = {}
    for snap in snapshots:
        groups.setdefault(snap.market_id, []).append(snap)

    replays = {}
    for market_id, group in groups.items():
        if len(group) < 3:
            continue
        try:
            replay = MarketReplay(group)
            if replay.is_tradeable:
                replays[market_id] = replay
        except Exception:
            pass  # пропускаем битые данные
    return replays
