"""
Самотесты для rows_to_replays — проверяем баги #1 и #2.
Запуск: pytest tests/test_rows_to_replays.py -v
"""
from datetime import datetime, timezone
import pytest
from polyflip.backtesting.market_replay import rows_to_replays, SnapshotRow, MarketReplay

def _make_row(market_id="mkt-1", asset="BTC", minutes_left=30.0,
              outcome="YES", idx=0):
    """Фабрика валидного SnapshotRow со всеми полями."""
    return SnapshotRow(
        id=idx,
        market_id=market_id,
        asset=asset,
        recorded_at=datetime(2024, 1, 1, 12, idx, tzinfo=timezone.utc),
        mid_price=0.65,
        price_velocity=0.01,
        time_left_min=minutes_left,
        final_outcome=outcome,
        p_flip=0.3,
        volume_5min=1000.0,
        spread=0.02,        # ← баг #1: должно быть в namedtuple
        hour_of_day=12,     # ← баг #1: должно быть в namedtuple
    )


class TestSnapshotRowFields:
    """Баг #1: SnapshotRow должен иметь spread и hour_of_day."""

    def test_has_spread_field(self):
        row = _make_row()
        assert hasattr(row, "spread"), "SnapshotRow missing 'spread' field"
        assert row.spread == 0.02

    def test_has_hour_of_day_field(self):
        row = _make_row()
        assert hasattr(row, "hour_of_day"), "SnapshotRow missing 'hour_of_day' field"
        assert row.hour_of_day == 12

    def test_to_tick_does_not_raise(self):
        """_to_tick внутри MarketReplay не должен бросать AttributeError."""
        rows = [_make_row(minutes_left=30.0 - i, idx=i) for i in range(3)]
        # Не должно быть исключений
        replay = MarketReplay(rows)
        assert len(replay.ticks) == 3

    def test_tick_spread_populated(self):
        rows = [_make_row(minutes_left=30.0 - i, idx=i) for i in range(3)]
        replay = MarketReplay(rows)
        assert replay.ticks[0].spread == 0.02

    def test_tick_hour_of_day_populated(self):
        rows = [_make_row(minutes_left=30.0 - i, idx=i) for i in range(3)]
        replay = MarketReplay(rows)
        assert replay.ticks[0].hour_of_day == 12


class TestRowsToReplays:
    """Интеграционные тесты rows_to_replays."""

    def test_basic_grouping(self):
        rows = [_make_row("mkt-1", minutes_left=30.0 - i, idx=i) for i in range(5)]
        replays = rows_to_replays(rows, min_snapshots=1)
        assert "mkt-1" in replays
        assert len(replays["mkt-1"].ticks) == 5

    def test_filters_non_tradeable(self):
        """Рынки с PENDING/INVALID outcome не должны попасть в replays."""
        rows_yes     = [_make_row("mkt-yes",     outcome="YES",     minutes_left=30 - i, idx=i)   for i in range(3)]
        rows_pending = [_make_row("mkt-pending", outcome="PENDING", minutes_left=30 - i, idx=i+3) for i in range(3)]
        rows_invalid = [_make_row("mkt-invalid", outcome="INVALID", minutes_left=30 - i, idx=i+6) for i in range(3)]
        
        replays = rows_to_replays(rows_yes + rows_pending + rows_invalid)
        assert "mkt-yes"     in replays
        assert "mkt-pending" not in replays, "PENDING markets should be filtered"
        assert "mkt-invalid" not in replays, "INVALID markets should be filtered"

    def test_min_snapshots_filter(self):
        rows_big   = [_make_row("mkt-big",   minutes_left=30 - i, idx=i)   for i in range(5)]
        rows_small = [_make_row("mkt-small", minutes_left=30 - i, idx=i+5) for i in range(2)]
        
        replays = rows_to_replays(rows_big + rows_small, min_snapshots=3)
        assert "mkt-big"   in replays
        assert "mkt-small" not in replays, "Markets with < min_snapshots should be excluded"

    def test_ticks_sorted_descending_time_left(self):
        """MarketReplay.ticks отсортированы от max к min time_left_min."""
        rows = [_make_row(minutes_left=float(t), idx=i) for i, t in enumerate([5, 20, 10, 30, 15])]
        replays = rows_to_replays(rows)
        ticks = replays["mkt-1"].ticks
        times = [t.time_left_min for t in ticks]
        assert times == sorted(times, reverse=True), f"Expected descending, got {times}"

    def test_multiple_markets(self):
        rows = (
            [_make_row("mkt-A", minutes_left=30 - i, idx=i)    for i in range(4)] +
            [_make_row("mkt-B", minutes_left=30 - i, idx=i+4)  for i in range(4)] +
            [_make_row("mkt-C", minutes_left=30 - i, idx=i+8,
                       outcome="NO")                            for i in range(4)]
        )
        replays = rows_to_replays(rows)
        assert len(replays) == 3
        assert all(mid in replays for mid in ["mkt-A", "mkt-B", "mkt-C"])

    def test_empty_rows(self):
        replays = rows_to_replays([])
        assert replays == {}

    def test_identical_to_group_snapshots_for_valid_markets(self):
        """rows_to_replays и group_snapshots_into_replays дают одинаковые market_ids."""
        from polyflip.backtesting.market_replay import group_snapshots_into_replays
        from unittest.mock import MagicMock

        # Создаём mock ORM-объекты с теми же данными
        rows = [_make_row("mkt-X", minutes_left=30 - i, idx=i) for i in range(5)]
        
        orm_snaps = []
        for r in rows:
            m = MagicMock()
            for field in r._fields:
                setattr(m, field, getattr(r, field))
            orm_snaps.append(m)

        replays_new = rows_to_replays(rows)
        replays_old = group_snapshots_into_replays(orm_snaps, min_snapshots=1)

        assert set(replays_new.keys()) == set(replays_old.keys())


class TestRunCpuTask:
    """Баг #3: проверяем что _run_cpu_task использует get_running_loop."""

    @pytest.mark.asyncio
    async def test_timeout_raises_http_exception(self):
        import time
        from fastapi import HTTPException
        from polyflip.api.backtest_api import _run_cpu_task

        def slow_fn():
            time.sleep(10)

        with pytest.raises(HTTPException) as exc_info:
            await _run_cpu_task(slow_fn, timeout_sec=0.1)
        assert exc_info.value.status_code == 504

    @pytest.mark.asyncio
    async def test_returns_result(self):
        from polyflip.api.backtest_api import _run_cpu_task

        result = await _run_cpu_task(lambda x: x * 2, 21)
        assert result == 42
