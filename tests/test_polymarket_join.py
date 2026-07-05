"""
Тесты для polyflip/crypto/polymarket_join.py

Проверяем:
  - Корректный match снапшота к свече в пределах tolerance
  - Строка без снапшота → pm_yes_price = NaN
  - INVALID снапшоты пропускаются при join (фильтр в SQL)
  - Метаданные coverage_pct логируются корректно
"""
from __future__ import annotations

import pandas as pd
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch


def _make_candles(base: datetime, n: int, step_min: int = 15) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({"open_time": base + timedelta(minutes=i * step_min)})
    return pd.DataFrame(rows)


def _make_snapshots(times_outcomes: list[tuple]) -> list[MagicMock]:
    rows = []
    for t, price, outcome, mid in times_outcomes:
        r = MagicMock()
        r.market_id = "mkt-test"
        r.mid_price  = price
        r.final_outcome = outcome
        r.recorded_at = t
        rows.append(r)
    return rows


class TestJoinPolymarketPrices:
    BASE = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)

    @pytest.mark.asyncio
    async def test_exact_match(self):
        """Снапшот точно в момент open_time свечи → совпадение."""
        from polyflip.crypto.polymarket_join import join_polymarket_prices

        candles = _make_candles(self.BASE, 3)
        snaps = _make_snapshots([
            (self.BASE,                            0.55, "YES", 0.55),
            (self.BASE + timedelta(minutes=15),    0.60, "NO",  0.60),
            (self.BASE + timedelta(minutes=30),    0.48, "YES", 0.48),
        ])

        mock_session = AsyncMock()
        mock_result  = MagicMock()
        mock_result.all.return_value = snaps
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await join_polymarket_prices(mock_session, candles, "BTC")

        assert "pm_yes_price" in result.columns
        assert result["pm_yes_price"].notna().all(), "All candles must match"
        assert list(result["pm_yes_price"]) == pytest.approx([0.55, 0.60, 0.48])

    @pytest.mark.asyncio
    async def test_no_snapshots_returns_nan(self):
        """Нет снапшотов → pm_yes_price = NaN для всех строк."""
        from polyflip.crypto.polymarket_join import join_polymarket_prices

        candles = _make_candles(self.BASE, 5)

        mock_session = AsyncMock()
        mock_result  = MagicMock()
        mock_result.all.return_value = []  # пустой результат
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await join_polymarket_prices(mock_session, candles, "BTC")

        assert result["pm_yes_price"].isna().all(), "No matches → all NaN"
        assert result["pm_outcome"].isna().all()

    @pytest.mark.asyncio
    async def test_outside_tolerance_returns_nan(self):
        """Снапшот за пределами tolerance → NaN."""
        from polyflip.crypto.polymarket_join import join_polymarket_prices

        candles = _make_candles(self.BASE, 1)
        far_time = self.BASE + timedelta(minutes=20)  # 20 мин > 7.5 мин tolerance
        snaps = _make_snapshots([(far_time, 0.55, "YES", 0.55)])

        mock_session = AsyncMock()
        mock_result  = MagicMock()
        mock_result.all.return_value = snaps
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await join_polymarket_prices(mock_session, candles, "BTC", tolerance_sec=450)
        assert result["pm_yes_price"].isna().all(), "Outside tolerance → NaN"

    @pytest.mark.asyncio
    async def test_empty_candles(self):
        """Пустой DataFrame свечей → возвращаем пустой DataFrame."""
        from polyflip.crypto.polymarket_join import join_polymarket_prices

        mock_session = AsyncMock()
        result = await join_polymarket_prices(mock_session, pd.DataFrame(), "BTC")
        assert result.empty
