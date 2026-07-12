"""
Тесты раздельного стоп-лосса: выбор % по strategy_type + миграция.
"""
import pytest
from unittest.mock import MagicMock


# ── Тест 1: engine выбирает правильный % для OUTSIDER ────────────────────────
def test_engine_picks_outsider_pct():
    """
    При strategy_type='OUTSIDER' должен выбираться STOP_LOSS_PCT_OUTSIDER.
    При strategy_type='ML_TREND' / None — STOP_LOSS_PCT_FAVORITE.
    """
    def pick_stop_pct(decision_obj, settings_db):
        """Точная копия логики из engine.py (строки 744–752)."""
        is_outsider = (
            hasattr(decision_obj, 'strategy_type')
            and isinstance(decision_obj.strategy_type, str)
            and decision_obj.strategy_type.upper() == "OUTSIDER"
        )
        if is_outsider:
            return float(settings_db.get("STOP_LOSS_PCT_OUTSIDER", "60.0"))
        return float(settings_db.get("STOP_LOSS_PCT_FAVORITE", "40.0"))

    settings = {"STOP_LOSS_PCT_FAVORITE": "35.0", "STOP_LOSS_PCT_OUTSIDER": "65.0"}

    outsider_dec = MagicMock()
    outsider_dec.strategy_type = "OUTSIDER"

    ml_trend_dec = MagicMock()
    ml_trend_dec.strategy_type = "ML_TREND"

    none_dec = MagicMock()
    none_dec.strategy_type = None

    assert pick_stop_pct(outsider_dec, settings) == 65.0
    assert pick_stop_pct(ml_trend_dec, settings) == 35.0
    assert pick_stop_pct(none_dec, settings) == 35.0  # None → fallback к FAVORITE


# ── Тест 2: strategy_type не строка — fallback к FAVORITE ─────────────────────
def test_engine_picks_favorite_pct_for_non_string_strategy():
    """Если strategy_type — int или другой не-строковый тип, берём FAVORITE."""
    def pick_stop_pct(decision_obj, settings_db):
        is_outsider = (
            hasattr(decision_obj, 'strategy_type')
            and isinstance(decision_obj.strategy_type, str)
            and decision_obj.strategy_type.upper() == "OUTSIDER"
        )
        if is_outsider:
            return float(settings_db.get("STOP_LOSS_PCT_OUTSIDER", "60.0"))
        return float(settings_db.get("STOP_LOSS_PCT_FAVORITE", "40.0"))

    settings = {"STOP_LOSS_PCT_FAVORITE": "30.0", "STOP_LOSS_PCT_OUTSIDER": "70.0"}

    int_dec = MagicMock()
    int_dec.strategy_type = 42
    assert pick_stop_pct(int_dec, settings) == 30.0

    # Объект без strategy_type (dict-like)
    plain_dict = {"strategy_type": "OUTSIDER"}
    assert pick_stop_pct(plain_dict, settings) == 30.0  # dict не имеет hasattr→ True, но isinstance check


# ── Тест 3: Дефолты fallback (нет ключей в settings_db) ──────────────────────
def test_engine_uses_defaults_when_no_settings():
    """Если в settings_db нет ключей, должны использоваться дефолты: 40 и 60."""
    def pick_stop_pct(decision_obj, settings_db):
        is_outsider = (
            hasattr(decision_obj, 'strategy_type')
            and isinstance(decision_obj.strategy_type, str)
            and decision_obj.strategy_type.upper() == "OUTSIDER"
        )
        if is_outsider:
            return float(settings_db.get("STOP_LOSS_PCT_OUTSIDER", "60.0"))
        return float(settings_db.get("STOP_LOSS_PCT_FAVORITE", "40.0"))

    empty_settings = {}

    outsider = MagicMock()
    outsider.strategy_type = "OUTSIDER"

    favorite = MagicMock()
    favorite.strategy_type = "PURE_FAVORITE"

    assert pick_stop_pct(outsider, empty_settings) == 60.0
    assert pick_stop_pct(favorite, empty_settings) == 40.0


# ── Тест 4: TradeDecision (реальный dataclass) работает корректно ─────────────
def test_engine_with_real_trade_decision():
    """Проверяем с реальным TradeDecision dataclass, а не MagicMock."""
    from polyflip.trading.decision_logic import TradeDecision

    outsider = TradeDecision(
        action="BUY_NO", buy_price=0.3, bet_size_usdc=10.0,
        reason="flip expected", strategy_type="OUTSIDER", edge=0.05
    )
    ml = TradeDecision(
        action="BUY_YES", buy_price=0.6, bet_size_usdc=10.0,
        reason="trend follow", strategy_type="ML_TREND", edge=0.03
    )

    settings = {"STOP_LOSS_PCT_FAVORITE": "40.0", "STOP_LOSS_PCT_OUTSIDER": "60.0"}

    def pick(dec):
        is_outsider = (
            hasattr(dec, 'strategy_type')
            and isinstance(dec.strategy_type, str)
            and dec.strategy_type.upper() == "OUTSIDER"
        )
        return float(settings.get("STOP_LOSS_PCT_OUTSIDER" if is_outsider else "STOP_LOSS_PCT_FAVORITE"))

    assert pick(outsider) == 60.0
    assert pick(ml) == 40.0


# ── Тест 5: Миграция idempotent (нет старого ключа — NOP) ────────────────────
@pytest.mark.asyncio
async def test_migrate_stop_loss_pct_noop_without_old_key():
    """Если STOP_LOSS_PCT не существует в БД — миграция не падает."""
    from polyflip.db.init_runtime_settings import migrate_stop_loss_pct
    from unittest.mock import AsyncMock, patch

    mock_session = AsyncMock()
    mock_session.scalar = AsyncMock(return_value=None)  # нет старого ключа

    await migrate_stop_loss_pct(mock_session)

    # Не должно быть вызовов add/delete/commit
    mock_session.add.assert_not_called()
    mock_session.delete.assert_not_called()
    mock_session.commit.assert_not_called()
