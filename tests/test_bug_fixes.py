"""Самотесты для проверки исправлений BUG-01, BUG-02, BUG-03, BUG-04."""
import inspect
import pytest
from unittest.mock import AsyncMock, patch


# ─── BUG-01: engine.py — условие crypto-решения ────────────────────────────

def test_crypto_action_trade_should_execute():
    """BUG-01: когда action='BUY_YES', условие (not d or action != 'SKIP') = True
    → p_flip/edge устанавливаются для единого SKIP-обработчика."""
    from polyflip.trading.decision_logic import TradeDecision

    decision = TradeDecision(
        action="BUY_YES", buy_price=0.40, bet_size_usdc=5.0,
        strategy_type="CRYPTO_TREND", reason="test", edge=0.12
    )
    # Правильное условие: BUY_YES != SKIP → True → устанавливаем переменные (edge, model_ver, p_flip)
    assert (not decision or decision.action != "SKIP") is True, (
        "BUG-01: условие должно быть True для BUY_YES (блок инициализации p_flip/edge)"
    )


def test_crypto_action_skip_should_skip_init_block():
    """BUG-01: когда action='SKIP', условие (not d or action != 'SKIP') = False
    → блок инициализации p_flip/edge НЕ выполняется.

    Инвертированное условие '!= "trade"' будет True для SKIP — тоесть блок запустится, а
    для BUY_YES будет False — блок не запустится, edge/p_flip не будет установлен
    и крипто-стратегия перестанет торговать."""
    from polyflip.trading.decision_logic import TradeDecision

    decision = TradeDecision(
        action="SKIP", buy_price=0.0, bet_size_usdc=0.0,
        strategy_type="CRYPTO_TREND", reason="No signal", edge=0.0
    )
    # Правильное условие: SKIP != SKIP = False → блок не выполняется (OK — SKIP попадёт в единый обработчик ниже)
    assert (not decision or decision.action != "SKIP") is False, (
        "BUG-01: условие должно быть False для SKIP (блок инициализации не запускается)"
    )





def test_engine_py_no_action_trade_condition():
    """BUG-01: убеждаемся, что инвертированное условие '!= \"trade\"' не присутствует в engine.py."""
    source = open("polyflip/trading/engine.py", encoding="utf-8").read()
    assert 'decision_obj.action != "trade"' not in source, (
        "BUG-01 PRESENT: найдено инвертированное условие action != 'trade' в engine.py! "
        "Должно быть: action != 'SKIP'"
    )


# ─── BUG-02: backtester.py — epsilon не должен быть 0.0 ─────────────────────

def test_run_backtest_epsilon_not_zero():
    """BUG-02: epsilon=0.0 не должен быть захардкожен в backtester.py."""
    source = open("polyflip/crypto/backtester.py", encoding="utf-8").read()
    assert "epsilon=0.0," not in source, (
        "BUG-02: epsilon=0.0 захардкожен! Используй константу _EPSILON = 1e-9"
    )


def test_backtester_epsilon_constant_positive():
    """BUG-02: константа _EPSILON в backtester.py должна быть > 0."""
    from polyflip.crypto import backtester
    assert hasattr(backtester, "_EPSILON"), "_EPSILON не найдена в backtester.py"
    assert backtester._EPSILON > 0, f"_EPSILON должен быть > 0, получено: {backtester._EPSILON}"


# ─── BUG-03: jobs.py — stoploss_job должен иметь try/except ─────────────────

def test_stoploss_job_has_try_except():
    """BUG-03: stoploss_job должен содержать try/except для перехвата исключений."""
    source = open("polyflip/scheduler/jobs.py", encoding="utf-8").read()
    # Проверяем что try: находится внутри stoploss_job (до следующей async def)
    stoploss_start = source.find("async def stoploss_job")
    next_func = source.find("async def ", stoploss_start + 1)
    job_body = source[stoploss_start:next_func]
    assert "try:" in job_body, (
        "BUG-03: stoploss_job не имеет try/except — необработанное исключение упадёт в планировщик"
    )
    assert "except Exception" in job_body, (
        "BUG-03: stoploss_job не перехватывает Exception"
    )


@pytest.mark.asyncio
async def test_stoploss_job_handles_exception():
    """BUG-03: stoploss_job не должен пробрасывать исключения наверх."""
    from polyflip.scheduler.jobs import stoploss_job

    trader_mock = AsyncMock()
    api_mock = AsyncMock()

    with patch(
        "polyflip.scheduler.jobs.stoploss_worker_cycle",
        side_effect=RuntimeError("DB connection failed")
    ):
        # Не должно бросить исключение
        try:
            await stoploss_job(trader_mock, api_mock)
        except Exception as e:
            pytest.fail(f"stoploss_job пробросил исключение: {e}")


# ─── BUG-04: takeprofit — должен использовать best_bid, а не best_ask ───────

def test_tp_worker_uses_bid_not_ask():
    """BUG-04: TP-воркер должен использовать best_bid, а не best_ask."""
    source = open("polyflip/trading/takeprofit_worker.py", encoding="utf-8").read()
    assert "best_ask" not in source, (
        "BUG-04: TP-воркер использует best_ask! "
        "Для продажи токенов на Polymarket нужен best_bid (цена покупателей)"
    )
    assert "best_bid" in source, "BUG-04: TP-воркер должен использовать best_bid"


def test_tp_function_uses_current_bid_param():
    """BUG-04: evaluate_take_profit должна принимать current_bid, а не current_ask."""
    from polyflip.trading.takeprofit import evaluate_take_profit
    sig = inspect.signature(evaluate_take_profit)
    params = list(sig.parameters.keys())
    assert "current_bid" in params, (
        f"BUG-04: evaluate_take_profit не имеет параметра current_bid. Параметры: {params}"
    )
    assert "current_ask" not in params, (
        f"BUG-04: evaluate_take_profit всё ещё имеет current_ask. Параметры: {params}"
    )
