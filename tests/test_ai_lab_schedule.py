from polyflip.ai_lab.schedule import evaluate_schedule_start


def test_schedule_is_disabled_by_default():
    decision = evaluate_schedule_start(
        enabled=False,
        mode="RESEARCH",
        trading_enabled=False,
        active_runs=0,
        runs_today=0,
    )
    assert decision.allowed is False
    assert decision.reason == "schedule_disabled"


def test_schedule_only_creates_research_runs_and_respects_limits():
    allowed = evaluate_schedule_start(
        enabled=True,
        mode="RESEARCH",
        trading_enabled=False,
        active_runs=0,
        runs_today=0,
    )
    assert allowed.allowed is True

    assert evaluate_schedule_start(
        enabled=True, mode="STANDARD", trading_enabled=False, active_runs=0, runs_today=0
    ).reason == "schedule_requires_research_mode"
    assert evaluate_schedule_start(
        enabled=True, mode="RESEARCH", trading_enabled=True, active_runs=0, runs_today=0
    ).reason == "research_schedule_blocked_while_trading_enabled"
    assert evaluate_schedule_start(
        enabled=True, mode="RESEARCH", trading_enabled=False, active_runs=1, runs_today=0
    ).reason == "active_run_limit_reached"
    assert evaluate_schedule_start(
        enabled=True, mode="RESEARCH", trading_enabled=False, active_runs=0, runs_today=1
    ).reason == "daily_run_limit_reached"
