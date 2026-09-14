from datetime import datetime, timezone, timedelta
from decimal import Decimal
import pytest

from polyflip.research.lp_rewards.models import MarketRewardConfig
from polyflip.research.lp_rewards.quoting_fsm import MarketQuotingFSM


def is_market_eligible_for_quoting(
    market: MarketRewardConfig,
    current_utc_time: datetime,
    expiry_guard_buffer_sec: float = 1200.0,  # 20 minutes
) -> bool:
    """Blocks quoting if time until end_date_iso is less than expiry_guard_buffer_sec."""
    if not market.end_date_iso:
        return True

    end_dt = datetime.fromisoformat(market.end_date_iso.replace("Z", "+00:00"))
    time_left = (end_dt - current_utc_time).total_seconds()
    return time_left >= expiry_guard_buffer_sec


def test_market_expiry_guard_blocks_near_expiry():
    now = datetime(2026, 9, 14, 12, 0, 0, tzinfo=timezone.utc)

    # Market expiring in 10 minutes (less than 20 min guard)
    near_end = now + timedelta(minutes=10)
    market_expiring_soon = MarketRewardConfig(
        condition_id="c_soon",
        question="Expiring soon",
        rewards_daily_rate=Decimal("10.0"),
        rewards_max_spread=Decimal("0.05"),
        rewards_min_size=Decimal("10.0"),
        yes_token_id="y1",
        no_token_id="n1",
        end_date_iso=near_end.isoformat(),
    )
    assert is_market_eligible_for_quoting(market_expiring_soon, now, 1200.0) is False

    # Market expiring in 2 hours (well above guard)
    far_end = now + timedelta(hours=2)
    market_far = MarketRewardConfig(
        condition_id="c_far",
        question="Expiring later",
        rewards_daily_rate=Decimal("10.0"),
        rewards_max_spread=Decimal("0.05"),
        rewards_min_size=Decimal("10.0"),
        yes_token_id="y2",
        no_token_id="n2",
        end_date_iso=far_end.isoformat(),
    )
    assert is_market_eligible_for_quoting(market_far, now, 1200.0) is True
