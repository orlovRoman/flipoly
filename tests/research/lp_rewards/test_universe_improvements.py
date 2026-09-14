from decimal import Decimal
import httpx
import pytest
import respx

from polyflip.research.lp_rewards.models import MarketRewardConfig
from polyflip.research.lp_rewards.universe import (
    CLOB_REWARDS_URL,
    fetch_all_rewards_markets,
    parse_market_reward_config,
    rank_markets_by_reward_density,
)


@pytest.mark.asyncio
@respx.mock
async def test_fetch_all_rewards_markets_pagination_and_deduplication():
    # Page 1 has cond_1 and cond_2
    respx.get(CLOB_REWARDS_URL).mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "data": [
                        {"condition_id": "cond_1", "name": "Market 1"},
                        {"condition_id": "cond_2", "name": "Market 2"},
                    ],
                    "next_cursor": "cursor_p2",
                },
            ),
            httpx.Response(
                200,
                json={
                    "data": [
                        {"condition_id": "cond_2", "name": "Market 2 Duplicate"},
                        {"condition_id": "cond_3", "name": "Market 3"},
                    ],
                    "next_cursor": "LTE=",
                },
            ),
        ]
    )

    markets = await fetch_all_rewards_markets(page_size=2, max_pages=10)
    assert len(markets) == 3
    cids = [m["condition_id"] for m in markets]
    assert cids == ["cond_1", "cond_2", "cond_3"]


@pytest.mark.asyncio
@respx.mock
async def test_fetch_all_rewards_markets_truncation_guard():
    # Never returns terminal marker "LTE="
    respx.get(CLOB_REWARDS_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [{"condition_id": "cond_inf", "name": "Infinite Market"}],
                "next_cursor": "keep_going_cursor",
            },
        )
    )

    with pytest.raises(RuntimeError, match="Pagination truncated"):
        await fetch_all_rewards_markets(max_pages=3)


def test_parse_market_reward_config_oas_and_taker_fees():
    raw_item = {
        "condition_id": "cond_real",
        "question": "Will X happen?",
        "total_daily_rate": "25.0",
        "rewards_max_spread": "0.04",
        "rewards_min_size": "20.0",
        "tokens": [{"token_id": "tok_yes_1"}, {"token_id": "tok_no_1"}],
    }
    clob_info = {
        "minimum_order_age": 10,
        "taker_fee_bps": 15,  # 15 bps = 0.0015
    }

    config = parse_market_reward_config(raw_item, clob_market_info=clob_info)
    assert config is not None
    assert config.condition_id == "cond_real"
    assert config.rewards_daily_rate == Decimal("25.0")
    assert config.rewards_max_spread == Decimal("0.04")
    assert config.rewards_min_size == Decimal("20.0")
    assert config.oas == Decimal("10.0")
    assert config.taker_fee_rate == Decimal("0.0015")
    assert config.yes_token_id == "tok_yes_1"
    assert config.no_token_id == "tok_no_1"


def test_parse_market_reward_config_fee_schedule_dict():
    raw_item = {
        "condition_id": "cond_fs",
        "question": "Will Y happen?",
        "total_daily_rate": "15.0",
        "rewards_max_spread": "0.05",
        "rewards_min_size": "10.0",
        "tokens": [{"token_id": "tok_yes_2"}, {"token_id": "tok_no_2"}],
    }
    clob_info = {
        "order_age_seconds": 3,
        "feeSchedule": {
            "takerFee": "0.002",
        },
    }

    config = parse_market_reward_config(raw_item, clob_market_info=clob_info)
    assert config is not None
    assert config.oas == Decimal("3.0")
    assert config.taker_fee_rate == Decimal("0.002")


def test_parse_market_reward_config_neg_risk_filter():
    raw_item = {
        "condition_id": "cond_neg_risk",
        "neg_risk": True,
        "tokens": [{"token_id": "tok_yes_3"}, {"token_id": "tok_no_3"}],
    }
    assert parse_market_reward_config(raw_item, exclude_neg_risk=True) is None
    assert parse_market_reward_config(raw_item, exclude_neg_risk=False) is not None


def test_rank_markets_by_reward_density():
    m_high_reward = MarketRewardConfig(
        condition_id="c_high",
        question="High Reward Market",
        rewards_daily_rate=Decimal("100.0"),
        rewards_max_spread=Decimal("0.05"),
        rewards_min_size=Decimal("10.0"),
        yes_token_id="tok_y1",
        no_token_id="tok_n1",
        oas=Decimal("2.0"),
        taker_fee_rate=Decimal("0.0"),
    )
    m_low_reward = MarketRewardConfig(
        condition_id="c_low",
        question="Low Reward Market",
        rewards_daily_rate=Decimal("1.0"),
        rewards_max_spread=Decimal("0.05"),
        rewards_min_size=Decimal("10.0"),
        yes_token_id="tok_y2",
        no_token_id="tok_n2",
        oas=Decimal("10.0"),
        taker_fee_rate=Decimal("0.01"),
    )

    active, reserve = rank_markets_by_reward_density(
        [m_low_reward, m_high_reward],
        market_depths={"c_high": Decimal("100.0"), "c_low": Decimal("100.0")},
        top_n_active=1,
        top_n_reserve=1,
    )

    assert len(active) == 1
    assert active[0].condition_id == "c_high"
    assert len(reserve) == 1
    assert reserve[0].condition_id == "c_low"
