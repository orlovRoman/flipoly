from decimal import Decimal
import pytest

from polyflip.research.lp_rewards.universe import parse_market_reward_config


def test_neg_risk_strictly_excluded():
    raw_neg_risk_market = {
        "condition_id": "0xabc123",
        "question": "Who will win the presidential election?",
        "neg_risk": True,
        "tokens": [{"token_id": "tok1"}, {"token_id": "tok2"}],
        "rewards_daily_rate": "100.0",
        "rewards_max_spread": "0.05",
        "rewards_min_size": "10",
        "order_age_seconds": "5",
    }

    # With exclude_neg_risk=True -> Must return None
    config = parse_market_reward_config(raw_neg_risk_market, exclude_neg_risk=True)
    assert config is None

    # Normal non-neg-risk market
    raw_binary_market = {
        "condition_id": "0xdef456",
        "question": "Will BTC reach $100k by end of year?",
        "neg_risk": False,
        "tokens": [{"token_id": "tok_yes"}, {"token_id": "tok_no"}],
        "rewards_daily_rate": "50.0",
        "rewards_max_spread": "0.04",
        "rewards_min_size": "20",
        "order_age_seconds": "5",
    }
    config_binary = parse_market_reward_config(raw_binary_market, exclude_neg_risk=True)
    assert config_binary is not None
    assert config_binary.condition_id == "0xdef456"
    assert config_binary.neg_risk is False
    assert config_binary.oas == Decimal("5")
