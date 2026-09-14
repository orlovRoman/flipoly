from decimal import Decimal
import pytest

from polyflip.research.lp_rewards.scoring import normalize_sample_scores


def test_reward_normalization_per_sample():
    # Sample t=1: Maker A has 10, Maker B has 30 -> Total 40
    sample_1 = {
        "maker_a": Decimal("10.0"),
        "maker_b": Decimal("30.0"),
    }
    norm_1 = normalize_sample_scores(sample_1)
    assert norm_1["maker_a"] == Decimal("0.25")
    assert norm_1["maker_b"] == Decimal("0.75")
    assert sum(norm_1.values()) == Decimal("1.0")

    # Sample t=2: Maker A has 20, Maker B has 20 -> Total 40
    sample_2 = {
        "maker_a": Decimal("20.0"),
        "maker_b": Decimal("20.0"),
    }
    norm_2 = normalize_sample_scores(sample_2)
    assert norm_2["maker_a"] == Decimal("0.50")
    assert norm_2["maker_b"] == Decimal("0.50")

    # Sum across samples: Q_epoch
    q_epoch_a = norm_1["maker_a"] + norm_2["maker_a"]
    q_epoch_b = norm_1["maker_b"] + norm_2["maker_b"]
    assert q_epoch_a == Decimal("0.75")
    assert q_epoch_b == Decimal("1.25")


def test_reward_normalization_zero_denominator():
    sample_zero = {
        "maker_a": Decimal("0.0"),
        "maker_b": Decimal("0.0"),
    }
    norm_zero = normalize_sample_scores(sample_zero)
    assert norm_zero["maker_a"] == Decimal("0.0")
    assert norm_zero["maker_b"] == Decimal("0.0")
