from decimal import Decimal, getcontext
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel
from .models import OrderbookLevel, OrderbookSnapshot, OrderSide, SampleScore, VirtualOrder

getcontext().prec = 28


def calculate_cutoff_midpoint(
    yes_bids: List[OrderbookLevel],
    yes_asks: List[OrderbookLevel],
    min_size: Decimal,
) -> Optional[Decimal]:
    """Calculate size-cutoff adjusted midpoint p_mid*.
    
    Filters out levels with size < min_size.
    Returns (best_bid + best_ask) / 2 if both sides exist.
    If either side lacks qualifying levels, returns None (MID_UNCERTAIN).
    """
    valid_bids = [b for b in yes_bids if b.size >= min_size]
    valid_asks = [a for a in yes_asks if a.size >= min_size]

    if not valid_bids or not valid_asks:
        return None

    best_bid = max(b.price for b in valid_bids)
    best_ask = min(a.price for a in valid_asks)

    if best_bid >= best_ask:
        # Crossed book in cutoff levels -> uncertain
        return None

    return (best_bid + best_ask) / Decimal("2.0")


def calculate_level_score(
    price: Decimal,
    size: Decimal,
    midpoint: Decimal,
    max_spread: Decimal,
    min_size: Decimal,
    multiplier: Decimal = Decimal("1.0"),
    is_no_token: bool = False,
) -> Decimal:
    """Calculate positional score S(v, d) * size for a single order level.
    
    S(v, d) = ((v - d) / v)^2 * b
    d = |price - midpoint| for YES
    d = |price - (1 - midpoint)| for NO
    Only eligible if size >= min_size and d <= max_spread.
    """
    if size < min_size or max_spread <= Decimal("0.0"):
        return Decimal("0.0")

    target_mid = (Decimal("1.0") - midpoint) if is_no_token else midpoint
    d = abs(price - target_mid)

    if d > max_spread:
        return Decimal("0.0")

    # ((v - d) / v)^2 * b * size
    spread_factor = (max_spread - d) / max_spread
    score_density = (spread_factor * spread_factor) * multiplier
    return size * score_density


def calculate_q_components(
    yes_bids: List[OrderbookLevel],
    yes_asks: List[OrderbookLevel],
    no_bids: List[OrderbookLevel],
    no_asks: List[OrderbookLevel],
    midpoint: Decimal,
    max_spread: Decimal,
    min_size: Decimal,
    multiplier: Decimal = Decimal("1.0"),
) -> Tuple[Decimal, Decimal, Decimal]:
    """Calculate cross-complementary Q_one, Q_two and resulting Q_min.
    
    Q_one = bids(YES) + asks(NO)
    Q_two = asks(YES) + bids(NO)
    
    If midpoint in [0.10, 0.90]:
        Q_min = max(min(Q_one, Q_two), max(Q_one / 3, Q_two / 3))
    Else:
        Q_min = min(Q_one, Q_two)
    """
    score_yes_bids = sum(
        (calculate_level_score(b.price, b.size, midpoint, max_spread, min_size, multiplier, is_no_token=False) for b in yes_bids),
        Decimal("0.0"),
    )
    score_yes_asks = sum(
        (calculate_level_score(a.price, a.size, midpoint, max_spread, min_size, multiplier, is_no_token=False) for a in yes_asks),
        Decimal("0.0"),
    )
    score_no_bids = sum(
        (calculate_level_score(b.price, b.size, midpoint, max_spread, min_size, multiplier, is_no_token=True) for b in no_bids),
        Decimal("0.0"),
    )
    score_no_asks = sum(
        (calculate_level_score(a.price, a.size, midpoint, max_spread, min_size, multiplier, is_no_token=True) for a in no_asks),
        Decimal("0.0"),
    )

    q_one = score_yes_bids + score_no_asks
    q_two = score_yes_asks + score_no_bids

    low_bound = Decimal("0.10")
    high_bound = Decimal("0.90")

    if low_bound <= midpoint <= high_bound:
        # In 10-90c range, one-sided orders can qualify up to 1/3 of max side
        q_min = max(min(q_one, q_two), max(q_one / Decimal("3.0"), q_two / Decimal("3.0")))
    else:
        # Outside 10-90c, strict two-sided required
        q_min = min(q_one, q_two)

    return q_one, q_two, q_min


def calculate_sample_scores(
    condition_id: str,
    timestamp_ns: int,
    yes_bids: List[OrderbookLevel],
    yes_asks: List[OrderbookLevel],
    no_bids: List[OrderbookLevel],
    no_asks: List[OrderbookLevel],
    max_spread: Decimal,
    min_size: Decimal,
    multiplier: Decimal = Decimal("1.0"),
    is_uncertain: bool = False,
) -> SampleScore:
    """Calculate sample score for a market at a single minute sample."""
    if is_uncertain:
        return SampleScore(
            condition_id=condition_id,
            timestamp_ns=timestamp_ns,
            p_mid_star=None,
            q_one=Decimal("0.0"),
            q_two=Decimal("0.0"),
            q_min=Decimal("0.0"),
            status="BOOK_UNCERTAIN",
        )

    p_mid_star = calculate_cutoff_midpoint(yes_bids, yes_asks, min_size)
    if p_mid_star is None:
        return SampleScore(
            condition_id=condition_id,
            timestamp_ns=timestamp_ns,
            p_mid_star=None,
            q_one=Decimal("0.0"),
            q_two=Decimal("0.0"),
            q_min=Decimal("0.0"),
            status="MID_UNCERTAIN",
        )

    q_one, q_two, q_min = calculate_q_components(
        yes_bids, yes_asks, no_bids, no_asks, p_mid_star, max_spread, min_size, multiplier
    )

    return SampleScore(
        condition_id=condition_id,
        timestamp_ns=timestamp_ns,
        p_mid_star=p_mid_star,
        q_one=q_one,
        q_two=q_two,
        q_min=q_min,
        status="VALID",
    )


def normalize_sample_scores(
    market_q_min: Dict[str, Decimal]  # participant_id -> q_min for this sample
) -> Dict[str, Decimal]:
    """Calculate per-sample normalized score Q_normal(n, t) = Q_min(n, t) / sum(Q_min(m, t))."""
    total_q_min = sum(market_q_min.values(), Decimal("0.0"))
    if total_q_min <= Decimal("0.0"):
        return {p_id: Decimal("0.0") for p_id in market_q_min}

    return {p_id: val / total_q_min for p_id, val in market_q_min.items()}


def calculate_daily_reward(
    q_epoch: Decimal,
    total_samples: int,
    daily_reward_pool: Decimal,
    dust_threshold_usdc: Decimal = Decimal("1.00"),
) -> Decimal:
    """Calculate daily reward from cumulative normalized score:
    Reward = Daily_Pool * (Q_epoch / total_samples)
    Applies $1.00 dust cutoff filter.
    """
    if total_samples <= 0:
        return Decimal("0.0")

    fraction = q_epoch / Decimal(str(total_samples))
    raw_reward = daily_reward_pool * fraction

    if raw_reward < dust_threshold_usdc:
        return Decimal("0.0")

    return raw_reward


class LPRewardShareEstimate(BaseModel):
    condition_id: str
    timestamp_ns: int
    p_mid_star: Optional[Decimal] = None
    q_own: Decimal = Decimal("0.0")
    q_competitor_min: Decimal = Decimal("0.0")
    q_competitor_max: Decimal = Decimal("0.0")
    q_competitor_expected: Decimal = Decimal("0.0")
    share_min: Decimal = Decimal("0.0")
    share_max: Decimal = Decimal("0.0")
    share_expected: Decimal = Decimal("0.0")
    status: str = "VALID"  # VALID, MID_UNCERTAIN, BOOK_UNCERTAIN


def calculate_competitor_and_own_scores(
    condition_id: str,
    timestamp_ns: int,
    public_yes_bids: List[OrderbookLevel],
    public_yes_asks: List[OrderbookLevel],
    public_no_bids: List[OrderbookLevel],
    public_no_asks: List[OrderbookLevel],
    our_orders: List[VirtualOrder],
    max_spread: Decimal,
    min_size: Decimal,
    multiplier: Decimal = Decimal("1.0"),
    yes_token_id: Optional[str] = None,
    no_token_id: Optional[str] = None,
    is_uncertain: bool = False,
) -> LPRewardShareEstimate:
    """Calculate our own qualifying score vs competitor denominator range and resulting reward shares.

    Competitor range:
    - q_competitor_min: strict two-sided qualifying competitor liquidity (lower bound)
    - q_competitor_max: upper bound of competitor liquidity
    - q_competitor_expected: standard Q_min from public book
    """
    if is_uncertain:
        return LPRewardShareEstimate(
            condition_id=condition_id,
            timestamp_ns=timestamp_ns,
            p_mid_star=None,
            status="BOOK_UNCERTAIN",
        )

    p_mid_star = calculate_cutoff_midpoint(public_yes_bids, public_yes_asks, min_size)
    if p_mid_star is None:
        return LPRewardShareEstimate(
            condition_id=condition_id,
            timestamp_ns=timestamp_ns,
            p_mid_star=None,
            status="MID_UNCERTAIN",
        )

    # 1. Competitor Q components from public book
    comp_q1, comp_q2, comp_q_expected = calculate_q_components(
        public_yes_bids, public_yes_asks, public_no_bids, public_no_asks,
        p_mid_star, max_spread, min_size, multiplier
    )
    # Competitor bounds: strict 2-sided (min) vs max (sum or one-sided max)
    comp_q_min = min(comp_q1, comp_q2)
    comp_q_max = max(comp_q_expected, max(comp_q1, comp_q2))

    # 2. Our own orders: separate by token and side
    def is_yes(asset_id: str) -> bool:
        if yes_token_id:
            return asset_id == yes_token_id
        return "yes" in asset_id.lower() or asset_id.endswith("_YES")

    def is_no(asset_id: str) -> bool:
        if no_token_id:
            return asset_id == no_token_id
        return "no" in asset_id.lower() or asset_id.endswith("_NO")

    our_yes_bids = [
        OrderbookLevel(price=o.price, size=o.size - o.filled_size)
        for o in our_orders if o.side == OrderSide.BUY and is_yes(o.asset_id)
    ]
    our_yes_asks = [
        OrderbookLevel(price=o.price, size=o.size - o.filled_size)
        for o in our_orders if o.side == OrderSide.SELL and is_yes(o.asset_id)
    ]
    our_no_bids = [
        OrderbookLevel(price=o.price, size=o.size - o.filled_size)
        for o in our_orders if o.side == OrderSide.BUY and is_no(o.asset_id)
    ]
    our_no_asks = [
        OrderbookLevel(price=o.price, size=o.size - o.filled_size)
        for o in our_orders if o.side == OrderSide.SELL and is_no(o.asset_id)
    ]

    our_q1, our_q2, q_own = calculate_q_components(
        our_yes_bids, our_yes_asks, our_no_bids, our_no_asks,
        p_mid_star, max_spread, min_size, multiplier
    )

    # 3. Denominators (our liquidity + competitor liquidity)
    d_max = comp_q_max + q_own
    d_min = comp_q_min + q_own
    d_expected = comp_q_expected + q_own

    share_min = (q_own / d_max) if d_max > Decimal("0.0") else Decimal("0.0")
    share_max = (q_own / d_min) if d_min > Decimal("0.0") else Decimal("0.0")
    share_expected = (q_own / d_expected) if d_expected > Decimal("0.0") else Decimal("0.0")

    return LPRewardShareEstimate(
        condition_id=condition_id,
        timestamp_ns=timestamp_ns,
        p_mid_star=p_mid_star,
        q_own=q_own,
        q_competitor_min=comp_q_min,
        q_competitor_max=comp_q_max,
        q_competitor_expected=comp_q_expected,
        share_min=min(Decimal("1.0"), share_min),
        share_max=min(Decimal("1.0"), share_max),
        share_expected=min(Decimal("1.0"), share_expected),
        status="VALID",
    )


def simulate_minute_monte_carlo(
    snapshots: List[Dict[str, Any]],
    our_orders: List[VirtualOrder],
    daily_reward_pool: Decimal,
    n_samples: int = 1440,
    seeds: Tuple[int, ...] = (42, 123, 999),
    max_spread: Decimal = Decimal("0.05"),
    min_size: Decimal = Decimal("10.0"),
    dust_threshold_usdc: Decimal = Decimal("1.00"),
) -> Dict[str, Any]:
    """Simulate daily LP rewards using minute Monte Carlo sampling across seeds."""
    if not snapshots:
        return {
            "mean_daily_reward": Decimal("0.0"),
            "ci_lower_95": Decimal("0.0"),
            "ci_upper_95": Decimal("0.0"),
            "min_reward": Decimal("0.0"),
            "max_reward": Decimal("0.0"),
            "simulations": [],
        }

    sim_rewards: List[Decimal] = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        sample_indices = rng.integers(0, len(snapshots), size=n_samples)

        sample_shares = []
        for idx in sample_indices:
            s = snapshots[idx]
            est = calculate_competitor_and_own_scores(
                condition_id=s.get("condition_id", "c_sim"),
                timestamp_ns=s.get("timestamp_ns", 0),
                public_yes_bids=s.get("yes_bids", []),
                public_yes_asks=s.get("yes_asks", []),
                public_no_bids=s.get("no_bids", []),
                public_no_asks=s.get("no_asks", []),
                our_orders=our_orders,
                max_spread=max_spread,
                min_size=min_size,
                multiplier=s.get("multiplier", Decimal("1.0")),
                yes_token_id=s.get("yes_token_id"),
                no_token_id=s.get("no_token_id"),
                is_uncertain=s.get("is_uncertain", False),
            )
            sample_shares.append(est.share_expected)

        mean_share = sum(sample_shares, Decimal("0.0")) / Decimal(str(n_samples))
        sim_reward = daily_reward_pool * mean_share
        if sim_reward < dust_threshold_usdc:
            sim_reward = Decimal("0.0")
        sim_rewards.append(sim_reward)

    arr = np.array([float(r) for r in sim_rewards])
    mean_val = Decimal(str(round(float(np.mean(arr)), 4)))
    ci_lower = Decimal(str(round(float(np.percentile(arr, 2.5)), 4)))
    ci_upper = Decimal(str(round(float(np.percentile(arr, 97.5)), 4)))
    min_val = Decimal(str(round(float(np.min(arr)), 4)))
    max_val = Decimal(str(round(float(np.max(arr)), 4)))

    return {
        "mean_daily_reward": mean_val,
        "ci_lower_95": ci_lower,
        "ci_upper_95": ci_upper,
        "min_reward": min_val,
        "max_reward": max_val,
        "simulations": [{"seed": s, "reward": r} for s, r in zip(seeds, sim_rewards)],
    }
