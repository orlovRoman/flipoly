from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Tuple
from .models import OrderbookLevel, OrderbookSnapshot, SampleScore

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
