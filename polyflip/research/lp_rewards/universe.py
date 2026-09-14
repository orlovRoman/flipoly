import asyncio
from decimal import Decimal, getcontext
import logging
from typing import Any, Dict, List, Optional, Tuple
import httpx
from .models import MarketRewardConfig

getcontext().prec = 28
logger = logging.getLogger(__name__)

CLOB_REWARDS_URL = "https://clob.polymarket.com/rewards/markets/current"
CLOB_MARKET_URL = "https://clob.polymarket.com/markets"


async def fetch_all_rewards_markets(
    client: Optional[httpx.AsyncClient] = None,
    terminal_marker: str = "LTE=",
    page_size: int = 500,
    max_pages: int = 50,
) -> List[Dict[str, Any]]:
    """Fetch all rewards markets with full pagination until next_cursor is LTE= or empty."""
    should_close = False
    if client is None:
        client = httpx.AsyncClient(timeout=30.0)
        should_close = True

    all_raw_markets: List[Dict[str, Any]] = []
    next_cursor: Optional[str] = None
    page = 0

    try:
        while page < max_pages:
            params = {}
            if next_cursor:
                params["next_cursor"] = next_cursor

            resp = await client.get(CLOB_REWARDS_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

            items = data.get("data", [])
            all_raw_markets.extend(items)

            next_cursor = data.get("next_cursor")
            page += 1

            if not next_cursor or next_cursor == terminal_marker:
                break
    finally:
        if should_close:
            await client.aclose()

    return all_raw_markets


async def enrich_market_info(
    condition_id: str,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Fetch CLOB market details for a condition_id."""
    async with semaphore:
        try:
            resp = await client.get(f"{CLOB_MARKET_URL}/{condition_id}", timeout=10.0)
            if resp.status_code == 200:
                return condition_id, resp.json()
        except Exception as e:
            logger.debug(f"Error fetching info for {condition_id}: {e}")
        return condition_id, None


def parse_market_reward_config(
    raw_item: Dict[str, Any],
    clob_market_info: Optional[Dict[str, Any]] = None,
    exclude_neg_risk: bool = True,
) -> Optional[MarketRewardConfig]:
    """Parse raw reward market dictionary into MarketRewardConfig.
    
    Excludes markets with neg_risk=True if exclude_neg_risk is True.
    Extracts tokens, rewards rate, max spread, min size, and oas.
    """
    condition_id = raw_item.get("condition_id")
    if not condition_id:
        return None

    # Check neg_risk
    neg_risk = raw_item.get("neg_risk", False)
    if clob_market_info and "neg_risk" in clob_market_info:
        neg_risk = clob_market_info.get("neg_risk", False)

    if exclude_neg_risk and neg_risk:
        return None

    # Tokens
    tokens = raw_item.get("tokens", [])
    if len(tokens) < 2 and clob_market_info:
        tokens = clob_market_info.get("tokens", [])

    if len(tokens) < 2:
        return None

    yes_token_id = str(tokens[0].get("token_id"))
    no_token_id = str(tokens[1].get("token_id"))

    # Rewards parameters
    daily_rate_val = (
        raw_item.get("total_daily_rate")
        or raw_item.get("native_daily_rate")
        or raw_item.get("rewards_daily_rate")
        or raw_item.get("daily_rate")
        or "0"
    )
    daily_rate = Decimal(str(daily_rate_val))

    max_spread_val = raw_item.get("rewards_max_spread") or raw_item.get("max_spread") or "0.05"
    max_spread = Decimal(str(max_spread_val))
    # If spread is in percentage points (e.g. 5.5 means 5.5% / 5.5 cents)
    if max_spread > Decimal("1.0"):
        max_spread = max_spread / Decimal("100.0")

    min_size_val = raw_item.get("rewards_min_size") or raw_item.get("min_size") or "10"
    min_size = Decimal(str(min_size_val))

    # OAS (Order Age Seconds)
    oas_val = None
    if clob_market_info:
        oas_val = (
            clob_market_info.get("order_age_seconds")
            or clob_market_info.get("seconds_delay")
            or clob_market_info.get("oas")
        )
    if oas_val is None:
        oas_val = raw_item.get("order_age_seconds", raw_item.get("oas"))

    if oas_val is not None:
        oas = Decimal(str(oas_val))
    else:
        oas = Decimal("5.0")

    question = (
        (clob_market_info.get("question") if clob_market_info else None)
        or raw_item.get("question")
        or raw_item.get("market_slug")
        or condition_id
    )

    end_date_iso = (
        (clob_market_info.get("end_date_iso") if clob_market_info else None)
        or raw_item.get("end_date_iso")
    )

    fee_schedule = (
        (clob_market_info.get("feeSchedule") if clob_market_info else None)
        or raw_item.get("feeSchedule")
        or {}
    )
    taker_fee_rate = Decimal(str(fee_schedule.get("takerFee", "0.0")))

    return MarketRewardConfig(
        condition_id=condition_id,
        question=question,
        rewards_daily_rate=daily_rate,
        rewards_max_spread=max_spread,
        rewards_min_size=min_size,
        oas=oas,
        neg_risk=neg_risk,
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        end_date_iso=end_date_iso,
        taker_fee_rate=taker_fee_rate,
    )


def rank_markets_by_reward_density(
    configs: List[MarketRewardConfig],
    market_depths: Optional[Dict[str, Decimal]] = None,
    capital_per_market: Decimal = Decimal("25.0"),
    top_n_active: int = 20,
    top_n_reserve: int = 10,
) -> Tuple[List[MarketRewardConfig], List[MarketRewardConfig]]:
    """Rank markets by reward density (daily reward pool relative to market depth / capital)."""
    if market_depths is None:
        market_depths = {}

    def score_market(m: MarketRewardConfig) -> Decimal:
        depth = market_depths.get(m.condition_id, Decimal("500.0"))
        if depth <= Decimal("0.0"):
            depth = Decimal("1.0")
        return m.rewards_daily_rate / depth

    sorted_markets = sorted(configs, key=score_market, reverse=True)
    active = sorted_markets[:top_n_active]
    reserve = sorted_markets[top_n_active : top_n_active + top_n_reserve]
    return active, reserve
