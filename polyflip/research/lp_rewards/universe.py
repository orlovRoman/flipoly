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
    seen_condition_ids = set()
    next_cursor: Optional[str] = None
    page = 0

    try:
        while page < max_pages:
            params: Dict[str, Any] = {"limit": page_size}
            if next_cursor:
                params["next_cursor"] = next_cursor

            resp = await client.get(CLOB_REWARDS_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

            items = data.get("data", [])
            for item in items:
                cid = item.get("condition_id")
                if cid and cid in seen_condition_ids:
                    continue
                if cid:
                    seen_condition_ids.add(cid)
                all_raw_markets.append(item)

            next_cursor = data.get("next_cursor")
            page += 1

            if not next_cursor or next_cursor == terminal_marker:
                break
        else:
            if next_cursor and next_cursor != terminal_marker:
                raise RuntimeError(
                    f"Pagination truncated: reached max_pages ({max_pages}) "
                    f"without reaching terminal marker '{terminal_marker}'."
                )
    finally:
        if should_close:
            await client.aclose()

    return all_raw_markets


CLOB_MARKET_INFO_URL = "https://clob.polymarket.com/clob-markets"
CLOB_MARKET_URL = "https://clob.polymarket.com/markets"


async def enrich_market_info(
    condition_id: str,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Fetch CLOB market details for a condition_id from official /clob-markets and /markets."""
    async with semaphore:
        combined_info: Dict[str, Any] = {}
        # 1. Fetch CLOB V2 market info (official source for OAS moas, fee fd.r, tokens t, neg_risk nr)
        try:
            resp_clob = await client.get(f"{CLOB_MARKET_INFO_URL}/{condition_id}", timeout=10.0)
            if resp_clob.status_code == 200:
                data_clob = resp_clob.json()
                if isinstance(data_clob, dict):
                    combined_info.update(data_clob)
        except Exception as e:
            logger.debug(f"Error fetching /clob-markets for {condition_id}: {e}")

        # 2. Fetch /markets for question title, description, market_slug
        try:
            resp_mkt = await client.get(f"{CLOB_MARKET_URL}/{condition_id}", timeout=10.0)
            if resp_mkt.status_code == 200:
                data_mkt = resp_mkt.json()
                if isinstance(data_mkt, dict):
                    for k, v in data_mkt.items():
                        if k not in combined_info or combined_info[k] is None:
                            combined_info[k] = v
        except Exception as e:
            logger.debug(f"Error fetching /markets for {condition_id}: {e}")

        return condition_id, combined_info if combined_info else None


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
    if clob_market_info:
        if "nr" in clob_market_info and clob_market_info["nr"] is not None:
            neg_risk = bool(clob_market_info["nr"])
        elif "neg_risk" in clob_market_info and clob_market_info["neg_risk"] is not None:
            neg_risk = bool(clob_market_info["neg_risk"])

    if exclude_neg_risk and neg_risk:
        return None

    # Extract tokens
    yes_token_id = None
    no_token_id = None
    if clob_market_info:
        if "t" in clob_market_info and isinstance(clob_market_info["t"], list):
            for t_entry in clob_market_info["t"]:
                t_id = str(t_entry.get("t", ""))
                outcome = str(t_entry.get("o", "")).lower()
                if outcome in ("yes", "up"):
                    yes_token_id = t_id
                elif outcome in ("no", "down"):
                    no_token_id = t_id
            if not yes_token_id and len(clob_market_info["t"]) >= 2:
                yes_token_id = str(clob_market_info["t"][0].get("t"))
                no_token_id = str(clob_market_info["t"][1].get("t"))
        elif "tokens" in clob_market_info and isinstance(clob_market_info["tokens"], list):
            toks = clob_market_info["tokens"]
            if len(toks) >= 2:
                for t_entry in toks:
                    t_id = str(t_entry.get("token_id", ""))
                    outcome = str(t_entry.get("outcome", "")).lower()
                    if outcome in ("yes", "up"):
                        yes_token_id = t_id
                    elif outcome in ("no", "down"):
                        no_token_id = t_id
                if not yes_token_id:
                    yes_token_id = str(toks[0].get("token_id"))
                    no_token_id = str(toks[1].get("token_id"))

    if not yes_token_id or not no_token_id:
        raw_tokens = raw_item.get("tokens", [])
        if len(raw_tokens) >= 2:
            yes_token_id = str(raw_tokens[0].get("token_id"))
            no_token_id = str(raw_tokens[1].get("token_id"))

    if not yes_token_id or not no_token_id:
        return None

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
        r_info = clob_market_info.get("r")
        if isinstance(r_info, dict) and "moas" in r_info and r_info["moas"] is not None:
            oas_val = r_info["moas"]
        else:
            oas_val = (
                clob_market_info.get("moas")
                or clob_market_info.get("minimum_order_age")
                or clob_market_info.get("order_age_seconds")
                or clob_market_info.get("seconds_delay")
                or clob_market_info.get("oas")
                or (clob_market_info.get("rewards", {}).get("order_age_seconds") if isinstance(clob_market_info.get("rewards"), dict) else None)
                or (clob_market_info.get("rewards", {}).get("min_order_age") if isinstance(clob_market_info.get("rewards"), dict) else None)
            )
    if oas_val is None:
        oas_val = (
            raw_item.get("moas")
            or raw_item.get("minimum_order_age")
            or raw_item.get("order_age_seconds")
            or raw_item.get("oas")
        )

    oas = Decimal(str(oas_val)) if oas_val is not None else Decimal("5.0")

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

    # Taker fee extraction from CLOB market info
    taker_fee_val = None
    if clob_market_info:
        fd = clob_market_info.get("fd")
        if isinstance(fd, dict) and "r" in fd and fd["r"] is not None:
            taker_fee_val = Decimal(str(fd["r"]))
        elif "taker_fee_bps" in clob_market_info:
            taker_fee_val = Decimal(str(clob_market_info["taker_fee_bps"])) / Decimal("10000.0")
        elif "taker_base_fee" in clob_market_info:
            taker_fee_val = Decimal(str(clob_market_info["taker_base_fee"])) / Decimal("100000.0")
        elif "fee_schedule" in clob_market_info and isinstance(clob_market_info["fee_schedule"], dict):
            fs = clob_market_info["fee_schedule"]
            taker_fee_val = fs.get("takerFee") or fs.get("taker_fee")
            if taker_fee_val is not None and "bps" in str(fs.get("type", "")).lower():
                taker_fee_val = Decimal(str(taker_fee_val)) / Decimal("10000.0")
        elif "feeSchedule" in clob_market_info and isinstance(clob_market_info["feeSchedule"], dict):
            fs = clob_market_info["feeSchedule"]
            taker_fee_val = fs.get("takerFee") or fs.get("taker_fee")
        elif "taker_fee" in clob_market_info:
            taker_fee_val = clob_market_info["taker_fee"]
        elif "taker_fee_rate" in clob_market_info:
            taker_fee_val = clob_market_info["taker_fee_rate"]

    if taker_fee_val is None:
        if "taker_fee_bps" in raw_item:
            taker_fee_val = Decimal(str(raw_item["taker_fee_bps"])) / Decimal("10000.0")
        else:
            fee_schedule = (
                raw_item.get("fee_schedule")
                or raw_item.get("feeSchedule")
                or {}
            )
            if isinstance(fee_schedule, dict):
                taker_fee_val = fee_schedule.get("takerFee") or fee_schedule.get("taker_fee")

    taker_fee_rate = Decimal(str(taker_fee_val)) if taker_fee_val is not None else Decimal("0.00")

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
    """Rank markets by multi-factor reward density incorporating depth, spread buffer, OAS, and fees."""
    if market_depths is None:
        market_depths = {}

    def score_market(m: MarketRewardConfig) -> Decimal:
        if m.condition_id in market_depths and market_depths[m.condition_id] > Decimal("0.0"):
            depth = market_depths[m.condition_id]
        else:
            depth = max(Decimal("1.0"), m.rewards_min_size * Decimal("4.0"))

        daily_rate = m.rewards_daily_rate
        fee_factor = max(Decimal("0.0"), Decimal("1.0") - m.taker_fee_rate)
        spread_factor = Decimal("1.0") + m.rewards_max_spread
        oas_penalty = Decimal("1.0") / (Decimal("1.0") + Decimal("0.1") * m.oas)

        return (daily_rate * fee_factor * spread_factor * oas_penalty) / depth

    sorted_markets = sorted(configs, key=score_market, reverse=True)
    active = sorted_markets[:top_n_active]
    reserve = sorted_markets[top_n_active : top_n_active + top_n_reserve]
    return active, reserve
