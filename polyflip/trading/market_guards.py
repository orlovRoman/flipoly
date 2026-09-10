import dataclasses
from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timedelta
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import structlog
from polyflip.db.models import LiveMarket, TradeHistory
from polyflip.trading.trading_config import TradingConfig

try:
    from polyflip.constants import TRADE_CHECK_LIMIT
except ImportError:
    TRADE_CHECK_LIMIT = 5

logger = structlog.get_logger(__name__)


@dataclass
class GuardResult:
    passed: bool
    skip_reason: Optional[str]
    existing_skipped: Optional[TradeHistory]


async def check_market_guards(
    db_session: AsyncSession,
    market: LiveMarket,
    cfg: TradingConfig,
    asset_mode: str,
    time_left_sec: float,
    start_time: datetime,
    is_outsider: Optional[bool] = None,
) -> GuardResult:
    """
    Выполняет все предварительные проверки рынка (guards) перед принятием решения.
    """
    is_ct_mode = (
        str(asset_mode).lower() in ("ct_outsider", "ct")
        or str(getattr(cfg, "trading_mode", "")).lower() in ("ct_outsider", "ct")
    )

    # 1. СНАЧАЛА ищем existing_skipped запись (нужна во всех последующих return)
    if is_ct_mode:
        skipped_check = select(TradeHistory).where(
            TradeHistory.market_id == market.market_id,
            TradeHistory.status == "SKIPPED",
            (TradeHistory.strategy_name == "BTC_CT_T5_V1") | (TradeHistory.strategy_type == "CT_OUTSIDER"),
        ).limit(1)
    else:
        skipped_check = select(TradeHistory).where(
            TradeHistory.market_id == market.market_id,
            TradeHistory.status == "SKIPPED"
        ).limit(1)
    skipped_res = await db_session.execute(skipped_check)
    existing_skipped = skipped_res.scalar_one_or_none()

    # A CT skip is the canonical decision for this market.  Check it before
    # the time-window guard so later scheduler heartbeats cannot overwrite its
    # reason with the generic ``Outside time window`` message.  This matters
    # after the first decision: the market naturally leaves the 210--300 s
    # window while the UI still polls it.
    if is_ct_mode and existing_skipped is not None:
        return GuardResult(
            passed=False,
            skip_reason="guard: Decision already recorded in window (SKIP)",
            existing_skipped=existing_skipped,
        )

    if time_left_sec <= 0:
        return GuardResult(passed=False, skip_reason="guard: Time left <= 0", existing_skipped=existing_skipped)
        
    if is_ct_mode:
        # Pinned CT decision window: 210 to 300 seconds (3.5 to 5.0 minutes)
        if not (210.0 <= time_left_sec <= 300.0):
            return GuardResult(passed=False, skip_reason="guard: Outside time window", existing_skipped=existing_skipped)
    else:
        global_min_sec = min(cfg.favor_min_time_left, cfg.outs_min_time_left)
        global_max_sec = max(cfg.favor_max_time_left, cfg.outs_max_time_left)
        if not (global_min_sec <= time_left_sec <= global_max_sec):
            return GuardResult(passed=False, skip_reason="guard: Outside time window", existing_skipped=existing_skipped)
            
    # 2. Проверяем дубликаты сделок — existing_skipped уже доступен
    # Item 19: CT and foreign strategies are isolated
    if is_ct_mode:
        trade_check = select(TradeHistory).where(
            TradeHistory.market_id == market.market_id,
            TradeHistory.status.in_(["SUCCESS", "LIVE", "FAILED", "PAPER", "SHADOW", "PENDING"]),
            (TradeHistory.strategy_name == "BTC_CT_T5_V1") | (TradeHistory.strategy_type == "CT_OUTSIDER"),
        )
    else:
        trade_check = select(TradeHistory).where(
            TradeHistory.market_id == market.market_id,
            TradeHistory.status.in_(["SUCCESS", "LIVE", "FAILED", "PAPER", "SHADOW", "PENDING"]),
            (TradeHistory.strategy_type != "CT_OUTSIDER") | (TradeHistory.strategy_type.is_(None)),
            (TradeHistory.strategy_name != "BTC_CT_T5_V1") | (TradeHistory.strategy_name.is_(None)),
        )
    result = await db_session.execute(trade_check)
    already_traded = result.scalars().first() is not None
    if already_traded:
        return GuardResult(passed=False, skip_reason="guard: Trade already exists", existing_skipped=existing_skipped)

    if is_ct_mode:
        if market.asset.upper() != "BTC":
            return GuardResult(passed=False, skip_reason="guard: Asset not in TRADE_ASSETS", existing_skipped=existing_skipped)
    elif market.asset not in cfg.trade_assets:
        return GuardResult(passed=False, skip_reason="guard: Asset not in TRADE_ASSETS", existing_skipped=existing_skipped)
        
    yes_token_id = market.yes_token_id
    no_token_id = market.no_token_id
    if not yes_token_id or not no_token_id or yes_token_id == 'N/A' or no_token_id == 'N/A':
        logger.error("cannot_find_token_id_in_db", market_id=market.market_id)
        return GuardResult(passed=False, skip_reason="guard: Token IDs missing in DB", existing_skipped=existing_skipped)

    if getattr(cfg, "require_reversion_regime", False):
        # Item 29: Distinguish outsider vs favorite limits.
        # Do not block entire market (favorites) because outsider reversion failed!
        is_strictly_favorite = (is_outsider is False) or (
            is_outsider is None
            and getattr(cfg, "trade_on_favorite", False) is True
            and getattr(cfg, "trade_on_flip", False) is False
        )
        if not is_strictly_favorite:
            from polyflip.db.models import MarketSnapshot
            from polyflip.research.regime_features import classify_local_regime
            try:
                snaps_stmt = (
                    select(MarketSnapshot.mid_price)
                    .where(
                        MarketSnapshot.market_id == market.market_id,
                        MarketSnapshot.recorded_at <= start_time,
                        MarketSnapshot.recorded_at >= start_time - timedelta(minutes=15),
                    )
                    .order_by(MarketSnapshot.recorded_at.asc())
                )
                snaps_res = await db_session.execute(snaps_stmt)
                prices = [float(p) for p in snaps_res.scalars().all() if p is not None]
                if len(prices) >= 3:
                    regime = classify_local_regime(prices, min_observations=3)
                    if regime["state"] != "REVERSION":
                        return GuardResult(
                            passed=False,
                            skip_reason=f"guard: Non-reversion regime ({regime['state']} != REVERSION)",
                            existing_skipped=existing_skipped,
                        )
            except Exception as exc:
                logger.warning("regime_guard_check_failed", market_id=market.market_id, error=str(exc))

    return GuardResult(passed=True, skip_reason=None, existing_skipped=existing_skipped)
