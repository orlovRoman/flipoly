import dataclasses
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import structlog
from polyflip.db.models import LiveMarket, TradeHistory
from polyflip.trading.trading_config import TradingConfig
from polyflip.constants import TRADING_MODE_FAVORITE

try:
    from polyflip.constants import TRADE_CHECK_LIMIT
except ImportError:
    TRADE_CHECK_LIMIT = 5

try:
    from polyflip.constants import FAVORITE_MODE_ENTRY_WINDOW_SEC
except ImportError:
    try:
        from polyflip.config import settings
        FAVORITE_MODE_ENTRY_WINDOW_SEC = getattr(settings, 'FAVORITE_MODE_ENTRY_WINDOW_SEC', 60)
    except ImportError:
        FAVORITE_MODE_ENTRY_WINDOW_SEC = 60

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
) -> GuardResult:
    """
    Выполняет все предварительные проверки рынка (guards) перед принятием решения.
    """
    if time_left_sec <= 0:
        return GuardResult(passed=False, skip_reason="Time left <= 0", existing_skipped=None)
        
    if asset_mode == TRADING_MODE_FAVORITE:
        window_min = cfg.entry_sec
        window_max = cfg.entry_sec + FAVORITE_MODE_ENTRY_WINDOW_SEC
        if not (window_min <= time_left_sec <= window_max):
            return GuardResult(passed=False, skip_reason="Outside time window", existing_skipped=None)
    else:
        if not (cfg.min_time_left <= time_left_sec <= cfg.max_time_left):
            return GuardResult(passed=False, skip_reason="Outside time window", existing_skipped=None)
            
    trade_check = select(TradeHistory).where(
        TradeHistory.market_id == market.market_id
    ).limit(TRADE_CHECK_LIMIT)
    result = await db_session.execute(trade_check)
    existing_trades = result.scalars().all()
    
    existing_statuses = [t.status for t in existing_trades]
    if any(s in ("SUCCESS", "LIVE", "FAILED") for s in existing_statuses):
        return GuardResult(passed=False, skip_reason="Trade already exists", existing_skipped=None)
        
    existing_skipped = next((t for t in existing_trades if t.status == "SKIPPED"), None)
    
    if market.asset not in cfg.trade_assets:
        return GuardResult(passed=False, skip_reason="Asset not in TRADE_ASSETS", existing_skipped=existing_skipped)
        
    yes_token_id = market.yes_token_id
    no_token_id = market.no_token_id
    if not yes_token_id or not no_token_id or yes_token_id == 'N/A' or no_token_id == 'N/A':
        logger.error("cannot_find_token_id_in_db", market_id=market.market_id)
        return GuardResult(passed=False, skip_reason="Token IDs missing in DB", existing_skipped=existing_skipped)

    return GuardResult(passed=True, skip_reason=None, existing_skipped=existing_skipped)
