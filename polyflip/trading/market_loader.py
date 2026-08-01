from datetime import datetime, timedelta
from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession
import structlog
from polyflip.db.models import LiveMarket, TradeHistory
from polyflip.trading.trading_config import TradingConfig

# Attempt to import FAVORITE_MODE_ENTRY_WINDOW_SEC, fallback to default 60
try:
    from polyflip.constants import FAVORITE_MODE_ENTRY_WINDOW_SEC
except ImportError:
    try:
        from polyflip.config import settings
        FAVORITE_MODE_ENTRY_WINDOW_SEC = getattr(settings, 'FAVORITE_MODE_ENTRY_WINDOW_SEC', 60)
    except ImportError:
        FAVORITE_MODE_ENTRY_WINDOW_SEC = 60

logger = structlog.get_logger(__name__)


async def load_eligible_markets(
    db_session: AsyncSession,
    cfg: TradingConfig,
    start_time: datetime,
) -> list[LiveMarket] | None:
    """
    Загружает рынки из LiveMarket во временном окне и проверяет дневной лимит.
    Если достигнут дневной лимит, возвращает None (сигнал остановки).
    Иначе возвращает список рынков (может быть пустым).
    """
    union_min_sec = min(cfg.min_time_left, cfg.entry_sec)
    union_max_sec = max(cfg.max_time_left, cfg.entry_sec + FAVORITE_MODE_ENTRY_WINDOW_SEC)
    
    min_td = timedelta(seconds=union_min_sec)
    max_td = timedelta(seconds=union_max_sec)
    
    live_markets_stmt = select(LiveMarket).where(
        and_(
            LiveMarket.end_time_est >= start_time + min_td,
            LiveMarket.end_time_est <= start_time + max_td
        )
    )
    result = await db_session.execute(live_markets_stmt)
    markets = result.scalars().all()
    
    # Проверяем дневной PnL перед тем как торговать (лимит убытка)
    start_of_today = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_today = start_of_today + timedelta(days=1)
    
    daily_pnl_stmt = select(func.sum(TradeHistory.pnl)).where(
        and_(
            TradeHistory.created_at >= start_of_today,
            TradeHistory.created_at < end_of_today,
            TradeHistory.status == "SUCCESS",
            TradeHistory.pnl.is_not(None)
        )
    )

    daily_pnl = (await db_session.execute(daily_pnl_stmt)).scalar() or 0.0
    if daily_pnl <= cfg.daily_limit:
        logger.warning("daily_loss_limit_reached", pnl=daily_pnl, limit=cfg.daily_limit)
        return None

    if not markets:
        return []
        
    return list(markets)
