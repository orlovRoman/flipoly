from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from polyflip.db.models import RuntimeSettings
from polyflip.config import settings


async def load_trading_settings(
    db_session: AsyncSession,
    trade_assets: list[str] | None = None,
) -> dict[str, str]:
    """
    Загружает все настройки из RuntimeSettings.
    Возвращает сырой dict[str, str] — парсинг на стороне вызывающего.
    """
    stmt = select(RuntimeSettings)
    result = await db_session.execute(stmt)
    settings_db = {s.key: str(s.value) for s in result.scalars().all()}
    
    return settings_db
