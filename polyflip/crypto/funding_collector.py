"""
Коллектор Funding Rate с Binance Futures API.
Endpoint: GET /fapi/v1/fundingRate
Частота обновления: каждые 8 часов (00:00, 08:00, 16:00 UTC)
"""
import httpx
import structlog
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from polyflip.db.models import RuntimeSettings
from sqlalchemy import select

logger = structlog.get_logger(__name__)

BINANCE_FUTURES_URL = "https://fapi.binance.com"

async def fetch_and_store_funding_rate(db: AsyncSession, symbol: str) -> float | None:
    """Получает последний funding rate и сохраняет в RuntimeSettings."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{BINANCE_FUTURES_URL}/fapi/v1/fundingRate",
                params={"symbol": symbol, "limit": 3}
            )
            resp.raise_for_status()
            data = resp.json()

        if not data:
            return None

        # Последние 3 значения → MA3
        rates = [float(r["fundingRate"]) for r in data[-3:]]
        current_rate = rates[-1]
        ma3_rate = sum(rates) / len(rates)

        now = datetime.now(timezone.utc)
        for key, val in [
            (f"FUNDING_RATE_{symbol}", str(round(current_rate, 8))),
            (f"FUNDING_RATE_MA3_{symbol}", str(round(ma3_rate, 8))),
        ]:
            row = (await db.execute(
                select(RuntimeSettings).where(RuntimeSettings.key == key)
            )).scalar_one_or_none()
            if row:
                row.value = val
                row.updated_at = now
                row.updated_by = "funding_collector"
            else:
                db.add(RuntimeSettings(key=key, value=val,
                                       updated_at=now, updated_by="funding_collector"))
        await db.commit()

        logger.info("funding_rate_stored", symbol=symbol,
                    current=current_rate, ma3=ma3_rate)
        return current_rate

    except Exception as e:
        logger.warning("funding_rate_fetch_failed", symbol=symbol, error=str(e))
        return None
