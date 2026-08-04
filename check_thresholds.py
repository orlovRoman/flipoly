import asyncio
from sqlalchemy import text
from polyflip.db.session import async_session

async def run():
    async with async_session() as session:
        result = await session.execute(text("""
        SELECT
            m.asset,
            m.version,
            m.activation_source,
            m.accuracy AS auc,
            m.ece,
            m.decision_threshold AS model_threshold,
            rs.value AS runtime_threshold,
            CASE
                WHEN m.decision_threshold IS NULL THEN 'MODEL THRESHOLD NULL'
                WHEN rs.value IS NULL THEN 'RUNTIME THRESHOLD MISSING'
                WHEN abs(m.decision_threshold - rs.value::numeric) > 0.0001
                    THEN 'MISMATCH'
                ELSE 'OK'
            END AS threshold_status
        FROM model_registry m
        LEFT JOIN runtime_settings rs
            ON rs.key = 'CRYPTO_THRESHOLD_' || m.asset
        WHERE m.is_active IS TRUE
          AND m.asset LIKE '%USDT_%'
        ORDER BY m.asset;
        """))
        for row in result:
            print(row)

if __name__ == "__main__":
    asyncio.run(run())
