"""
Очищает некорректные пороги из таблицы runtime_settings.
"""
import asyncio
from sqlalchemy import select, delete, text
from polyflip.db.connection import async_session
from polyflip.db.models import RuntimeSettings

async def clean_bad_thresholds():
    async with async_session() as session:
        stmt = text("""
            DELETE FROM runtime_settings 
            WHERE key LIKE 'CRYPTO_THRESHOLD_%' 
              AND (CAST(value AS FLOAT) = 0.0 
                OR CAST(value AS FLOAT) < 0.30 
                OR CAST(value AS FLOAT) > 0.75)
        """)
        res = await session.execute(stmt)
        await session.commit()
        print(f"Очистка завершена. Удалено некорректных порогов: {res.rowcount}")

        # Вывод оставшихся порогов
        check_stmt = select(RuntimeSettings).where(RuntimeSettings.key.like("CRYPTO_THRESHOLD_%"))
        rows = (await session.execute(check_stmt)).scalars().all()
        print("Текущие пороги в БД:")
        for r in rows:
            print(f"  {r.key} = {r.value}")

if __name__ == "__main__":
    asyncio.run(clean_bad_thresholds())
