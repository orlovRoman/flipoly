import asyncio
import os
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from polyflip.config import settings

async def main():
    # Подключаемся к базе внутри контейнера, используя URL из настроек (Postgres)
    db_url = settings.DATABASE_URL
    print(f"Connecting to: {db_url}")
    engine = create_async_engine(db_url)
    try:
        async with engine.begin() as conn:
            res = await conn.execute(text("SELECT key, value FROM runtime_settings ORDER BY key"))
            runtime_settings = {row[0]: row[1] for row in res}
            
            print("\n=== ТЕКУЩИЕ БОЕВЫЕ НАСТРОЙКИ (runtime_settings) ===\n")
            for k, v in runtime_settings.items():
                print(f"{k}: {v}")
            print("\n===================================================\n")
    except Exception as e:
        print(f"Ошибка чтения базы: {e}")

if __name__ == "__main__":
    asyncio.run(main())
