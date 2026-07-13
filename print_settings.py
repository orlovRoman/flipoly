import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
import json

async def main():
    # Подключаемся к базе внутри контейнера
    engine = create_async_engine("sqlite+aiosqlite:////app/vault/database.sqlite")
    try:
        async with engine.begin() as conn:
            res = await conn.execute(text("SELECT key, value FROM runtime_settings ORDER BY key"))
            settings = {row[0]: row[1] for row in res}
            
            print("\n=== ТЕКУЩИЕ БОЕВЫЕ НАСТРОЙКИ (runtime_settings) ===\n")
            for k, v in settings.items():
                print(f"{k}: {v}")
            print("\n===================================================\n")
    except Exception as e:
        print(f"Ошибка чтения базы: {e}")

if __name__ == "__main__":
    asyncio.run(main())
