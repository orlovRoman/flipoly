import asyncio, json, os, sys
sys.path.insert(0, "/app")

from polyflip.db.engine import get_db_session
from polyflip.db.models import TradeHistory
from sqlalchemy import select

async def main():
    async with get_db_session() as db:
        rows = (await db.execute(
            select(TradeHistory.lgbm_metadata)
            .where(TradeHistory.lgbm_metadata.is_not(None))
        )).scalars().all()

        total = len(rows)
        veto_count = 0
        none_count = 0
        agree_count = 0
        fallback_count = 0

        for raw in rows:
            try:
                data = json.loads(raw)
            except Exception:
                continue

            mult = data.get("bet_size_multiplier", 1.0)
            action = data.get("vote_action", "")
            direction = data.get("lgbm_direction")
            ok = data.get("lgbm_features_ok", True)

            if not ok or data.get("is_fallback"):
                fallback_count += 1
            elif mult == 0.0 or action == "SKIP":
                veto_count += 1
            elif mult == 0.5 or direction == "NONE":
                none_count += 1
            elif mult == 1.0:
                agree_count += 1

        print("=== ТОЧНАЯ СТАТИСТИКА В ЦИФРАХ ===")
        print(f"Всего обработано сигналов в COMBINED режиме: {total}")
        print(f"1. Вето (SKIP, mult=0.0):                   {veto_count}")
        print(f"2. Нейтральный флэт (NONE, mult=0.5):        {none_count}")
        print(f"3. Полное согласие (AGREE, mult=1.0):       {agree_count}")
        print(f"4. Сбои фичей / Fallback:                    {fallback_count}")

if __name__ == "__main__":
    asyncio.run(main())
