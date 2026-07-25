import asyncio
from polyflip.db.engine import get_db_session
from polyflip.db.models import TradeHistory
from sqlalchemy import select, func

async def main():
    async with get_db_session() as db:
        skipped = (await db.execute(
            select(TradeHistory.error_msg, func.count(TradeHistory.id))
            .where(TradeHistory.status == "SKIPPED")
            .group_by(TradeHistory.error_msg)
        )).all()
        print("=== SKIPPED REASONS IN DB ===")
        total_veto = 0
        total_skipped = 0
        for r, c in skipped:
            total_skipped += c
            if r and ("veto" in r.lower() or "lightgbm" in r.lower() or "confirm" in r.lower()):
                total_veto += c
            print(f"[{c} times]: {r}")
        print(f"\nTOTAL SKIPPED: {total_skipped}, TOTAL VETO: {total_veto}")

        success = (await db.execute(
            select(TradeHistory.active_features, func.count(TradeHistory.id))
            .where(TradeHistory.status == "SUCCESS")
            .group_by(TradeHistory.active_features)
        )).all()
        print("\n=== SUCCESS TRADES BY STRATEGY/FEATURES ===")
        for f, c in success:
            print(f"[{c} trades]: {f}")

if __name__ == "__main__":
    asyncio.run(main())
