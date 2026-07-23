import asyncio
from polyflip.db.connection import async_session
from polyflip.db.models import RuntimeSettings, DecisionFunnelLog
from sqlalchemy import select, func

async def main():
    async with async_session() as s:
        tm_row = (await s.execute(select(RuntimeSettings).where(RuntimeSettings.key == "TRADING_MODE"))).scalar_one_or_none()
        mode = tm_row.value if tm_row else "ML"
        print(f"Current TRADING_MODE: {mode}")

        total_cnt = (await s.execute(select(func.count(DecisionFunnelLog.id)))).scalar() or 0
        g7_cnt = (await s.execute(select(func.count(DecisionFunnelLog.id)).where(DecisionFunnelLog.g7_crypto_confirm == False))).scalar() or 0
        g8_cnt = (await s.execute(select(func.count(DecisionFunnelLog.id)).where(DecisionFunnelLog.g8_combined_vote == False))).scalar() or 0
        traded_cnt = (await s.execute(select(func.count(DecisionFunnelLog.id)).where(DecisionFunnelLog.final_action.in_(["BUY_YES", "BUY_NO"])))).scalar() or 0

        print(f"Total funnel logs: {total_cnt}")
        print(f"Traded (BUY_YES / BUY_NO): {traded_cnt}")
        print(f"Blocked by G7 (Crypto Confirm in ML mode): {g7_cnt}")
        print(f"Blocked by G8 (Combined Vote in COMBINED mode): {g8_cnt}")

if __name__ == "__main__":
    asyncio.run(main())
