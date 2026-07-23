import asyncio
from polyflip.db.connection import async_session
from polyflip.db.models import DecisionFunnelLog
from sqlalchemy import select

async def main():
    async with async_session() as s:
        stmt = select(DecisionFunnelLog).order_by(DecisionFunnelLog.id.desc()).limit(10)
        res = await s.execute(stmt)
        for r in res.scalars().all():
            print(f"ID: {r.id:5} | Asset: {r.asset:5} | Action: {r.final_action:8} | BlockedBy: {r.gate_blocked_by:15} | Reason: {r.reason}")

if __name__ == "__main__":
    asyncio.run(main())
