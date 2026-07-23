import asyncio
from polyflip.db.connection import async_session
from polyflip.api.trading_dashboard import get_pnl_markers

async def main():
    async with async_session() as db:
        res = await get_pnl_markers(hours=168, db=db)
        print(f"OK: PnL markers count = {res['count']}")
        for m in res["markers"][:5]:
            print(f"  {m['timestamp']} | {m['label']} | {m['marker_type']}")

if __name__ == "__main__":
    asyncio.run(main())
