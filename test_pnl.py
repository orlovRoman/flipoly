import asyncio
from polyflip.db.connection import async_session
from polyflip.api.dashboard import get_model_pnl

async def main():
    async with async_session() as s:
        res = await get_model_pnl("PAPER", s)
        for k, v in res.get("data", {}).items():
            if v.get("total_trades", 0) > 0:
                print(f"{k}: trades={v['total_trades']}, pnl={v['pnl']}, wr={v['win_rate']}%")

if __name__ == "__main__":
    asyncio.run(main())
