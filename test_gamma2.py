import asyncio
import aiohttp

async def run():
    async with aiohttp.ClientSession() as http_session:
        http_session.headers.update({"User-Agent": "Mozilla/5.0"})
        async with http_session.get("https://gamma-api.polymarket.com/markets/3103319") as resp:
            data = await resp.json()
            print("By /markets/id:")
            print(data)

asyncio.run(run())
