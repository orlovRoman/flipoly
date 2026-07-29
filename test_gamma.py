import sys
import asyncio
import aiohttp

async def run():
    async with aiohttp.ClientSession() as http_session:
        # User-agent bypass for Cloudflare
        http_session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        async with http_session.get("https://gamma-api.polymarket.com/markets?condition_id=3103319") as resp:
            data = await resp.json()
            print("By condition_id:")
            print(data)

        async with http_session.get("https://gamma-api.polymarket.com/events/3103319") as resp:
            # Events API? Or markets API? Let's check by id
            pass

        async with http_session.get("https://gamma-api.polymarket.com/markets?id=3103319") as resp:
            data = await resp.json()
            print("By id:")
            print(len(data))
            if data:
                m = data[0]
                print(f"closed: {m.get('closed')}, active: {m.get('active')}")
                
asyncio.run(run())
