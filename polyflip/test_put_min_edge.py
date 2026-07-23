import asyncio
import httpx

async def main():
    async with httpx.AsyncClient() as client:
        res = await client.put(
            "http://127.0.0.1:8001/api/settings/bulk",
            json={"settings": {"MIN_EDGE": 0.05}},
            headers={"X-API-Key": "test-key"}
        )
        print(f"Status: {res.status_code}, Response: {res.text}")

if __name__ == "__main__":
    asyncio.run(main())
