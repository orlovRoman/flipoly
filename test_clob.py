import asyncio
import httpx

async def main():
    async with httpx.AsyncClient() as client:
        # Get active 15m markets
        res = await client.get("https://gamma-api.polymarket.com/events", params={"active": "true", "closed": "false", "tag_slug": "15m", "limit": 100})
        events = res.json()
        if not events:
            return
        
        # find BTC market
        btc_event = next((e for e in events if "Bitcoin" in e.get("title", "")), None)
        if not btc_event:
            return
            
        market = btc_event["markets"][0]
        import json
        clob_ids = json.loads(market.get("clobTokenIds", "[]"))
        if not clob_ids:
            return
            
        token = clob_ids[0]
        print("Token:", token)
        
        book_res = await client.get("https://clob.polymarket.com/book", params={"token_id": token})
        book = book_res.json()
        bids = book.get("bids", [])
        asks = book.get("asks", [])
        best_bid = float(bids[0].get("price", 0)) if bids else 0
        best_ask = float(asks[0].get("price", 1)) if asks else 1
        print("Bid:", best_bid, "Ask:", best_ask, "Mid:", (best_bid + best_ask)/2)

asyncio.run(main())
