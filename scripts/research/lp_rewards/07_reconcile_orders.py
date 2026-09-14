import asyncio
import datetime
from decimal import Decimal
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import httpx

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from polyflip.research.lp_rewards.protocol import load_protocol

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("reconcile_orders")

CLOB_API_URL = "https://clob.polymarket.com"


async def fetch_remote_orders(
    wallet_address: str,
    client: httpx.AsyncClient,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fetch active orders for user wallet from Polymarket CLOB REST API with pagination and L2 auth."""
    headers = {}
    if api_key:
        headers["POLY_API_KEY"] = api_key
    for env_key, hdr_key in [
        ("POLY_ADDRESS", "POLY_ADDRESS"),
        ("POLY_SIGNATURE", "POLY_SIGNATURE"),
        ("POLY_TIMESTAMP", "POLY_TIMESTAMP"),
        ("POLY_PASSPHRASE", "POLY_PASSPHRASE")
    ]:
        val = os.getenv(env_key)
        if val:
            headers[hdr_key] = val

    all_orders = []
    cursor = None

    url = f"{CLOB_API_URL}/data/orders"
    while True:
        try:
            params = {"maker_address": wallet_address}
            if cursor:
                params["next_cursor"] = cursor

            resp = await client.get(url, params=params, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    all_orders.extend(data)
                    break
                elif isinstance(data, dict):
                    all_orders.extend(data.get("data", []))
                    cursor = data.get("next_cursor")
                    if not cursor or cursor == "LTE=":
                        break
            else:
                logger.warning(f"CLOB orders query returned status {resp.status_code}: {resp.text}")
                break
        except Exception as e:
            logger.warning(f"Network error querying remote CLOB orders: {e}")
            break

    return all_orders


def reconcile_orders(
    local_orders: List[Dict[str, Any]],
    remote_orders: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compare local order state against remote exchange orders."""
    local_by_id = {o.get("order_id") or o.get("id"): o for o in local_orders if o.get("order_id") or o.get("id")}
    remote_by_id = {o.get("order_id") or o.get("id"): o for o in remote_orders if o.get("order_id") or o.get("id")}

    phantom_orders = [r for r_id, r in remote_by_id.items() if r_id not in local_by_id]
    lost_orders = [l for l_id, l in local_by_id.items() if l_id not in remote_by_id]

    price_mismatches = []
    size_mismatches = []

    for oid, l_ord in local_by_id.items():
        if oid in remote_by_id:
            r_ord = remote_by_id[oid]
            l_price = Decimal(str(l_ord.get("price", "0")))
            r_price = Decimal(str(r_ord.get("price", "0")))
            if abs(l_price - r_price) > Decimal("0.0001"):
                price_mismatches.append({"order_id": oid, "local": str(l_price), "remote": str(r_price)})

            l_size = Decimal(str(l_ord.get("size", "0")))
            r_orig_size = Decimal(str(r_ord.get("original_size", r_ord.get("size", "0"))))
            r_matched = Decimal(str(r_ord.get("size_matched", "0")))
            r_size = r_orig_size - r_matched
            
            if abs(l_size - r_size) > Decimal("0.001"):
                size_mismatches.append({"order_id": oid, "local": str(l_size), "remote": str(r_size)})

    in_sync = not (phantom_orders or lost_orders or price_mismatches or size_mismatches)
    return {
        "status": "RECONCILED_IN_SYNC" if in_sync else "DISCREPANCY_DETECTED",
        "in_sync": in_sync,
        "local_count": len(local_orders),
        "remote_count": len(remote_orders),
        "phantom_orders": phantom_orders,
        "lost_orders": lost_orders,
        "price_mismatches": price_mismatches,
        "size_mismatches": size_mismatches,
    }


async def main():
    protocol = load_protocol()
    storage_path = Path(protocol.data_storage.root_path)
    wallet_address = os.getenv("LP_ISOLATED_WALLET_ADDRESS", "0x0000000000000000000000000000000000000000")
    api_key = os.getenv("LP_RELAYER_API_KEY") or os.getenv("POLYMARKET_RELAYER_API_KEY")

    logger.info(f"Reconciling orders for protocol: {protocol.protocol_id} (Wallet: {wallet_address})")

    # Load local open orders if present
    local_orders_file = storage_path / "live_open_orders.json"
    local_orders = []
    if local_orders_file.exists():
        with open(local_orders_file, "r", encoding="utf-8") as f:
            local_orders = json.load(f)

    async with httpx.AsyncClient(timeout=30.0) as client:
        remote_orders = await fetch_remote_orders(wallet_address, client, api_key)

    recon_result = reconcile_orders(local_orders, remote_orders)

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    recon_report = {
        "protocol_id": protocol.protocol_id,
        "protocol_hash": protocol.sha256_hash,
        "reconciled_at": now_utc.isoformat(),
        "wallet_address": wallet_address,
        **recon_result,
    }

    recon_dir = storage_path / "reconciliation"
    recon_dir.mkdir(parents=True, exist_ok=True)
    out_file = recon_dir / f"orders_{now_utc.strftime('%Y-%m-%d_%H%M%S')}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(recon_report, f, indent=2)

    logger.info(f"Reconciliation result: {recon_result['status']}")
    logger.info(f"Local orders: {recon_result['local_count']} | Remote orders: {recon_result['remote_count']}")
    if not recon_result["in_sync"]:
        logger.warning(f"Discrepancies: phantoms={len(recon_result['phantom_orders'])}, lost={len(recon_result['lost_orders'])}")
    logger.info(f"Saved order reconciliation report to {out_file}")


if __name__ == "__main__":
    asyncio.run(main())
