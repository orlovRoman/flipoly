import asyncio
import json
import logging
import os
from pathlib import Path
import sys
import time
from decimal import Decimal
from typing import Optional

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from polyflip.research.lp_rewards.execution import LiveOrderExecutor
from polyflip.research.lp_rewards.models import MarketRewardConfig
from polyflip.research.lp_rewards.protocol import load_protocol

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("live_calibration")


def fetch_live_midpoint(clob_client, token_id: str) -> Optional[Decimal]:
    """Fetch real-time midpoint from CLOB for a given token."""
    try:
        if hasattr(clob_client, "get_midpoint"):
            res = clob_client.get_midpoint(token_id)
            if isinstance(res, dict) and "mid" in res and res["mid"] is not None:
                return Decimal(str(res["mid"]))
        if hasattr(clob_client, "get_order_book"):
            book = clob_client.get_order_book(token_id)
            bids = book.get("bids", [])
            asks = book.get("asks", [])
            if bids and asks:
                best_bid = Decimal(str(bids[0].get("price", bids[0][0] if isinstance(bids[0], (list, tuple)) else 0)))
                best_ask = Decimal(str(asks[0].get("price", asks[0][0] if isinstance(asks[0], (list, tuple)) else 0)))
                return (best_bid + best_ask) / Decimal("2")
    except Exception as e:
        logger.warning(f"Could not fetch live midpoint for token {token_id}: {e}")
    return None


async def run_live_calibration():
    protocol = load_protocol()
    storage_path = Path(protocol.data_storage.root_path)
    logger.info(f"Protocol loaded: {protocol.protocol_id} (SHA-256: {protocol.sha256_hash})")

    # Gate 1: Environment flag
    env_live = os.getenv("LP_LIVE_ENABLED", "false").lower() in ("1", "true", "yes")
    if not env_live:
        logger.error("[BLOCKED BY HARD GATE] LP_LIVE_ENABLED is not 'true'. Live trading disabled.")
        sys.exit(1)

    # Gate 2: Gate A verdict artifact verification
    gate_a_file = storage_path / "gate_a_verdict.json"
    if not gate_a_file.exists():
        logger.error(f"[BLOCKED BY HARD GATE] Gate A verdict file not found at {gate_a_file}. Cannot proceed to live mode.")
        sys.exit(1)

    with open(gate_a_file, "r", encoding="utf-8") as f:
        gate_a_data = json.load(f)

    if gate_a_data.get("verdict") != "PROCEED_LIVE":
        logger.error(f"[BLOCKED BY HARD GATE] Gate A verdict is '{gate_a_data.get('verdict')}' (required 'PROCEED_LIVE').")
        sys.exit(1)

    if gate_a_data.get("protocol_hash") != protocol.sha256_hash:
        logger.error(f"[BLOCKED BY HARD GATE] Gate A protocol hash mismatch! Expected {protocol.sha256_hash}, found {gate_a_data.get('protocol_hash')}")
        sys.exit(1)

    logger.info(f"[GATE A CONFIRMED] Verdict: PROCEED_LIVE (Hash: {protocol.sha256_hash[:12]}...)")

    # Gate 3: Dedicated Isolated Wallet validation
    isolated_wallet = os.getenv("LP_ISOLATED_WALLET_ADDRESS")
    main_wallet = os.getenv("PROD_MAIN_WALLET_ADDRESS") or os.getenv("POLYGON_WALLET_ADDRESS")
    wallet_key = os.getenv("LP_WALLET_PRIVATE_KEY")

    zero_address = "0x0000000000000000000000000000000000000000"
    if (
        not isolated_wallet
        or not isolated_wallet.strip()
        or isolated_wallet.strip().lower() == zero_address
    ):
        logger.error("[BLOCKED BY HARD GATE] LP_ISOLATED_WALLET_ADDRESS must be configured and non-zero.")
        sys.exit(1)
    isolated_wallet = isolated_wallet.strip()

    if not wallet_key or not wallet_key.strip():
        logger.error("[BLOCKED BY HARD GATE] LP_WALLET_PRIVATE_KEY must be configured.")
        sys.exit(1)

    if main_wallet and isolated_wallet.lower() == main_wallet.strip().lower():
        logger.error(f"[BLOCKED BY HARD GATE] Isolated wallet {isolated_wallet} matches main production wallet {main_wallet}!")
        sys.exit(1)

    # Gate 4: Active universe allowlist
    universe_file = storage_path / "universe_active.json"
    if not universe_file.exists():
        logger.error(f"[BLOCKED BY HARD GATE] Universe active file not found at {universe_file}.")
        sys.exit(1)

    with open(universe_file, "r", encoding="utf-8") as f:
        raw_univ = json.load(f)
    active_configs = [MarketRewardConfig(**m) for m in raw_univ]

    allowlist_tokens = set()
    for m in active_configs:
        allowlist_tokens.add(m.yes_token_id)
        allowlist_tokens.add(m.no_token_id)

    logger.info(f"[ALLOWLIST LOADED] {len(active_configs)} markets ({len(allowlist_tokens)} tokens allowed).")

    try:
        from py_clob_client_v2 import ClobClient, ApiCreds

        creds = ApiCreds(
            api_key=os.getenv("POLY_API_KEY", ""),
            api_secret=os.getenv("POLY_API_SECRET", ""),
            api_passphrase=os.getenv("POLY_PASSPHRASE", ""),
        )
        clob_client = ClobClient(
            host="https://clob.polymarket.com",
            chain_id=137,
            key=wallet_key,
            creds=creds,
            signature_type=0,
            funder=isolated_wallet,
        )
        clob_client.set_api_creds(creds)
    except Exception as e:
        logger.error(f"[BLOCKED BY HARD GATE] Failed to initialize real CLOB V2 ClobClient: {e}")
        sys.exit(1)

    # Initialize Live Executor
    executor = LiveOrderExecutor(
        expected_protocol_hash=protocol.sha256_hash,
        gate_a_verdict_path=gate_a_file,
        wallet_private_key=wallet_key,
        wallet_address=isolated_wallet,
        main_wallet_address=main_wallet,
        allocated_capital_limit=protocol.capital_allocation.allocated_working_capital,
        allowlist_tokens=allowlist_tokens,
        require_gate_a=True,
        clob_client=clob_client,
    )

    logger.info("[ALL GATES PASSED] LiveOrderExecutor initialized with all safety constraints active.")
    logger.info(f"Allocated working capital limit: ${protocol.capital_allocation.allocated_working_capital}")
    logger.info("Live calibration worker is armed and ready for active order placement.")

    live_orders_file = storage_path / "live_open_orders.json"
    submitted = []

    from polyflip.research.lp_rewards.quoting_fsm import MarketQuotingFSM
    from polyflip.research.lp_rewards.capital_allocator import CapitalAllocator

    allocator = CapitalAllocator(
        allocated_working_capital=Decimal("50.0"),
        max_unhedged_per_market=Decimal("25.0"),
        max_unhedged_total=Decimal("50.0"),
    )

    if wallet_key and active_configs:
        fsms = {m.condition_id: MarketQuotingFSM(m) for m in active_configs}
        try:
            while True:
                for top_m in active_configs:
                    fsm = fsms[top_m.condition_id]
                    now_ns = time.time_ns()
                    live_mid = fetch_live_midpoint(clob_client, top_m.yes_token_id)
                    if live_mid is None or live_mid <= Decimal("0.0") or live_mid >= Decimal("1.0"):
                        logger.warning(
                            f"Skipping market {top_m.condition_id}: live midpoint unavailable from CLOB ({live_mid})"
                        )
                        continue

                    quotes = fsm.generate_quote_orders(midpoint=live_mid, timestamp_ns=now_ns)

                    positions = {top_m.condition_id: fsm.position}
                    open_orders = {top_m.condition_id: list(fsm.open_orders.values())}
                    allowed, _ = allocator.can_allocate_orders(top_m.condition_id, positions, open_orders, quotes)

                    if allowed:
                        for q in quotes:
                            res = executor.submit_order(
                                token_id=q.asset_id,
                                side="BUY" if q.side.value == "BUY" else "SELL",
                                price=q.price,
                                size=q.size,
                                protocol_hash=protocol.sha256_hash,
                            )
                            submitted.append(res)

                with open(live_orders_file, "w", encoding="utf-8") as lf:
                    json.dump(submitted, lf, indent=2)

                logger.info("FSM loop calibration: tick complete, waiting 5 seconds...")
                await asyncio.sleep(5)
        except Exception as e:
            logger.warning(f"Could not place initial calibration orders: {e}")
        except KeyboardInterrupt:
            logger.info("Live calibration interrupted by user.")
        finally:
            executor.cancel_all_orders()
            submitted.clear()
            with open(live_orders_file, "w", encoding="utf-8") as lf:
                json.dump(submitted, lf, indent=2)

    logger.info(f"Updated live open orders registry at {live_orders_file} ({len(submitted)} active orders)")


def main():
    asyncio.run(run_live_calibration())


if __name__ == "__main__":
    main()
