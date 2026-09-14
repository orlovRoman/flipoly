import json
import logging
import os
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from polyflip.research.lp_rewards.execution import LiveOrderExecutor
from polyflip.research.lp_rewards.models import MarketRewardConfig
from polyflip.research.lp_rewards.protocol import load_protocol

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("live_calibration")


def main():
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

    if not isolated_wallet and not wallet_key:
        logger.error("[BLOCKED BY HARD GATE] LP_ISOLATED_WALLET_ADDRESS or LP_WALLET_PRIVATE_KEY must be configured.")
        sys.exit(1)

    if isolated_wallet and main_wallet and isolated_wallet.lower() == main_wallet.lower():
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
    )

    logger.info("[ALL GATES PASSED] LiveOrderExecutor initialized with all safety constraints active.")
    logger.info(f"Allocated working capital limit: ${protocol.capital_allocation.allocated_working_capital}")
    logger.info("Live calibration worker is armed and ready for active order placement.")


if __name__ == "__main__":
    main()
