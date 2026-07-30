import asyncio
import os
import sys
import structlog
from decimal import Decimal

from polyflip.execution.gateways.polymarket import PolymarketGateway

logger = structlog.get_logger(__name__)


async def main() -> None:
    private_key = os.getenv("POLYGON_PRIVATE_KEY")
    wallet_address = os.getenv("POLYGON_ADDRESS")

    if not private_key or not wallet_address:
        logger.error("Missing POLYGON_PRIVATE_KEY or POLYGON_ADDRESS environment variables.")
        sys.exit(1)

    print(f"=== Starting Polymarket Approvals Setup for {wallet_address} ===")
    gateway = PolymarketGateway(private_key=private_key, wallet_address=wallet_address)

    # 1. Проверка состояния readiness до утверждений
    readiness = await gateway.get_readiness()
    print(f"Initial Balance USDC: {readiness.balance.balance_usdc}")
    print(f"Collateral Allowance Ready: {readiness.collateral_allowance_ready}")
    print(f"Conditional Allowance Ready: {readiness.conditional_allowance_ready}")

    # 2. Выставление Collateral Allowance
    print("Enabling Collateral (USDC) Allowance...")
    try:
        await gateway.approve_token("USDC")
        print("USDC Approval transaction completed successfully.")
    except Exception as e:
        logger.error("Failed to approve USDC collateral", error=str(e))
        sys.exit(1)

    # 3. Выставление Conditional Token (ERC-1155) Approvals for CTF Exchange
    print("Enabling Conditional Tokens (ERC-1155) Allowance...")
    try:
        await gateway.approve_token("CTF")
        print("CTF Approval transaction completed successfully.")
    except Exception as e:
        logger.error("Failed to approve CTF tokens", error=str(e))
        sys.exit(1)

    # 4. Повторная проверка допусков
    print("Re-checking allowances after approvals...")
    final_readiness = await gateway.get_readiness()

    print(f"Final Collateral Allowance Ready: {final_readiness.collateral_allowance_ready}")
    print(f"Final Conditional Allowance Ready: {final_readiness.conditional_allowance_ready}")

    if not final_readiness.collateral_allowance_ready or not final_readiness.conditional_allowance_ready:
        print("ERROR: Approvals re-check failed! One or more allowances are still missing.")
        sys.exit(1)

    print("=== All Polymarket Approvals Successfully Verified ===")
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
