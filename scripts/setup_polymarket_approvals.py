import argparse
import asyncio
import os
import structlog

from polyflip.execution.gateways.polymarket import PolymarketExecutionGateway

logger = structlog.get_logger(__name__)


async def run_setup(token_ids: list[str] | None = None) -> None:
    token_ids = token_ids or []

    if not token_ids:
        logger.error(
            "At least one --token-id is required to verify Conditional Token approvals"
        )
        raise SystemExit(2)

    private_key = os.getenv("POLYGON_PRIVATE_KEY")
    wallet_address = os.getenv("POLYGON_ADDRESS")
    relayer_api_key = os.getenv("POLYMARKET_RELAYER_API_KEY")
    relayer_api_key_address = os.getenv("POLYMARKET_RELAYER_API_KEY_ADDRESS")

    if (
        not private_key
        or not wallet_address
        or not relayer_api_key
        or not relayer_api_key_address
    ):
        logger.error(
            "Missing POLYGON_PRIVATE_KEY, POLYGON_ADDRESS, "
            "POLYMARKET_RELAYER_API_KEY, or POLYMARKET_RELAYER_API_KEY_ADDRESS "
            "environment variables."
        )
        raise SystemExit(1)

    host = os.getenv("POLYMARKET_HOST", "https://clob.polymarket.com")
    gateway = PolymarketExecutionGateway(
        private_key=private_key,
        wallet_address=wallet_address,
        relayer_api_key=relayer_api_key,
        relayer_api_key_address=relayer_api_key_address,
        host=host,
    )

    client = await gateway.get_client()
    if client is None:
        logger.error("Polymarket client initialization failed")
        raise SystemExit(1)

    print("Setting up trading approvals via Polymarket SDK...")
    await client.setup_trading_approvals()
    print("SDK setup_trading_approvals() finished.")

    print("Verifying readiness...")
    readiness = await gateway.get_readiness(conditional_token_ids=tuple(token_ids))

    if readiness.collateral_allowance_ready is not True:
        print("ERROR: Collateral allowance is not ready")
        raise SystemExit(1)

    if readiness.conditional_allowance_ready is not True:
        print("ERROR: Conditional Token approval is not ready")
        raise SystemExit(1)

    if not readiness.ready:
        print(
            f"ERROR: Polymarket readiness check failed after approvals setup: {readiness}"
        )
        raise SystemExit(1)

    print("=== All Polymarket Approvals Successfully Verified ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup Polymarket trading approvals")
    parser.add_argument(
        "--token-id",
        action="append",
        dest="token_ids",
        help="Token IDs to verify conditional allowances",
    )
    args = parser.parse_args()

    asyncio.run(run_setup(args.token_ids))


if __name__ == "__main__":
    main()
