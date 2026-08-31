from polyflip.execution.config import ExecutionSettings, ExecutionMode
from polyflip.execution.gateways.fake import FakeExecutionGateway
from polyflip.execution.gateways.shadow import ShadowExecutionGateway


def build_execution_gateway(settings: ExecutionSettings, *, paper_config=None, quote_provider=None):
    match settings.execution_mode:
        case ExecutionMode.PAPER:
            if quote_provider is None:
                async def quote_provider(token_id: str):
                    from polyflip.collector.client import PolymarketClient

                    async with PolymarketClient() as client:
                        return await client.get_market_prices(token_id)

            config = paper_config or {}
            profile = str(config.get("profile", settings.paper_execution_profile)).strip().upper()
            if profile not in {"INSTANT", "LIVE_PARITY"}:
                profile = "LIVE_PARITY"
            # INSTANT is an explicit test-only compatibility profile and keeps
            # zero-cost deterministic fills. Production PAPER defaults to the
            # LIVE_PARITY branch below.
            parity_enabled = profile == "LIVE_PARITY"
            return FakeExecutionGateway(
                profile=profile,
                quote_provider=quote_provider,
                delay_sec=(config.get("delay_sec", settings.paper_live_delay_sec) if parity_enabled else 0),
                slippage_pct=(config.get("slippage_pct", settings.paper_slippage_pct) if parity_enabled else 0),
                fee_rate=(config.get("fee_rate", settings.paper_fee_rate) if parity_enabled else 0),
                fee_exponent=(config.get("fee_exponent", settings.paper_fee_exponent) if parity_enabled else 1),
                fee_model=(config.get("fee_model", settings.paper_fee_model) if parity_enabled else "FLAT_NOTIONAL"),
                min_order_shares=(config.get("min_order_shares", settings.paper_min_order_shares) if parity_enabled else settings.paper_min_order_shares),
            )
        case ExecutionMode.SHADOW:
            return ShadowExecutionGateway()
        case ExecutionMode.LIVE:
            from polyflip.execution.gateways.polymarket import (
                PolymarketExecutionGateway,
            )

            return PolymarketExecutionGateway(
                private_key=settings.polygon_private_key,  # type: ignore
                wallet_address=settings.polygon_address,  # type: ignore
                relayer_api_key=settings.polymarket_relayer_api_key,  # type: ignore
                relayer_api_key_address=(
                    settings.polymarket_relayer_api_key_address
                ),  # type: ignore
                host=settings.polymarket_host,
            )
        case _:
            raise RuntimeError(f"Unsupported execution mode: {settings.execution_mode}")
