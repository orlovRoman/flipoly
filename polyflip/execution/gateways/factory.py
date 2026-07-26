from polyflip.execution.config import ExecutionSettings, ExecutionMode
from polyflip.execution.gateways.fake import FakeExecutionGateway

def build_execution_gateway(settings: ExecutionSettings):
    match settings.execution_mode:
        case ExecutionMode.PAPER:
            return FakeExecutionGateway()
        case ExecutionMode.SHADOW:
            raise NotImplementedError("Shadow execution gateway is not implemented yet")
        case ExecutionMode.LIVE:
            from polyflip.execution.gateways.polymarket import PolymarketExecutionGateway
            return PolymarketExecutionGateway(
                private_key=settings.polygon_private_key, # type: ignore
                wallet_address=settings.polygon_address, # type: ignore
                host=settings.polymarket_host
            )
        case _:
            raise RuntimeError(f"Unsupported execution mode: {settings.execution_mode}")
