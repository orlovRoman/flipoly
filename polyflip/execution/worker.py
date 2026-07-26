import asyncio
import structlog
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import select
from polyflip.db.connection import async_session
from polyflip.db.execution_models import ExecutionRequest, ExecutionAttempt
from polyflip.db.models import LiveMarket, RuntimeSettings
from polyflip.execution.config import ExecutionSettings, ExecutionMode
from polyflip.execution.gateways.factory import build_execution_gateway
from polyflip.execution.contracts import GatewayOrder

logger = structlog.get_logger(__name__)

async def process_ready_requests():
    settings = ExecutionSettings()
    
    async with async_session() as session:
        # Use FOR UPDATE SKIP LOCKED to get a single READY request
        stmt = select(ExecutionRequest).where(
            ExecutionRequest.state == "READY"
        ).with_for_update(skip_locked=True)
        
        result = await session.execute(stmt)
        req = result.scalar_one_or_none()
        
        if not req:
            return
            
        logger.info("execution_request_claimed", request_id=str(req.id), intent=req.intent)
        req.state = "CLAIMED"
        req.updated_at = datetime.now(timezone.utc)
        await session.commit()
        
        # Determine actual mode
        mode_str = "PAPER"
        if req.requested_mode == "LIVE" and settings.execution_mode == ExecutionMode.LIVE:
            rt_stmt = select(RuntimeSettings).where(RuntimeSettings.key == "LIVE_TRADING_ENABLED")
            rt_res = await session.execute(rt_stmt)
            rt_set = rt_res.scalar_one_or_none()
            if rt_set and rt_set.value.lower() == "true":
                mode_str = "LIVE"
                
        elif req.requested_mode == "SHADOW" and settings.execution_mode in (ExecutionMode.SHADOW, ExecutionMode.LIVE):
            mode_str = "SHADOW"
            
        actual_settings = ExecutionSettings(execution_mode=ExecutionMode(mode_str))
        gateway = build_execution_gateway(actual_settings)
        
        # Get token_id
        market_stmt = select(LiveMarket).where(LiveMarket.market_id == req.market_id)
        market_res = await session.execute(market_stmt)
        market = market_res.scalar_one_or_none()
        
        if not market:
            req.state = "REJECTED"
            req.error_reason = "Market not found"
            req.updated_at = datetime.now(timezone.utc)
            await session.commit()
            return
            
        token_id = market.yes_token_id if req.outcome_to_buy == "YES" else market.no_token_id
        side = "BUY" if req.intent == "OPEN" else "SELL"
        
        limit_price = Decimal("0")
        if req.requested_shares and req.requested_shares > 0 and req.target_amount_usdc:
            limit_price = req.target_amount_usdc / req.requested_shares
            
        # Create attempt
        attempt = ExecutionAttempt(
            request_id=req.id,
            gateway=gateway.name,
            started_at=datetime.now(timezone.utc)
        )
        session.add(attempt)
        
        req.state = "SUBMITTING"
        req.updated_at = datetime.now(timezone.utc)
        await session.commit()
        
        order = GatewayOrder(
            attempt_id=attempt.id,
            market_id=req.market_id,
            asset=req.asset,
            outcome_to_buy=req.outcome_to_buy,
            token_id=token_id,
            side=side,
            limit_price=limit_price,
            requested_shares=req.requested_shares or Decimal("0")
        )
        
        try:
            sub_res = await gateway.submit(order)
            attempt.finished_at = datetime.now(timezone.utc)
            if "REJECTED" in sub_res.status or "ERROR" in sub_res.status:
                attempt.status = "FAILED"
                attempt.error_msg = sub_res.status
                req.state = "REJECTED"
                req.error_reason = sub_res.status
            elif sub_res.status == "SUBMITTED" or sub_res.status == "UNKNOWN":
                attempt.status = "SUCCESS" # The attempt to submit was successful
                req.state = "UNKNOWN"
            elif sub_res.status == "MATCHED":
                attempt.status = "SUCCESS"
                req.state = "FILLED"
                req.filled_shares = req.requested_shares or Decimal("0")
                req.filled_cost_usdc = req.target_amount_usdc or Decimal("0")
                
                from polyflip.db.execution_models import ExecutionFill
                for f in sub_res.fills:
                    session.add(ExecutionFill(
                        attempt_id=attempt.id,
                        price=f.price,
                        shares=f.shares,
                        fee_usdc=f.fee_usdc,
                        timestamp=f.matched_at
                    ))
            else:
                attempt.status = "UNKNOWN"
                req.state = "UNKNOWN"
            
            req.updated_at = datetime.now(timezone.utc)
            await session.commit()
            
        except Exception as e:
            logger.exception("gateway_submit_failed", error=str(e), attempt_id=str(attempt.id))
            attempt.status = "FAILED"
            attempt.error_msg = str(e)
            attempt.finished_at = datetime.now(timezone.utc)
            
            req.state = "UNKNOWN"
            req.error_reason = str(e)
            req.updated_at = datetime.now(timezone.utc)
            await session.commit()


async def execution_worker_loop():
    logger.info("execution_worker_started")
    while True:
        try:
            await process_ready_requests()
        except Exception as e:
            logger.exception("execution_worker_error", error=str(e))
        await asyncio.sleep(1)

if __name__ == "__main__":
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ]
    )
    asyncio.run(execution_worker_loop())
