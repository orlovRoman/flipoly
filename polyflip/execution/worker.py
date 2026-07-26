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

import os
from datetime import timedelta
import uuid

async def claim_one(session) -> ExecutionRequest | None:
    now = datetime.now(timezone.utc)
    # SQLite doesn't support skip_locked, but postgres does.
    # In sqlite, skip_locked is ignored or raises an error, wait, SQLAlchemy might suppress it or we can just not use it if sqlite.
    # Wait, earlier I noted that sqlite doesn't support skip_locked.
    dialect = session.bind.dialect.name
    
    # We want requests that are READY, or CLAIMED but expired
    from sqlalchemy import or_, and_
    
    where_clause = or_(
        ExecutionRequest.state == "READY",
        and_(
            ExecutionRequest.state == "CLAIMED",
            ExecutionRequest.lease_expires_at < now
        )
    )
    
    stmt = select(ExecutionRequest).where(where_clause).limit(1)
    if dialect != 'sqlite':
        stmt = stmt.with_for_update(skip_locked=True)
        
    result = await session.execute(stmt)
    req = result.scalar_one_or_none()
    
    if req:
        req.state = "CLAIMED"
        req.claimed_by = f"worker-{os.getpid()}"
        req.claimed_at = now
        req.lease_expires_at = now + timedelta(seconds=30)
        req.updated_at = now
        await session.commit()
        return req
    return None

async def process_ready_requests():
    settings = ExecutionSettings()
    
    async with async_session() as session:
        req = await claim_one(session)
        if not req:
            return
            
        logger.info("execution_request_claimed", request_id=str(req.id), intent=req.intent)
        
        # Determine actual mode
        mode_str = "PAPER"
        is_live_allowed = False
        if req.requested_mode == "LIVE":
            rt_stmt = select(RuntimeSettings).where(RuntimeSettings.key == "LIVE_TRADING_ENABLED")
            rt_res = await session.execute(rt_stmt)
            rt_set = rt_res.scalar_one_or_none()
            if rt_set and rt_set.value.lower() == "true":
                is_live_allowed = True
                if settings.execution_mode == ExecutionMode.LIVE:
                    mode_str = "LIVE"
                
        elif req.requested_mode == "SHADOW" and settings.execution_mode in (ExecutionMode.SHADOW, ExecutionMode.LIVE):
            mode_str = "SHADOW"

        # LIVE mode must never be executed via fake gateway. If LIVE is requested but kill switch is off, block the request.
        if req.requested_mode == "LIVE" and not is_live_allowed:
            req.state = "REJECTED"
            req.error_reason = "LIVE trading kill switch is off"
            req.updated_at = datetime.now(timezone.utc)
            await session.commit()
            if req.trade_history_id and req.intent == "OPEN":
                from polyflip.db.execution_models import ExposureReservation
                from sqlalchemy import delete
                await session.execute(delete(ExposureReservation).where(ExposureReservation.trade_history_id == req.trade_history_id))
                await session.commit()
            return
            
        actual_settings = ExecutionSettings(execution_mode=ExecutionMode(mode_str))
        gateway = build_execution_gateway(actual_settings)
        
        if req.requested_mode == "LIVE" and gateway.name == "FAKE":
            req.state = "REJECTED"
            req.error_reason = "LIVE mode cannot be executed via fake gateway"
            req.updated_at = datetime.now(timezone.utc)
            await session.commit()
            if req.trade_history_id and req.intent == "OPEN":
                from polyflip.db.execution_models import ExposureReservation
                from sqlalchemy import delete
                await session.execute(delete(ExposureReservation).where(ExposureReservation.trade_history_id == req.trade_history_id))
                await session.commit()
            return
        
        # Get token_id
        market_stmt = select(LiveMarket).where(LiveMarket.market_id == req.market_id)
        market_res = await session.execute(market_stmt)
        market = market_res.scalar_one_or_none()
        
        if not market:
            req.state = "REJECTED"
            req.error_reason = "Market not found"
            req.updated_at = datetime.now(timezone.utc)
            await session.commit()
            if req.trade_history_id and req.intent == "OPEN":
                from polyflip.db.execution_models import ExposureReservation
                from sqlalchemy import delete
                await session.execute(delete(ExposureReservation).where(ExposureReservation.trade_history_id == req.trade_history_id))
                await session.commit()
            return
            
        token_id = market.yes_token_id if req.outcome_to_buy == "YES" else market.no_token_id
        side = "BUY" if req.intent == "OPEN" else "SELL"
        
        limit_price = req.limit_price or Decimal("0")
        max_spend_usdc = req.max_spend_usdc or Decimal("0")
            
        # Deterministic submission key
        attempt_count_stmt = select(ExecutionAttempt).where(ExecutionAttempt.request_id == req.id)
        attempt_count = len((await session.execute(attempt_count_stmt)).scalars().all())
        attempt_no = attempt_count + 1
        
        submission_key = f"{req.idempotency_key}:{attempt_no}"
        
        # Create attempt
        attempt = ExecutionAttempt(
            request_id=req.id,
            gateway=gateway.name,
            attempt_no=attempt_no,
            submission_key=submission_key,
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
            requested_shares=req.requested_shares or Decimal("0"),
            max_spend_usdc=max_spend_usdc
        )
        
        try:
            sub_res = await gateway.submit(order)
            attempt.finished_at = datetime.now(timezone.utc)
            attempt.provider_order_id = sub_res.provider_order_id
            attempt.provider_status = sub_res.status
            
            if "REJECTED" in sub_res.status or "ERROR" in sub_res.status:
                attempt.status = "FAILED"
                attempt.error_msg = sub_res.status
                req.state = "REJECTED"
                req.error_reason = sub_res.status
            elif sub_res.status == "SUBMITTED" or sub_res.status == "UNKNOWN":
                attempt.status = "SUCCESS" 
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
                        provider_trade_id=f.provider_trade_id,
                        gateway=f.gateway,
                        gross_quote_usdc=f.gross_quote_usdc,
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
            
            if req.trade_history_id:
                await rebuild_trade_accounting(session, req.trade_history_id)
                
            # Clean up ExposureReservation if it exists
            if req.trade_history_id and req.intent == "OPEN" and req.state in ("FILLED", "REJECTED"):
                from polyflip.db.execution_models import ExposureReservation
                from sqlalchemy import delete
                await session.execute(
                    delete(ExposureReservation).where(ExposureReservation.trade_history_id == req.trade_history_id)
                )
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

from polyflip.db.models import TradeHistory

async def rebuild_trade_accounting(session, trade_id: int):
    # Fetch trade
    trade = (await session.execute(
        select(TradeHistory).where(TradeHistory.id == trade_id).with_for_update()
    )).scalar_one_or_none()
    
    if not trade:
        return

    # Fetch all requests
    reqs_result = await session.execute(
        select(ExecutionRequest).where(ExecutionRequest.trade_history_id == trade_id)
    )
    reqs = reqs_result.scalars().all()
    
    open_shares = Decimal("0")
    open_cost = Decimal("0")
    close_shares = Decimal("0")
    close_revenue = Decimal("0")
    total_fees = Decimal("0")
    
    for req in reqs:
        # Load attempts and fills
        attempts = (await session.execute(
            select(ExecutionAttempt).where(ExecutionAttempt.request_id == req.id)
        )).scalars().all()
        
        for attempt in attempts:
            from polyflip.db.execution_models import ExecutionFill
            fills = (await session.execute(
                select(ExecutionFill).where(ExecutionFill.attempt_id == attempt.id)
            )).scalars().all()
            
            for fill in fills:
                total_fees += fill.fee_usdc
                if req.intent == "OPEN":
                    open_shares += fill.shares
                    open_cost += (fill.gross_quote_usdc or fill.shares * fill.price)
                elif req.intent == "CLOSE":
                    close_shares += fill.shares
                    close_revenue += (fill.gross_quote_usdc or fill.shares * fill.price)

    avg_entry_price = open_cost / open_shares if open_shares > Decimal("0") else Decimal("0")
    realized_pnl = close_revenue - (close_shares * avg_entry_price) - total_fees

    trade.entry_filled_shares = open_shares
    trade.entry_cost_usdc = open_cost
    trade.remaining_shares = open_shares - close_shares
    trade.realized_pnl_usdc = realized_pnl
    
    if trade.entry_filled_shares > 0 and trade.remaining_shares <= Decimal("0"):
        trade.position_status = "CLOSED"
    elif trade.remaining_shares > Decimal("0") and trade.remaining_shares < trade.entry_filled_shares:
        trade.position_status = "PARTIALLY_CLOSED"
    elif trade.remaining_shares > Decimal("0"):
        trade.position_status = "OPEN"
        
    if open_shares > Decimal("0") or close_shares > Decimal("0"):
        trade.status = "SUCCESS"
    else:
        # Check if all requests are in a terminal failed state
        all_failed = all(r.state in ("REJECTED", "FAILED") for r in reqs)
        if all_failed and reqs:
            trade.status = "FAILED"
            
    trade.position_accounting_version = (trade.position_accounting_version or 0) + 1
    await session.commit()

async def reconcile_active_requests():
    settings = ExecutionSettings()
    async with async_session() as session:
        # Find requests in UNKNOWN or SUBMITTING state that are stuck for more than 1 minute
        now = datetime.now(timezone.utc)
        # using a simple select for now, SQLite/Postgres friendly
        stmt = select(ExecutionRequest).where(ExecutionRequest.state.in_(("UNKNOWN", "SUBMITTING")))
        result = await session.execute(stmt)
        reqs = result.scalars().all()
        
        for req in reqs:
            # Check how long it's been in this state
            if not req.updated_at or (now - req.updated_at).total_seconds() < 60:
                continue
                
            logger.info("reconciling_request", request_id=str(req.id), state=req.state)
            
            # Get latest attempt
            attempt_stmt = select(ExecutionAttempt).where(ExecutionAttempt.request_id == req.id).order_by(ExecutionAttempt.attempt_no.desc()).limit(1)
            attempt_res = await session.execute(attempt_stmt)
            attempt = attempt_res.scalar_one_or_none()
            
            if not attempt or not attempt.provider_order_id:
                logger.warning("cannot_reconcile_no_provider_id", request_id=str(req.id))
                # Mark as FAILED because we don't have an order id to check
                req.state = "REJECTED"
                req.error_reason = "Stuck in SUBMITTING/UNKNOWN with no provider_order_id"
                req.updated_at = now
                if attempt:
                    attempt.status = "FAILED"
                    attempt.error_msg = req.error_reason
                await session.commit()
                continue
                
            gateway = build_execution_gateway(settings)
            
            try:
                sub_res = await gateway.get_order(attempt.provider_order_id)
                if sub_res.status == "FILLED":
                    attempt.status = "SUCCESS"
                    attempt.provider_status = sub_res.status
                    req.state = "FILLED"
                    req.filled_shares = req.requested_shares or Decimal("0")
                    req.filled_cost_usdc = req.target_amount_usdc or Decimal("0")
                    
                    from polyflip.db.execution_models import ExecutionFill
                    for f in sub_res.fills:
                        # Check if fill already exists
                        fill_check = await session.execute(select(ExecutionFill).where(ExecutionFill.provider_trade_id == f.provider_trade_id))
                        if not fill_check.scalar_one_or_none():
                            session.add(ExecutionFill(
                                attempt_id=attempt.id,
                                provider_trade_id=f.provider_trade_id,
                                gateway=f.gateway,
                                gross_quote_usdc=f.gross_quote_usdc,
                                price=f.price,
                                shares=f.shares,
                                fee_usdc=f.fee_usdc,
                                timestamp=f.matched_at
                            ))
                            
                    req.updated_at = now
                    await session.commit()
                    
                    if req.trade_history_id:
                        await rebuild_trade_accounting(session, req.trade_history_id)
                        
                    # Clean up ExposureReservation if it exists
                    if req.trade_history_id and req.intent == "OPEN":
                        from polyflip.db.execution_models import ExposureReservation
                        from sqlalchemy import delete
                        await session.execute(
                            delete(ExposureReservation).where(ExposureReservation.trade_history_id == req.trade_history_id)
                        )
                        await session.commit()
                        
                elif sub_res.status in ("REJECTED", "FAILED", "EXPIRED", "CANCELED"):
                    attempt.status = "FAILED"
                    attempt.provider_status = sub_res.status
                    req.state = "REJECTED"
                    req.error_reason = f"Gateway returned final status: {sub_res.status}"
                    req.updated_at = now
                    await session.commit()
                    
                    # Clean up ExposureReservation if it exists
                    if req.trade_history_id and req.intent == "OPEN":
                        from polyflip.db.execution_models import ExposureReservation
                        from sqlalchemy import delete
                        await session.execute(
                            delete(ExposureReservation).where(ExposureReservation.trade_history_id == req.trade_history_id)
                        )
                        await session.commit()
                else:
                    # Still pending/unknown on the gateway
                    logger.info("reconcile_still_pending", request_id=str(req.id), gateway_status=sub_res.status)
                    
            except Exception as e:
                logger.exception("reconcile_failed", request_id=str(req.id), error=str(e))

async def execution_worker_loop():
    logger.info("execution_worker_started")
    while True:
        try:
            await process_ready_requests()
            await reconcile_active_requests()
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
