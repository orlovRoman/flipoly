import asyncio
import structlog
import os
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import uuid

from sqlalchemy import select, or_, and_, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from polyflip.db.connection import async_session
from polyflip.db.execution_models import ExecutionRequest, ExecutionAttempt, ExposureReservation, ExecutionFill, ExecutionWorkerStatus
from polyflip.db.models import LiveMarket, RuntimeSettings, TradeHistory
from polyflip.execution.config import ExecutionSettings, ExecutionMode
from polyflip.execution.gateways.factory import build_execution_gateway
from polyflip.execution.contracts import GatewayOrder, GatewayUnavailable
from polyflip.execution.outbox import finalize_request
from polyflip.execution.states import ACTIVE_REQUEST_STATES
from polyflip.execution.risk_checks import check_risk_limits

logger = structlog.get_logger(__name__)

async def claim_one(session) -> ExecutionRequest | None:
    now = datetime.now(timezone.utc)
    dialect = session.bind.dialect.name
    
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
        if req.expires_at and req.expires_at < now:
            await finalize_request(session, req, state="EXPIRED", error="TTL expired")
            await session.commit()
            return None
            
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
        
        mode_str = "PAPER"
        is_live_allowed = False
        if req.requested_mode == "LIVE":
            rt_stmt = select(RuntimeSettings).where(RuntimeSettings.key == "LIVE_TRADING_ENABLED")
            rt_res = await session.execute(rt_stmt)
            rt_set = rt_res.scalar_one_or_none()
            if rt_set and rt_set.value.lower() == "true":
                is_live_allowed = True
            
            if is_live_allowed or req.intent == "CLOSE":
                if settings.execution_mode == ExecutionMode.LIVE:
                    mode_str = "LIVE"
                
        elif req.requested_mode == "SHADOW" and settings.execution_mode in (ExecutionMode.SHADOW, ExecutionMode.LIVE):
            mode_str = "SHADOW"

        if req.requested_mode == "LIVE" and not is_live_allowed and req.intent == "OPEN":
            await finalize_request(session, req, state="REJECTED", error="LIVE trading kill switch is off")
            await session.commit()
            return
            
        actual_settings = ExecutionSettings(execution_mode=ExecutionMode(mode_str))
        gateway = build_execution_gateway(actual_settings)
        
        if req.requested_mode == "LIVE" and gateway.name == "FAKE":
            await finalize_request(session, req, state="REJECTED", error="LIVE mode cannot be executed via fake gateway")
            await session.commit()
            return
        
        market_stmt = select(LiveMarket).where(LiveMarket.market_id == req.market_id)
        market_res = await session.execute(market_stmt)
        market = market_res.scalar_one_or_none()
        
        if not market:
            await finalize_request(session, req, state="REJECTED", error="Market not found")
            await session.commit()
            return
            
        token_id = market.yes_token_id if req.outcome_to_buy == "YES" else market.no_token_id
        side = "BUY" if req.intent == "OPEN" else "SELL"
        
        limit_price = req.limit_price or Decimal("0")
        max_spend_usdc = req.max_spend_usdc or Decimal("0")
            
        attempt_count_stmt = select(ExecutionAttempt).where(ExecutionAttempt.request_id == req.id)
        attempt_count = len((await session.execute(attempt_count_stmt)).scalars().all())
        attempt_no = attempt_count + 1
        
        submission_key = f"{req.idempotency_key}:{attempt_no}"
        
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
        
        risk_error = await check_risk_limits(session, req.intent, max_spend_usdc, req.requested_mode, req.id)
        if risk_error:
            logger.warning("risk_limit_breached", request_id=str(req.id), reason=risk_error)
            await finalize_request(session, req, state="REJECTED", error=f"Risk limit: {risk_error}")
            await session.commit()
            return
        
        try:
            sub_res = await gateway.submit(order)
            attempt.finished_at = datetime.now(timezone.utc)
            attempt.provider_order_id = sub_res.provider_order_id
            attempt.provider_status = sub_res.provider_status
            
            if not sub_res.accepted or "REJECTED" in sub_res.provider_status or "ERROR" in sub_res.provider_status:
                attempt.status = "FAILED"
                attempt.error_msg = sub_res.provider_status
                await finalize_request(session, req, state="REJECTED", error=sub_res.error_message or sub_res.provider_status)
            elif sub_res.provider_status == "SUBMITTED" or sub_res.provider_status == "UNKNOWN":
                attempt.status = "SUCCESS" 
                req.state = "UNKNOWN"
            elif sub_res.provider_status in ("MATCHED", "LIVE", "DELAYED"):
                attempt.status = "SUCCESS"
                
                fills = await gateway.fetch_order_fills(attempt.provider_order_id, token_id)
                filled_shares = sum((fill.shares for fill in fills), Decimal("0"))
                filled_quote = sum((fill.gross_quote_usdc for fill in fills), Decimal("0"))
                
                req.filled_shares = filled_shares
                req.filled_cost_usdc = filled_quote
                
                for f in fills:
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

                if filled_shares == Decimal("0"):
                    if sub_res.provider_status in ("LIVE", "DELAYED"):
                        req.state = "UNKNOWN"
                    else:
                        await finalize_request(session, req, state="REJECTED", error="FAK matched but 0 shares filled (cancelled)")
                elif filled_shares < (req.requested_shares or Decimal("0")):
                    await finalize_request(session, req, state="PARTIALLY_FILLED_FINAL")
                else:
                    await finalize_request(session, req, state="FILLED")
            else:
                attempt.status = "UNKNOWN"
                req.state = "UNKNOWN"
            
            req.updated_at = datetime.now(timezone.utc)
            await session.commit()
            
            if req.trade_history_id:
                await rebuild_trade_accounting(session, req.trade_history_id)
                
        except GatewayUnavailable as e:
            logger.warning("gateway_unavailable_submit", error=str(e), attempt_id=str(attempt.id))
            attempt.status = "UNKNOWN"
            attempt.error_msg = str(e)
            attempt.finished_at = datetime.now(timezone.utc)
            req.state = "UNKNOWN"
            req.updated_at = datetime.now(timezone.utc)
            await session.commit()
            
        except Exception as e:
            logger.exception("gateway_submit_failed", error=str(e), attempt_id=str(attempt.id))
            attempt.status = "FAILED"
            attempt.error_msg = str(e)
            attempt.finished_at = datetime.now(timezone.utc)
            await finalize_request(session, req, state="REJECTED", error=str(e))
            await session.commit()

async def rebuild_trade_accounting(session, trade_id: int):
    trade = (await session.execute(
        select(TradeHistory).where(TradeHistory.id == trade_id).with_for_update()
    )).scalar_one_or_none()
    
    if not trade:
        return

    reqs_result = await session.execute(
        select(ExecutionRequest).where(ExecutionRequest.trade_history_id == trade_id)
    )
    reqs = reqs_result.scalars().all()
    
    if not reqs and trade.position_status in ("OPEN", "CLOSED", "PARTIALLY_CLOSED"):
        return
    
    open_shares = Decimal("0")
    open_cost = Decimal("0")
    close_shares = Decimal("0")
    close_revenue = Decimal("0")
    total_fees = Decimal("0")
    latest_close_time = None
    
    for req in reqs:
        attempts = (await session.execute(
            select(ExecutionAttempt).where(ExecutionAttempt.request_id == req.id)
        )).scalars().all()
        
        for attempt in attempts:
            fills = (await session.execute(
                select(ExecutionFill).where(ExecutionFill.attempt_id == attempt.id)
            )).scalars().all()
            
            for fill in fills:
                total_fees += fill.fee_usdc
                if req.intent == "OPEN":
                    open_shares += fill.shares
                    # Fix: don't sum gross quote multiple times if it was already updated by later fills? 
                    # Actually gross_quote is per fill, so summing is correct.
                    open_cost += (fill.gross_quote_usdc or fill.shares * fill.price)
                elif req.intent == "CLOSE":
                    close_shares += fill.shares
                    close_revenue += (fill.gross_quote_usdc or fill.shares * fill.price)
                    if latest_close_time is None or fill.timestamp > latest_close_time:
                        latest_close_time = fill.timestamp

    if open_shares == Decimal("0") and trade.entry_filled_shares and trade.entry_filled_shares > 0:
        open_shares = Decimal(str(trade.entry_filled_shares))
        open_cost = Decimal(str(trade.entry_cost_usdc or trade.amount_usdc or "0"))

    # Phase 8: PnL formula fix. 
    # The previous code correctly computes avg_entry_price and realized_pnl:
    avg_entry_price = open_cost / open_shares if open_shares > Decimal("0") else Decimal("0")
    realized_pnl = close_revenue - (close_shares * avg_entry_price) - total_fees
    avg_close_price = close_revenue / close_shares if close_shares > Decimal("0") else Decimal("0")

    trade.entry_filled_shares = float(open_shares)
    trade.entry_cost_usdc = float(open_cost)
    trade.amount_usdc = float(open_cost)
    trade.remaining_shares = float(open_shares - close_shares)
    trade.realized_pnl_usdc = float(realized_pnl)
    trade.pnl = float(realized_pnl) # Sync to legacy PnL column
    
    if close_shares > Decimal("0"):
        trade.close_price = float(avg_close_price)
    
    if trade.entry_filled_shares > 0 and trade.remaining_shares <= 0:
        trade.position_status = "CLOSED"
        if latest_close_time:
            trade.closed_at = latest_close_time
    elif trade.remaining_shares > 0 and trade.remaining_shares < trade.entry_filled_shares:
        trade.position_status = "PARTIALLY_CLOSED"
    elif trade.remaining_shares > 0:
        trade.position_status = "OPEN"
        
    if trade.position_status in ("OPEN", "PARTIALLY_CLOSED") and trade.entry_filled_shares > 0:
        close_reqs = [r for r in reqs if r.intent == "CLOSE"]
        all_close_failed = len(close_reqs) > 0 and all(r.state in ("REJECTED", "FAILED") for r in close_reqs)
        
        if trade.stop_loss_pct is not None:
            if trade.stop_loss_status not in ("TRIGGERED", "ACTIVE") or (trade.stop_loss_status == "TRIGGERED" and all_close_failed):
                trade.stop_loss_status = "ACTIVE"
        if trade.take_profit_enabled:
            if trade.take_profit_status not in ("TRIGGERED", "ACTIVE") or (trade.take_profit_status == "TRIGGERED" and all_close_failed):
                trade.take_profit_status = "ACTIVE"
                
        if avg_entry_price > Decimal("0"):
            if trade.stop_loss_pct is not None and trade.stop_loss_price is None:
                trade.stop_loss_price = float(avg_entry_price * (Decimal("1") - Decimal(str(trade.stop_loss_pct)) / Decimal("100")))
            if trade.take_profit_enabled and trade.take_profit_price is None:
                trade.take_profit_price = 0.99
        
    if open_shares > Decimal("0") or close_shares > Decimal("0"):
        trade.status = "SUCCESS"
    else:
        all_failed = all(r.state in ("REJECTED", "FAILED") for r in reqs)
        if all_failed and reqs:
            trade.status = "FAILED"
            
    trade.position_accounting_version = (trade.position_accounting_version or 0) + 1
    await session.commit()

async def reconcile_active_requests():
    settings = ExecutionSettings()
    async with async_session() as session:
        now = datetime.now(timezone.utc)
        stmt = select(ExecutionRequest).where(ExecutionRequest.state.in_(("UNKNOWN", "SUBMITTING")))
        result = await session.execute(stmt)
        reqs = result.scalars().all()
        
        for req in reqs:
            if not req.updated_at or (now - req.updated_at).total_seconds() < 60:
                continue
                
            logger.info("reconciling_request", request_id=str(req.id), state=req.state)
            
            attempt_stmt = select(ExecutionAttempt).where(ExecutionAttempt.request_id == req.id).order_by(ExecutionAttempt.attempt_no.desc()).limit(1)
            attempt_res = await session.execute(attempt_stmt)
            attempt = attempt_res.scalar_one_or_none()
            
            if not attempt or not attempt.provider_order_id:
                logger.warning("cannot_reconcile_no_provider_id", request_id=str(req.id))
                if attempt:
                    attempt.status = "FAILED"
                    attempt.error_msg = "Stuck in SUBMITTING/UNKNOWN with no provider_order_id"
                await finalize_request(session, req, state="REJECTED", error="Stuck in SUBMITTING/UNKNOWN with no provider_order_id")
                await session.commit()
                continue
                
            gateway = build_execution_gateway(settings)
            
            try:
                sub_res = await gateway.get_order(attempt.provider_order_id)
                if sub_res.provider_status == "FILLED":
                    attempt.status = "SUCCESS"
                    attempt.provider_status = sub_res.provider_status
                    
                    market_stmt = select(LiveMarket).where(LiveMarket.market_id == req.market_id)
                    market = (await session.execute(market_stmt)).scalar_one_or_none()
                    token_id = market.yes_token_id if market and req.outcome_to_buy == "YES" else (market.no_token_id if market else "")
                    
                    fills = await gateway.fetch_order_fills(attempt.provider_order_id, token_id)
                    filled_shares = sum((fill.shares for fill in fills), Decimal("0"))
                    filled_quote = sum((fill.gross_quote_usdc for fill in fills), Decimal("0"))
                    
                    req.filled_shares = filled_shares
                    req.filled_cost_usdc = filled_quote
                    
                    for f in fills:
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
                            
                    if filled_shares == Decimal("0"):
                        await finalize_request(session, req, state="REJECTED", error="Order was FILLED but 0 shares matched (cancelled?)")
                    elif filled_shares < (req.requested_shares or Decimal("0")):
                        await finalize_request(session, req, state="PARTIALLY_FILLED_FINAL")
                    else:
                        await finalize_request(session, req, state="FILLED")
                    
                    await session.commit()
                    
                    if req.trade_history_id:
                        await rebuild_trade_accounting(session, req.trade_history_id)
                        
                elif sub_res.provider_status in ("REJECTED", "FAILED", "EXPIRED", "CANCELED"):
                    attempt.status = "FAILED"
                    attempt.provider_status = sub_res.provider_status
                    await finalize_request(session, req, state="REJECTED", error=f"Gateway returned final status: {sub_res.provider_status}")
                    await session.commit()
                else:
                    logger.info("reconcile_still_pending", request_id=str(req.id), gateway_status=sub_res.provider_status)
                    
            except GatewayUnavailable as e:
                logger.warning("gateway_unavailable_reconcile", error=str(e))
            except Exception as e:
                logger.exception("reconcile_failed", request_id=str(req.id), error=str(e))

async def publish_heartbeat():
    settings = ExecutionSettings()
    gateway = build_execution_gateway(settings)
    worker_id = f"worker-{os.getpid()}"
    
    while True:
        try:
            readiness = await gateway.get_readiness()
            now = datetime.now(timezone.utc)
            async with async_session() as session:
                dialect_name = session.bind.dialect.name
                insert_func = sqlite_insert if dialect_name == 'sqlite' else pg_insert
                
                bal = float(readiness.balance.balance_usdc) if readiness.balance else None
                
                stmt = insert_func(ExecutionWorkerStatus).values(
                    worker_id=worker_id,
                    execution_mode=settings.execution_mode.value,
                    heartbeat_at=now,
                    gateway_ready=readiness.ready,
                    credentials_loaded=readiness.credentials_loaded,
                    wallet_address=readiness.wallet_address,
                    balance_usdc=bal,
                    collateral_allowance_ready=readiness.collateral_allowance_ready,
                    conditional_allowance_ready=readiness.conditional_allowance_ready,
                    last_error_message=readiness.error_message
                )
                
                set_dict = {
                    "heartbeat_at": now,
                    "gateway_ready": readiness.ready,
                    "balance_usdc": bal,
                    "collateral_allowance_ready": readiness.collateral_allowance_ready,
                    "conditional_allowance_ready": readiness.conditional_allowance_ready,
                    "last_error_message": readiness.error_message,
                    "execution_mode": settings.execution_mode.value
                }
                
                if dialect_name == 'postgresql':
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["worker_id"],
                        set_=set_dict
                    )
                else:
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["worker_id"],
                        set_=set_dict
                    )
                await session.execute(stmt)
                await session.commit()
        except Exception as e:
            logger.error("heartbeat_failed", error=str(e))
        await asyncio.sleep(15)

async def execution_worker_loop():
    logger.info("execution_worker_started")
    settings = ExecutionSettings()
    if settings.execution_mode == ExecutionMode.SHADOW:
        raise RuntimeError("SHADOW execution mode is not supported by the gateway factory yet")
        
    asyncio.create_task(publish_heartbeat())
    
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
