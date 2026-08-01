from sqlalchemy.ext.asyncio import AsyncSession
from polyflip.db.execution_models import ExecutionRequest
from polyflip.execution.manual_review_service import evaluate_no_fill_eligibility_batch


async def serialize_execution_requests(
    db: AsyncSession,
    requests: list[ExecutionRequest],
) -> list[dict]:
    eligibility = await evaluate_no_fill_eligibility_batch(db, requests)

    return [
        {
            "id": str(req.id),
            "trade_history_id": req.trade_history_id,
            "intent": req.intent,
            "trigger_reason": req.trigger_reason,
            "market_id": req.market_id,
            "asset": req.asset,
            "state": req.state,
            "requested_mode": req.requested_mode,
            "requested_shares": (
                float(req.requested_shares) if req.requested_shares else None
            ),
            "limit_price": float(req.limit_price) if req.limit_price else None,
            "target_amount_usdc": (
                float(req.target_amount_usdc) if req.target_amount_usdc else None
            ),
            "filled_shares": float(req.filled_shares) if req.filled_shares else 0,
            "filled_cost_usdc": float(req.filled_cost_usdc) if req.filled_cost_usdc else 0,
            "ttl_seconds": req.ttl_seconds,
            "created_at": req.created_at.isoformat() if req.created_at else None,
            "updated_at": req.updated_at.isoformat() if req.updated_at else None,
            "error_reason": req.error_reason,
            "can_mark_no_fill": (
                req.id in eligibility and eligibility[req.id].allowed
            ),
            "review_blockers": list(
                eligibility[req.id].blockers
                if req.id in eligibility
                else ()
            ),
            "available_actions": ["MARK_FAILED_NO_FILL"] if (req.id in eligibility and eligibility[req.id].allowed) else [],
        }
        for req in requests
    ]
