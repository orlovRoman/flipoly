from polyflip.execution.states import RECONCILABLE_REQUEST_STATES
from sqlalchemy.ext.asyncio import AsyncSession
from polyflip.db.execution_models import ExecutionRequest
from polyflip.execution.manual_review_service import evaluate_no_fill_eligibility_batch


def _parse_error(error_reason: str | None) -> dict:
    if not error_reason:
        return {"error_code": None, "error_message_ru": None}

    normalized = error_reason.lower()
    if "post_only_rejected" in normalized or "post only" in normalized:
        return {
            "error_code": "POST_ONLY_REJECTED",
            "error_message_ru": "Maker order rejected because it would take resting liquidity",
        }
    if "manual_rejected" in normalized:
        return {
            "error_code": "MANUAL_REJECTED",
            "error_message_ru": "Execution rejected during manual review",
        }
    if "insufficient funds" in normalized:
        return {
            "error_code": "INSUFFICIENT_FUNDS",
            "error_message_ru": "Недостаточно средств на балансе или allowance",
        }
    below_minimum_markers = (
        "invalid amount",
        "min size",
        "minimum order",
        "below minimum",
        "order size too small",
        "order_size_below_minimum",
        "minimum_shares=",
        "lower than the minimum",
    )
    if any(marker in normalized for marker in below_minimum_markers):
        return {
            "error_code": "ORDER_BELOW_MINIMUM",
            "error_message_ru": (
                "Количество токенов меньше минимума Polymarket (5). "
                "Для текущей цены нужен больший бюджет."
            ),
        }
    if "max_slippage" in normalized or "slippage" in normalized:
        return {
            "error_code": "SLIPPAGE_EXCEEDED",
            "error_message_ru": "Превышено допустимое проскальзывание (slippage)",
        }
    if "market closed" in normalized or "market is closed" in normalized:
        return {
            "error_code": "MARKET_CLOSED",
            "error_message_ru": "Рынок уже закрыт или разрешен",
        }

    return {"error_code": "UNKNOWN_ERROR", "error_message_ru": error_reason}


async def serialize_execution_requests(
    db: AsyncSession,
    requests: list[ExecutionRequest],
) -> list[dict]:
    from sqlalchemy import select
    from polyflip.db.execution_models import ExecutionAttempt

    manual_review_requests = [
        r for r in requests if r.state == "MANUAL_REVIEW_REQUIRED"
    ]
    reconcilable_requests = [
        r
        for r in requests
        if r.state in RECONCILABLE_REQUEST_STATES or r.state == "MANUAL_REVIEW_REQUIRED"
    ]

    eligibility = await evaluate_no_fill_eligibility_batch(db, manual_review_requests)

    evidence_by_req = {}
    evidence_req_ids = [r.id for r in reconcilable_requests]
    if evidence_req_ids:
        rows = (
            await db.execute(
                select(
                    ExecutionAttempt.request_id,
                    ExecutionAttempt.provider_order_id,
                    ExecutionAttempt.transaction_hashes,
                ).where(ExecutionAttempt.request_id.in_(evidence_req_ids))
            )
        ).all()
        for r_id, p_id, tx_hashes in rows:
            if r_id not in evidence_by_req:
                evidence_by_req[r_id] = {"has_order_id": False, "has_tx_hash": False}
            if p_id is not None:
                evidence_by_req[r_id]["has_order_id"] = True
            if tx_hashes and len(tx_hashes) > 0:
                evidence_by_req[r_id]["has_tx_hash"] = True

    source_requests = {}
    source_ids = [
        r.source_paper_request_id for r in requests if r.source_paper_request_id
    ]
    if source_ids:
        source_rows = (
            (
                await db.execute(
                    select(ExecutionRequest).where(ExecutionRequest.id.in_(source_ids))
                )
            )
            .scalars()
            .all()
        )
        source_requests = {row.id: row for row in source_rows}
    results = []
    for req in requests:
        actions = []
        if req.id in eligibility and eligibility[req.id].allowed:
            actions.append("MARK_FAILED_NO_FILL")

        if req.id in evidence_by_req and (
            evidence_by_req[req.id]["has_order_id"]
            or evidence_by_req[req.id]["has_tx_hash"]
        ):
            actions.append("RECONCILE_WITH_POLYMARKET")

        results.append(
            {
                "id": str(req.id),
                "trade_history_id": req.trade_history_id,
                "intent": req.intent,
                "trigger_reason": req.trigger_reason,
                "market_id": req.market_id,
                "asset": req.asset,
                "outcome_to_buy": req.outcome_to_buy,
                "state": req.state,
                "requested_mode": req.requested_mode,
                "requested_shares": (
                    float(req.requested_shares) if req.requested_shares else None
                ),
                "limit_price": float(req.limit_price) if req.limit_price else None,
                "max_acceptable_price": (
                    float(req.max_acceptable_price)
                    if req.max_acceptable_price is not None
                    else None
                ),
                "target_amount_usdc": (
                    float(req.target_amount_usdc) if req.target_amount_usdc else None
                ),
                "filled_shares": float(req.filled_shares) if req.filled_shares else 0,
                "filled_cost_usdc": (
                    float(req.filled_cost_usdc) if req.filled_cost_usdc else 0
                ),
                "execution_order_mode": req.execution_order_mode,
                "post_only": bool(req.post_only),
                "decision_price": req.decision_price,
                "paper_price": (
                    float(source_requests[req.source_paper_request_id].limit_price)
                    if req.source_paper_request_id in source_requests
                    and source_requests[req.source_paper_request_id].limit_price
                    is not None
                    else None
                ),
                "release_quote_price": req.release_quote_price,
                "release_quote_at": (
                    req.release_quote_at.isoformat() if req.release_quote_at else None
                ),
                "submit_quote_price": req.submit_quote_price,
                "submit_quote_at": (
                    req.submit_quote_at.isoformat() if req.submit_quote_at else None
                ),
                "submitted_limit_price": req.submitted_limit_price,
                "paper_to_live_release_delay_sec": (
                    (
                        req.created_at
                        - source_requests[req.source_paper_request_id].created_at
                    ).total_seconds()
                    if req.source_paper_request_id in source_requests
                    and req.created_at
                    and source_requests[req.source_paper_request_id].created_at
                    else None
                ),
                "paper_to_live_submit_delay_sec": (
                    (
                        req.submit_quote_at
                        - source_requests[req.source_paper_request_id].created_at
                    ).total_seconds()
                    if req.source_paper_request_id in source_requests
                    and req.submit_quote_at
                    and source_requests[req.source_paper_request_id].created_at
                    else None
                ),
                # Preserve the legacy field while making it reflect the
                # actual submission delay when quote telemetry is available.
                "paper_to_live_delay_sec": (
                    (
                        req.submit_quote_at
                        - source_requests[req.source_paper_request_id].created_at
                    ).total_seconds()
                    if req.source_paper_request_id in source_requests
                    and req.submit_quote_at
                    and source_requests[req.source_paper_request_id].created_at
                    else (
                        (
                            req.created_at
                            - source_requests[req.source_paper_request_id].created_at
                        ).total_seconds()
                        if req.source_paper_request_id in source_requests
                        and req.created_at
                        and source_requests[req.source_paper_request_id].created_at
                        else None
                    )
                ),
                "cancel_due_at": (
                    req.cancel_due_at.isoformat() if req.cancel_due_at else None
                ),
                "terminal_code": req.terminal_code,
                "network_retry_count": req.network_retry_count,
                "ttl_seconds": req.ttl_seconds,
                "created_at": req.created_at.isoformat() if req.created_at else None,
                "updated_at": req.updated_at.isoformat() if req.updated_at else None,
                "error_reason": req.error_reason,
                "error_details": _parse_error(req.error_reason),
                "available_actions": actions,
                "can_mark_no_fill": (
                    req.id in eligibility and eligibility[req.id].allowed
                ),
                "review_blockers": list(
                    eligibility[req.id].blockers if req.id in eligibility else ()
                ),
            }
        )
    return results
