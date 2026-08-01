import re

def fix_api():
    with open('polyflip/api/execution_api.py', 'r', encoding='utf-8') as f:
        content = f.read()

    old_logic = """    if req is None:
        raise HTTPException(404, "Заявка не найдена")

    provider_evidence = await db.scalar(
        select(func.count(ExecutionAttempt.id)).where(
            ExecutionAttempt.request_id == request_id,
            ExecutionAttempt.provider_order_id.is_not(None),
        )
    )

    if not provider_evidence:
        raise HTTPException(
            422,
            "Нет provider_order_id — сверка с Polymarket невозможна",
        )

    req.state = "RECONCILING"
    req.updated_at = datetime.now(timezone.utc)
    await db.commit()

    return {"request_id": str(req.id), "state": "RECONCILING"}"""

    new_logic = """    if req is None:
        raise HTTPException(404, "Заявка не найдена")

    if req.requested_mode != "LIVE":
        raise HTTPException(409, "Сверка разрешена только для LIVE-заявок")

    if req.state == "RECONCILING":
        return {
            "request_id": str(req.id),
            "state": "RECONCILING",
            "idempotent": True,
        }

    allowed_states = {
        "SUBMITTING",
        "ACCEPTED",
        "UNKNOWN",
        "PARTIALLY_FILLED",
        "RECONCILING",
        "MANUAL_REVIEW_REQUIRED",
    }

    if req.state not in allowed_states:
        raise HTTPException(
            409,
            f"Заявку в статусе {req.state} нельзя переводить в RECONCILING",
        )

    provider_evidence = await db.scalar(
        select(func.count(ExecutionAttempt.id)).where(
            ExecutionAttempt.request_id == request_id,
            ExecutionAttempt.provider_order_id.is_not(None),
        )
    )

    if not provider_evidence:
        raise HTTPException(
            422,
            "Нет provider_order_id — сверка с Polymarket невозможна",
        )

    req.state = "RECONCILING"
    req.updated_at = datetime.now(timezone.utc)
    await db.commit()

    return {"request_id": str(req.id), "state": "RECONCILING"}"""

    content = content.replace(old_logic, new_logic)

    with open('polyflip/api/execution_api.py', 'w', encoding='utf-8') as f:
        f.write(content)

def fix_serializers():
    with open('polyflip/execution/serializers.py', 'r', encoding='utf-8') as f:
        content = f.read()

    old_parse = """    if "minimum order size" in normalized or "below minimum" in normalized:
        return {"error_code": "ORDER_BELOW_MINIMUM", "error_message_ru": "Сумма заявки меньше минимальной суммы Polymarket ($0.50)"}"""
    
    new_parse = """    below_minimum_markers = (
        "invalid amount",
        "min size",
        "minimum order",
        "below minimum",
        "order size too small",
    )
    if any(marker in normalized for marker in below_minimum_markers):
        return {
            "error_code": "ORDER_BELOW_MINIMUM",
            "error_message_ru": (
                "Сумма ордера ниже допустимого минимума. "
                "Для LIVE используется безопасный минимум 1.10 USDC"
            ),
        }"""
    
    content = content.replace(old_parse, new_parse)

    # Now for available actions. Currently it looks something like this:
    # "error_details": _parse_error(req.error_reason),
    # "available_actions": ["RECONCILE_WITH_POLYMARKET"] if req.state in RECONCILABLE_REQUEST_STATES else [],
    # ... and further down:
    # "available_actions": ["MARK_FAILED_NO_FILL"] if eligibility[req.id].allowed else [],
    # Let's replace the whole async def serialize_execution_requests
    
    # We will regex replace the whole function content since it's short.
    # Actually, let's just find the `return [...]` array comprehension and rewrite it.
    
    start_str = "    return ["
    end_str = "    ]"
    
    start_idx = content.find(start_str)
    end_idx = content.find(end_str, start_idx) + len(end_str)
    
    # We need to build the list manually instead of a list comprehension to calculate actions properly.
    
    new_func_body = """    results = []
    
    # Pre-calculate provider evidence
    req_ids = [r.id for r in requests]
    provider_evidence_request_ids = set()
    if req_ids:
        rows = await db.execute(
            select(ExecutionAttempt.request_id)
            .where(ExecutionAttempt.request_id.in_(req_ids))
            .where(ExecutionAttempt.provider_order_id.is_not(None))
            .distinct()
        )
        provider_evidence_request_ids = set(rows.scalars().all())
    
    for req in requests:
        actions = []
        if eligibility[req.id].allowed:
            actions.append("MARK_FAILED_NO_FILL")
            
        if (
            req.state in RECONCILABLE_REQUEST_STATES
            or req.state == "MANUAL_REVIEW_REQUIRED"
        ) and req.id in provider_evidence_request_ids:
            actions.append("RECONCILE_WITH_POLYMARKET")
            
        results.append({
            "id": str(req.id),
            "asset": req.asset,
            "outcome_to_buy": req.outcome_to_buy,
            "outcome_bought": req.outcome_bought,
            "market_id": req.market_id,
            "intent": req.intent,
            "requested_mode": req.requested_mode,
            "state": req.state,
            "target_amount_usdc": req.target_amount_usdc,
            "created_at": req.created_at,
            "error_reason": req.error_reason,
            "error_details": _parse_error(req.error_reason),
            "available_actions": actions,
            "can_mark_no_fill": eligibility[req.id].allowed,
            "review_blockers": eligibility[req.id].blockers,
        })
    return results"""
    
    old_func_body = content[start_idx:end_idx]
    
    # Wait, the current func has provider_evidence checking? No, it only checks `eligibility = await check_no_fill_batch(db, [r.id for r in requests])`
    
    # Let's replace the whole async def serialize_execution_requests body
    func_start = content.find("async def serialize_execution_requests(db: AsyncSession, requests: list[ExecutionRequest]):")
    if func_start != -1:
        # replace from func_start to end of file, assuming it's the last function
        old_body = content[func_start:]
        new_body = """async def serialize_execution_requests(db: AsyncSession, requests: list[ExecutionRequest]):
    from polyflip.execution.manual_review import check_no_fill_batch
    from polyflip.db.execution_models import ExecutionAttempt
    from sqlalchemy import select
    
    if not requests:
        return []

    eligibility = await check_no_fill_batch(db, [r.id for r in requests])
    
    req_ids = [r.id for r in requests]
    provider_evidence_request_ids = set()
    if req_ids:
        rows = await db.execute(
            select(ExecutionAttempt.request_id)
            .where(ExecutionAttempt.request_id.in_(req_ids))
            .where(ExecutionAttempt.provider_order_id.is_not(None))
            .distinct()
        )
        provider_evidence_request_ids = set(rows.scalars().all())

    results = []
    for req in requests:
        actions = []
        if eligibility[req.id].allowed:
            actions.append("MARK_FAILED_NO_FILL")
            
        if (
            req.state in RECONCILABLE_REQUEST_STATES
            or req.state == "MANUAL_REVIEW_REQUIRED"
        ) and req.id in provider_evidence_request_ids:
            actions.append("RECONCILE_WITH_POLYMARKET")
            
        results.append({
            "id": str(req.id),
            "asset": req.asset,
            "outcome_to_buy": req.outcome_to_buy,
            "outcome_bought": req.outcome_bought,
            "market_id": req.market_id,
            "intent": req.intent,
            "requested_mode": req.requested_mode,
            "state": req.state,
            "target_amount_usdc": req.target_amount_usdc,
            "created_at": req.created_at,
            "error_reason": req.error_reason,
            "error_details": _parse_error(req.error_reason),
            "available_actions": actions,
            "can_mark_no_fill": eligibility[req.id].allowed,
            "review_blockers": eligibility[req.id].blockers,
        })
    return results
"""
        content = content[:func_start] + new_body
    
    with open('polyflip/execution/serializers.py', 'w', encoding='utf-8') as f:
        f.write(content)


def fix_html():
    with open('polyflip/templates/execution.html', 'r', encoding='utf-8') as f:
        content = f.read()

    # The 10s interval is there, but there is also a 5s interval still.
    # The user says: "В результате API и readiness опрашиваются примерно в три раза чаще необходимого. Второй таймер удалить. Оставить один интервал 10 секунд."
    
    # We should look for setInterval and remove the 5000 one.
    import re
    
    # Just in case my previous regex missed it
    interval_5s_regex = re.compile(r'setInterval\([^)]+\);\s*}, 5000\);', re.DOTALL)
    content = interval_5s_regex.sub('', content)

    # Let's do a more robust regex if the above didn't catch it
    interval_5s_regex2 = re.compile(r'setInterval\(\(\) => \{[^}]*loadLiveDashboard\(\);[^}]*\}, 5000\);', re.DOTALL)
    content = interval_5s_regex2.sub('', content)
    
    interval_5s_regex3 = re.compile(r'// Auto-refresh readiness\s*setInterval\(\(\) => \{\s*if \(currentSessionId && document.getElementById\(\'readiness-panel\'\).style.display !== \'none\'\) \{\s*loadLiveDashboard\(\);\s*\}\s*\}, 5000\);', re.DOTALL)
    content = interval_5s_regex3.sub('', content)

    with open('polyflip/templates/execution.html', 'w', encoding='utf-8') as f:
        f.write(content)

def fix_whitespace():
    # Fix trailing whitespace and trailing blank lines in tests/test_release_gate.py
    with open('tests/test_release_gate.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    while lines and not lines[-1].strip():
        lines.pop()
    
    with open('tests/test_release_gate.py', 'w', encoding='utf-8') as f:
        f.write('\n'.join(line.rstrip(' \t\r') for line in lines) + '\n')

if __name__ == "__main__":
    fix_api()
    fix_serializers()
    fix_html()
    fix_whitespace()
