import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import APIRouter, Depends, HTTPException
import json
import sys

def modify_api():
    with open('polyflip/api/execution_api.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    code = """
@router.post("/requests/{request_id}/reconcile")
async def reconcile_request(
    request_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
):
    req = await db.scalar(
        select(ExecutionRequest)
        .where(ExecutionRequest.id == request_id)
        .with_for_update()
    )

    if req is None:
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

    return {"request_id": str(req.id), "state": "RECONCILING"}
"""
    with open('polyflip/api/execution_api.py', 'w', encoding='utf-8') as f:
        f.write(content + "\n" + code)

if __name__ == "__main__":
    modify_api()
