import argparse
import asyncio
import json
import uuid
from sqlalchemy import text, select, update, func, or_
from polyflip.db.connection import async_session
from polyflip.db.execution_models import ExecutionRequest, ExposureReservation
from polyflip.db.models import TradeHistory

async def reconstruct_reservations(apply: bool):
    async with async_session() as session:
        async with session.begin():
            # Step 1: Link trade_history_id
            # SQLite does not have regex easily available in UPDATE, so we fetch all rows and update them.
            res = await session.execute(
                select(ExposureReservation).where(ExposureReservation.trade_history_id == None)
            )
            unlinked = res.scalars().all()
            linked_trade_ids_count = 0
            for r in unlinked:
                if r.trade_id and r.trade_id.isdigit():
                    trade = await session.get(TradeHistory, int(r.trade_id))
                    if trade:
                        if apply:
                            r.trade_history_id = trade.id
                        linked_trade_ids_count += 1
            
            if apply:
                await session.flush()
                
            # Step 2: Check for ambiguous request mapping
            res = await session.execute(
                select(ExposureReservation.id, func.count(ExecutionRequest.id).label("matches"))
                .select_from(ExposureReservation)
                .outerjoin(ExecutionRequest, 
                           (ExecutionRequest.intent == 'OPEN') &
                           (ExecutionRequest.trade_history_id == ExposureReservation.trade_history_id) &
                           (ExecutionRequest.idempotency_key == 'OPEN:' + func.cast(ExposureReservation.trade_history_id, text("VARCHAR"))))
                .where(ExposureReservation.trade_history_id != None)
                .group_by(ExposureReservation.id)
                .having(func.count(ExecutionRequest.id) != 1)
            )
            ambiguous_rows = res.fetchall()
            
            if ambiguous_rows:
                print(f"ERROR: Found {len(ambiguous_rows)} reservations with ambiguous/missing requests.")
                for row in ambiguous_rows:
                    print(row)
                if apply:
                    raise RuntimeError("Ambiguous rows detected. Aborting reconstruction.")
            
            # Step 3: Link request_id
            res = await session.execute(
                select(ExposureReservation, ExecutionRequest.id)
                .join(ExecutionRequest,
                           (ExecutionRequest.intent == 'OPEN') &
                           (ExecutionRequest.trade_history_id == ExposureReservation.trade_history_id) &
                           (ExecutionRequest.idempotency_key == 'OPEN:' + func.cast(ExposureReservation.trade_history_id, text("VARCHAR"))))
                .where(ExposureReservation.request_id == None)
                .where(ExposureReservation.trade_history_id != None)
            )
            to_link_requests = res.all()
            linked_request_ids_count = 0
            for r, req_id in to_link_requests:
                if apply:
                    r.request_id = req_id
                linked_request_ids_count += 1
                
            if apply:
                await session.flush()

            # Step 4: Final checks
            res = await session.execute(
                select(func.count(ExposureReservation.id))
                .where(
                    or_(
                        ExposureReservation.request_id == None,
                        ExposureReservation.trade_history_id == None,
                        ExposureReservation.market_id == None,
                        ExposureReservation.expires_at == None
                    )
                )
            )
            unrecoverable_count = res.scalar()

            report = {
                "linked_trade_history_ids": linked_trade_ids_count,
                "linked_request_ids": linked_request_ids_count,
                "ambiguous_rows": len(ambiguous_rows),
                "unrecoverable_rows_remaining": unrecoverable_count,
                "mode": "apply" if apply else "dry-run"
            }
            
            print(json.dumps(report, indent=2))
            
            if not apply:
                await session.rollback()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct exposure_reservations after migration.")
    parser.add_argument("--apply", action="store_true", help="Apply the changes to the database.")
    args = parser.parse_args()
    
    asyncio.run(reconstruct_reservations(args.apply))
