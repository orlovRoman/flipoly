import argparse
import asyncio
import json
import uuid
from sqlalchemy import text
from polyflip.db.connection import async_session

async def reconstruct_reservations(apply: bool):
    async with async_session() as session:
        async with session.begin():
            # Apply advisory lock to prevent concurrent modifications
            await session.execute(text("SELECT pg_advisory_xact_lock(1001)"))
            
            # Step 1: Link trade_history_id
            update_trade_id_sql = """
            UPDATE exposure_reservations r
            SET trade_history_id = r.trade_id::integer
            WHERE r.trade_history_id IS NULL
              AND r.trade_id ~ '^[0-9]+$'
              AND EXISTS (
                  SELECT 1
                  FROM trade_history th
                  WHERE th.id = r.trade_id::integer
              )
            RETURNING id;
            """
            
            if apply:
                res = await session.execute(text(update_trade_id_sql))
                linked_trade_ids_count = len(res.fetchall())
            else:
                # Dry run for trade_id linkage
                check_sql = """
                SELECT count(*)
                FROM exposure_reservations r
                WHERE r.trade_history_id IS NULL
                  AND r.trade_id ~ '^[0-9]+$'
                  AND EXISTS (
                      SELECT 1
                      FROM trade_history th
                      WHERE th.id = r.trade_id::integer
                  )
                """
                res = await session.execute(text(check_sql))
                linked_trade_ids_count = res.scalar()
                
            # Step 2: Check for ambiguous request mapping
            check_ambiguous_sql = """
            SELECT r.id, COUNT(er.id) AS matches
            FROM exposure_reservations r
            LEFT JOIN execution_requests er
              ON er.intent = 'OPEN'
             AND er.trade_history_id = r.trade_history_id
             AND er.idempotency_key = 'OPEN:' || r.trade_history_id::text
            WHERE r.trade_history_id IS NOT NULL
            GROUP BY r.id
            HAVING COUNT(er.id) <> 1
            """
            res = await session.execute(text(check_ambiguous_sql))
            ambiguous_rows = res.fetchall()
            
            if ambiguous_rows:
                print(f"ERROR: Found {len(ambiguous_rows)} reservations with ambiguous/missing requests.")
                for row in ambiguous_rows:
                    print(row)
                if apply:
                    raise RuntimeError("Ambiguous rows detected. Aborting reconstruction.")
            
            # Step 3: Link request_id
            update_request_id_sql = """
            UPDATE exposure_reservations r
            SET request_id = er.id
            FROM execution_requests er
            WHERE r.request_id IS NULL
              AND r.trade_history_id IS NOT NULL
              AND er.intent = 'OPEN'
              AND er.trade_history_id = r.trade_history_id
              AND er.idempotency_key = 'OPEN:' || r.trade_history_id::text
            RETURNING r.id;
            """
            
            if apply:
                res = await session.execute(text(update_request_id_sql))
                linked_request_ids_count = len(res.fetchall())
            else:
                check_sql2 = """
                SELECT count(*)
                FROM exposure_reservations r
                JOIN execution_requests er
                  ON er.intent = 'OPEN'
                 AND er.trade_history_id = r.trade_history_id
                 AND er.idempotency_key = 'OPEN:' || r.trade_history_id::text
                WHERE r.request_id IS NULL
                  AND r.trade_history_id IS NOT NULL
                """
                res = await session.execute(text(check_sql2))
                linked_request_ids_count = res.scalar()

            # Step 4: Final checks
            unrecoverable_sql = """
            SELECT count(*)
            FROM exposure_reservations
            WHERE request_id IS NULL
               OR trade_history_id IS NULL
               OR market_id IS NULL
               OR expires_at IS NULL
            """
            res = await session.execute(text(unrecoverable_sql))
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
