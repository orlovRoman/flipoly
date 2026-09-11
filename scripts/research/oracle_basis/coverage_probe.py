"""Coverage probe for the first real oracle-basis export (read-only).

Checks market_snapshots, underlying_observations, live_markets, and
orderbook_depth_snapshots over 2026-09-09..10 UTC: schemas, row counts,
per-asset breakdown, outcome/strike availability, and L2 quality. Prints a JSON
report; exit 2 with DATA_INSUFFICIENT when the database is unreachable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any

# Project root on sys.path (repo convention for standalone scripts).
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)

DAY_START = "2026-09-09T00:00:00+00:00"
DAY_END = "2026-09-11T00:00:00+00:00"

TABLES = (
    "market_snapshots",
    "underlying_observations",
    "live_markets",
    "orderbook_depth_snapshots",
)
TS_CANDIDATES = ("recorded_at", "received_at", "event_at", "market_timestamp")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe Sept 9-10 coverage for oracle-basis export."
    )
    parser.add_argument(
        "--out", default=None, help="Optional path for the JSON report."
    )
    return parser


async def _columns(session, table: str) -> list[dict[str, str]]:
    from sqlalchemy import text

    rows = (
        await session.execute(
            text(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name = :t ORDER BY ordinal_position"
            ),
            {"t": table},
        )
    ).all()
    return [{"name": r[0], "type": r[1]} for r in rows]


async def _count(
    session, table: str, where: str = "", params: dict[str, Any] | None = None
) -> int:
    from sqlalchemy import text

    row = (
        await session.execute(
            text(f"SELECT COUNT(*) FROM {table} {where}"), params or {}
        )
    ).first()
    return int(row[0])


async def _group(
    session, table: str, column: str, where: str, params: dict[str, Any]
) -> dict[str, int]:
    from sqlalchemy import text

    rows = (
        await session.execute(
            text(
                f"SELECT {column}, COUNT(*) FROM {table} {where} GROUP BY 1 ORDER BY 2 DESC"
            ),
            params,
        )
    ).all()
    return {str(r[0]): int(r[1]) for r in rows}


async def probe() -> dict[str, Any]:
    from sqlalchemy import text

    from polyflip.db.connection import async_session

    report: dict[str, Any] = {
        "window": {"from": DAY_START, "to": DAY_END},
        "tables": {},
    }
    async with async_session() as session:
        # Read-only migration-state check for the prod deploy review (item 1):
        # report the server's alembic_version without touching anything.
        try:
            version = (
                await session.execute(text("SELECT version_num FROM alembic_version"))
            ).first()
            report["alembic_version"] = version[0] if version else None
        except Exception as exc:  # noqa: BLE001 - probe reports, never crashes
            report["alembic_version"] = None
            report["alembic_version_error"] = str(exc)[:200]
        for table in TABLES:
            cols = await _columns(session, table)
            names = {c["name"] for c in cols}
            entry: dict[str, Any] = {"exists": bool(cols), "columns": cols}
            if not cols:
                report["tables"][table] = entry
                continue
            entry["total_rows"] = await _count(session, table)
            ts_col = next((c for c in TS_CANDIDATES if c in names), None)
            entry["time_column"] = ts_col
            if ts_col and table != "live_markets":
                window = f"WHERE {ts_col} >= :a AND {ts_col} < :b"
                params = {"a": DAY_START, "b": DAY_END}
                entry["window_rows"] = await _count(session, table, window, params)
                for day, nxt in (
                    ("2026-09-09", "2026-09-10"),
                    ("2026-09-10", "2026-09-11"),
                ):
                    entry[f"rows_{day}"] = await _count(
                        session,
                        table,
                        f"WHERE {ts_col} >= :a AND {ts_col} < :b",
                        {"a": f"{day}T00:00:00+00:00", "b": f"{nxt}T00:00:00+00:00"},
                    )
            report["tables"][table] = entry

        snaps = report["tables"]["market_snapshots"]
        if snaps["exists"] and snaps.get("time_column"):
            window = (
                f"WHERE {snaps['time_column']} >= :a AND {snaps['time_column']} < :b"
            )
            params = {"a": DAY_START, "b": DAY_END}
            names = {c["name"] for c in snaps["columns"]}
            if "asset" in names:
                snaps["by_asset"] = await _group(
                    session, "market_snapshots", "asset", window, params
                )
            if "final_outcome" in names:
                snaps["by_outcome"] = await _group(
                    session, "market_snapshots", "final_outcome", window, params
                )
            if "strike_value" in names:
                snaps["strike_nonnull"] = await _count(
                    session,
                    "market_snapshots",
                    window + " AND strike_value IS NOT NULL",
                    params,
                )
            if "strike_source" in names:
                snaps["by_strike_source"] = await _group(
                    session, "market_snapshots", "strike_source", window, params
                )
            if "market_id" in names:
                snaps["distinct_markets"] = (
                    await session.execute(
                        text(
                            f"SELECT COUNT(DISTINCT market_id) FROM market_snapshots {window}"
                        ),
                        params,
                    )
                ).first()[0]

        und = report["tables"]["underlying_observations"]
        if und["exists"]:
            names = {c["name"] for c in und["columns"]}
            if "instrument" in names and "source" in names:
                rows = (
                    await session.execute(
                        text(
                            "SELECT instrument, source, COUNT(*) FROM underlying_observations "
                            "WHERE received_at >= :a AND received_at < :b "
                            "GROUP BY 1, 2 ORDER BY 3 DESC"
                        ),
                        {"a": DAY_START, "b": DAY_END},
                    )
                ).all()
                und["by_instrument_source"] = [
                    {"instrument": r[0], "source": r[1], "rows": int(r[2])}
                    for r in rows
                ]

        live = report["tables"]["live_markets"]
        if live["exists"]:
            names = {c["name"] for c in live["columns"]}
            if "end_time_est" in names:
                live["ending_in_window"] = await _count(
                    session,
                    "live_markets",
                    "WHERE end_time_est >= :a AND end_time_est < :b",
                    {"a": DAY_START, "b": DAY_END},
                )
            if "asset" in names:
                live["by_asset_total"] = await _group(
                    session, "live_markets", "asset", "", {}
                )

        depth = report["tables"]["orderbook_depth_snapshots"]
        if depth["exists"] and depth.get("time_column"):
            window = (
                f"WHERE {depth['time_column']} >= :a AND {depth['time_column']} < :b"
            )
            params = {"a": DAY_START, "b": DAY_END}
            names = {c["name"] for c in depth["columns"]}
            if "quality_status" in names:
                depth["by_quality"] = await _group(
                    session,
                    "orderbook_depth_snapshots",
                    "quality_status",
                    window,
                    params,
                )
            if "outcome_side" in names:
                depth["by_side"] = await _group(
                    session, "orderbook_depth_snapshots", "outcome_side", window, params
                )
            if "market_id" in names:
                depth["distinct_markets"] = (
                    await session.execute(
                        text(
                            f"SELECT COUNT(DISTINCT market_id) FROM orderbook_depth_snapshots {window}"
                        ),
                        params,
                    )
                ).first()[0]
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = asyncio.run(probe())
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "DATA_INSUFFICIENT",
                    "reason": f"database unreachable: {exc}",
                },
                indent=2,
            )
        )
        return 2
    print(json.dumps(report, indent=2, default=str))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
