import datetime
from decimal import Decimal
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from polyflip.research.lp_rewards.protocol import LPProtocol, load_protocol

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("audit_quality")

REQUIRED_L2_COLUMNS = {
    "timestamp_ns",
    "condition_id",
    "asset_id",
    "side",
    "price",
    "size",
    "valid_from_ns",
    "valid_to_ns",
    "order_age_sec",
}


def audit_l2_parquet(
    file_path: Path,
    now_ns: Optional[int] = None,
    clock_skew_tolerance_ns: int = 60_000_000_000,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Audit a single L2 parquet file for schema, price bounds, monotonic timestamps, and valid sizes."""
    violations = []
    stats: Dict[str, Any] = {
        "rows": 0,
        "unique_assets": 0,
        "min_price": None,
        "max_price": None,
        "duplicates": 0,
        "dates": set(),
    }

    if now_ns is None:
        now_ns = time.time_ns()

    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        violations.append(f"Unreadable parquet file {file_path}: {e}")
        return False, violations, stats

    stats["rows"] = len(df)
    if len(df) == 0:
        violations.append(f"Empty L2 parquet file: {file_path}")
        return False, violations, stats

    # 1. Schema check
    missing_cols = REQUIRED_L2_COLUMNS - set(df.columns)
    if missing_cols:
        violations.append(f"Missing required columns in {file_path.name}: {sorted(missing_cols)}")
        return False, violations, stats

    # 2. Price and Size validation
    try:
        prices = df["price"].astype(float)
        sizes = df["size"].astype(float)
    except Exception as e:
        violations.append(f"Failed to cast price/size to float in {file_path.name}: {e}")
        return False, violations, stats

    stats["min_price"] = float(prices.min())
    stats["max_price"] = float(prices.max())

    if (prices <= 0.0).any() or (prices >= 1.0).any():
        bad_prices = prices[(prices <= 0.0) | (prices >= 1.0)].tolist()[:5]
        violations.append(f"Prices out of valid probability range (0.0, 1.0) in {file_path.name}: {bad_prices}")

    if (sizes <= 0.0).any():
        bad_sizes = sizes[sizes <= 0.0].tolist()[:5]
        violations.append(f"Non-positive order sizes detected in {file_path.name}: {bad_sizes}")

    # 3. Timestamp sanity and monotonicity
    try:
        ts = df["timestamp_ns"].astype("int64")
        vf = df["valid_from_ns"].astype("int64")
        vt = df["valid_to_ns"].astype("int64")
    except Exception as e:
        violations.append(f"Failed to parse integer timestamps in {file_path.name}: {e}")
        return False, violations, stats

    if (ts <= 0).any():
        bad_ts = ts[ts <= 0].tolist()[:5]
        violations.append(f"Non-positive timestamps detected in {file_path.name}: {bad_ts}")

    if (vf <= 0).any():
        bad_vf = vf[vf <= 0].tolist()[:5]
        violations.append(f"Non-positive valid_from_ns detected in {file_path.name}: {bad_vf}")

    if (vt <= 0).any():
        bad_vt = vt[vt <= 0].tolist()[:5]
        violations.append(f"Non-positive valid_to_ns detected in {file_path.name}: {bad_vt}")

    if (vf > vt).any():
        violations.append(f"valid_from_ns > valid_to_ns detected in {file_path.name}")

    if (ts > (now_ns + clock_skew_tolerance_ns)).any():
        future_ts = ts[ts > (now_ns + clock_skew_tolerance_ns)].tolist()[:3]
        violations.append(f"Future timestamps detected in {file_path.name}: {future_ts} > {now_ns}")

    # Calendar coverage strictly by timestamp_ns for valid positive timestamps
    valid_ts = ts[ts > 0]
    if len(valid_ts) > 0:
        unique_days = (valid_ts // 86_400_000_000_000).unique()
        stats["dates"] = {
            time.strftime("%Y-%m-%d", time.gmtime(int(d) * 86400))
            for d in unique_days
        }

    # Check monotonicity per asset stream
    for asset_id, group in df.groupby("asset_id"):
        asset_ts = group["timestamp_ns"].astype("int64")
        if not asset_ts.is_monotonic_increasing:
            violations.append(f"Non-monotonic timestamps for asset {asset_id} in {file_path.name}")

    # 4. Duplicate levels check (same timestamp, asset, side, price)
    dup_mask = df.duplicated(subset=["timestamp_ns", "asset_id", "side", "price"], keep=False)
    stats["duplicates"] = int(dup_mask.sum())
    if stats["duplicates"] > 0:
        violations.append(f"Unaggregated duplicate orderbook levels found in {file_path.name}: {stats['duplicates']} rows")

    is_valid = len(violations) == 0
    return is_valid, violations, stats


def audit_trade_parquet(
    file_path: Path,
    now_ns: Optional[int] = None,
    clock_skew_tolerance_ns: int = 60_000_000_000,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Audit a public trades parquet file."""
    violations = []
    stats: Dict[str, Any] = {"rows": 0}

    if now_ns is None:
        now_ns = time.time_ns()

    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        violations.append(f"Unreadable trade parquet file {file_path}: {e}")
        return False, violations, stats

    stats["rows"] = len(df)
    if len(df) == 0:
        return True, violations, stats

    req_cols = {"price", "size", "side", "asset_id"}
    missing = req_cols - set(df.columns)
    if missing:
        violations.append(f"Missing columns in trade file {file_path.name}: {sorted(missing)}")
        return False, violations, stats

    try:
        prices = df["price"].astype(float)
        sizes = df["size"].astype(float)
    except Exception as e:
        violations.append(f"Non-numeric trade price/size in {file_path.name}: {e}")
        return False, violations, stats

    if (prices <= 0.0).any() or (prices >= 1.0).any():
        violations.append(f"Trade prices out of valid range (0.0, 1.0) in {file_path.name}")

    if (sizes <= 0.0).any():
        violations.append(f"Non-positive trade sizes in {file_path.name}")

    ts_col = "timestamp_ns" if "timestamp_ns" in df.columns else "observed_at_ns"
    if ts_col in df.columns:
        ts = df[ts_col].astype("int64")
        if (ts <= 0).any():
            violations.append(f"Non-positive trade timestamps in {file_path.name}")
        if (ts > (now_ns + clock_skew_tolerance_ns)).any():
            violations.append(f"Future trade timestamps in {file_path.name}")

    is_valid = len(violations) == 0
    return is_valid, violations, stats


def audit_daily_evaluations(
    eval_dir: Path,
    expected_protocol_hash: str,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Audit daily evaluations JSON records."""
    violations = []
    eval_files = sorted(eval_dir.glob("*.json")) if eval_dir.exists() else []
    stats = {
        "files_count": len(eval_files),
        "dates": [],
        "total_quote_hours": Decimal("0.0"),
        "total_simulated_quote_hours": Decimal("0.0"),
        "total_actual_quote_hours": Decimal("0.0"),
    }

    dates_seen = set()
    for f in eval_files:
        try:
            with open(f, "r", encoding="utf-8") as jf:
                data = json.load(jf)
        except Exception as e:
            violations.append(f"Corrupted daily evaluation JSON {f.name}: {e}")
            continue

        rec_date = data.get("date")
        if not rec_date:
            violations.append(f"Missing date in {f.name}")
        elif rec_date in dates_seen:
            violations.append(f"Duplicate evaluation date {rec_date} in {f.name}")
        else:
            dates_seen.add(rec_date)
            stats["dates"].append(rec_date)

        rec_hash = data.get("protocol_hash")
        if not rec_hash or rec_hash != expected_protocol_hash:
            violations.append(f"Protocol hash mismatch in {f.name}: {rec_hash} != {expected_protocol_hash}")

        # Check simulated and actual quote-hours
        sim_val = data.get("simulated_quote_hours")
        if sim_val is None:
            sim_val = data.get("quote_hours")
        if sim_val is None:
            sim_val = "0.0"
        try:
            sim_qh = Decimal(str(sim_val))
            if sim_qh < Decimal("0.0"):
                violations.append(f"Negative quote_hours {sim_qh} in {f.name}")
            elif sim_qh > Decimal("24.01"):
                violations.append(f"quote_hours {sim_qh} exceeds 24h per day in {f.name}")
            stats["total_simulated_quote_hours"] += sim_qh
            stats["total_quote_hours"] += sim_qh
        except Exception as e:
            violations.append(f"Invalid quote_hours in {f.name}: {e}")

        act_val = data.get("actual_quote_hours")
        if act_val is None:
            act_val = "0.0"
        try:
            act_qh = Decimal(str(act_val))
            if act_qh < Decimal("0.0"):
                violations.append(f"Negative actual_quote_hours {act_qh} in {f.name}")
            elif act_qh > Decimal("24.01"):
                violations.append(f"actual_quote_hours {act_qh} exceeds 24h per day in {f.name}")
            stats["total_actual_quote_hours"] += act_qh
        except Exception as e:
            violations.append(f"Invalid actual_quote_hours in {f.name}: {e}")

        pnl_val = data.get("net_pnl")
        if pnl_val is None:
            violations.append(f"Missing net_pnl in {f.name}")
        else:
            try:
                _ = Decimal(str(pnl_val))
            except Exception as e:
                violations.append(f"Invalid net_pnl in {f.name}: {e}")

    is_valid = len(violations) == 0
    return is_valid, violations, stats


def audit_data_quality(
    storage_path: Path,
    protocol: LPProtocol,
    now_ns: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute comprehensive audit across all collected LP Rewards data."""
    l2_dir = storage_path / "l2_snapshots"
    trades_dir = storage_path / "public_trades"
    eval_dir = storage_path / "daily_evaluations"

    all_violations = []

    # 1. Audit L2 Snapshots
    l2_markets = list(l2_dir.glob("*")) if l2_dir.exists() else []
    total_l2_files = 0
    total_l2_rows = 0
    date_coverage: Dict[str, Set[str]] = {}

    for m_dir in l2_markets:
        if not m_dir.is_dir():
            continue
        for p_file in m_dir.glob("*.parquet"):
            total_l2_files += 1
            valid, viols, stats = audit_l2_parquet(p_file, now_ns=now_ns)
            if not valid:
                all_violations.extend(viols)
            total_l2_rows += stats.get("rows", 0)
            date_coverage.setdefault(m_dir.name, set()).update(stats.get("dates", set()))

    # 2. Audit Public Trades
    trade_markets = list(trades_dir.glob("*")) if trades_dir.exists() else []
    total_trade_files = 0
    total_trade_rows = 0

    for m_dir in trade_markets:
        if not m_dir.is_dir():
            continue
        for p_file in m_dir.glob("*.parquet"):
            total_trade_files += 1
            valid, viols, stats = audit_trade_parquet(p_file, now_ns=now_ns)
            if not valid:
                all_violations.extend(viols)
            total_trade_rows += stats.get("rows", 0)

    # 3. Audit Daily Evaluations
    eval_valid, eval_viols, eval_stats = audit_daily_evaluations(eval_dir, protocol.sha256_hash)
    if not eval_valid:
        all_violations.extend(eval_viols)

    all_market_ids = set(date_coverage.keys()) | set(p.name for p in trade_markets if p.is_dir())

    # 4. Coverage calculation against frozen protocol
    coverage_ratios: Dict[str, float] = {}
    for cid in all_market_ids:
        dates = date_coverage.get(cid, set())
        ratio = min(1.0, len(dates) / float(protocol.gates.gate_a.min_calendar_days))
        coverage_ratios[cid] = ratio

    min_cov = min(coverage_ratios.values()) if coverage_ratios else 0.0
    avg_cov = (sum(coverage_ratios.values()) / len(coverage_ratios)) if coverage_ratios else 0.0

    # Determine status
    req_markets = protocol.gates.gate_a.min_active_markets
    req_cov = float(protocol.gates.gate_a.min_market_coverage_ratio)
    req_days = protocol.gates.gate_a.min_calendar_days
    req_quote_hours = Decimal(str(protocol.gates.gate_a.min_quote_hours))

    if all_violations:
        status = "DATA_INTEGRITY_VIOLATION"
    elif len(all_market_ids) >= req_markets and min_cov >= req_cov and len(eval_stats["dates"]) >= req_days and eval_stats["total_simulated_quote_hours"] >= req_quote_hours:
        status = "DATA_QUALITY_VERIFIED_PASS"
    elif len(all_market_ids) > 0 or total_l2_files > 0:
        status = "DATA_QUALITY_ACCUMULATING"
    else:
        status = "PENDING_DATA_ACCUMULATION"

    return {
        "status": status,
        "is_valid": len(all_violations) == 0,
        "violations": all_violations,
        "total_unique_markets": len(all_market_ids),
        "total_l2_files": total_l2_files,
        "total_l2_rows": total_l2_rows,
        "total_trade_files": total_trade_files,
        "total_trade_rows": total_trade_rows,
        "min_coverage_ratio": min_cov,
        "avg_coverage_ratio": avg_cov,
        "eval_days_count": len(eval_stats["dates"]),
        "total_quote_hours": str(eval_stats["total_quote_hours"]),
        "total_simulated_quote_hours": str(eval_stats["total_simulated_quote_hours"]),
        "total_actual_quote_hours": str(eval_stats["total_actual_quote_hours"]),
    }


def main():
    protocol = load_protocol()
    storage_path = Path(protocol.data_storage.root_path)

    print("=" * 60)
    print(f"Data Quality & Integrity Audit — Protocol {protocol.protocol_id}")
    print(f"Storage Root: {storage_path}")
    print("=" * 60)
    print("Gate A Requirements Checklist:")
    print(f"  - Min full UTC days: {protocol.gates.gate_a.min_calendar_days}")
    print(f"  - Min quote-hours: {protocol.gates.gate_a.min_quote_hours}")
    print(f"  - Min active markets: {protocol.gates.gate_a.min_active_markets}")
    print(f"  - Min coverage ratio: {float(protocol.gates.gate_a.min_market_coverage_ratio) * 100:.1f}%")
    print(f"  - Max single market PnL share: {float(protocol.gates.gate_a.max_single_market_pnl_share) * 100:.1f}%\n")

    result = audit_data_quality(storage_path, protocol)

    print(f"Total Unique Markets Recorded: {result['total_unique_markets']}")
    print(f"  - Total L2 Parquet Files: {result['total_l2_files']} ({result['total_l2_rows']} rows)")
    print(f"  - Total Trade Parquet Files: {result['total_trade_files']} ({result['total_trade_rows']} rows)")
    print(f"  - Daily Evaluation Days: {result['eval_days_count']}")
    print(f"  - Total Simulated Quote-Hours: {result['total_simulated_quote_hours']}")
    print(f"  - Total Actual Quote-Hours: {result['total_actual_quote_hours']} (shadow mode = 0.0, real CLOB orders only in canary)")
    print(f"  - Total Quote-Hours (Legacy): {result['total_quote_hours']}")
    print(f"\nCoverage Statistics:")
    print(f"  - Min Coverage Ratio: {result['min_coverage_ratio'] * 100:.1f}%")
    print(f"  - Avg Coverage Ratio: {result['avg_coverage_ratio'] * 100:.1f}%")

    if result["violations"]:
        print(f"\n[INTEGRITY VIOLATIONS DETECTED] ({len(result['violations'])} issues):")
        for v in result["violations"][:10]:
            print(f"  - {v}")
        if len(result["violations"]) > 10:
            print(f"  ... and {len(result['violations']) - 10} more violations.")

    print(f"\nOverall Audit Status: {result['status']}")
    print("=" * 60)

    if not result["is_valid"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
