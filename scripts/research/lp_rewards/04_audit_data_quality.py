import datetime
from decimal import Decimal
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List
import pandas as pd

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from polyflip.research.lp_rewards.protocol import load_protocol

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("audit_quality")


def main():
    protocol = load_protocol()
    storage_path = Path(protocol.data_storage.root_path)

    print("=" * 60)
    print(f"Data Quality & Integrity Audit — Protocol {protocol.protocol_id}")
    print(f"Storage Root: {storage_path}")
    print("=" * 60)
    print(f"Gate A Requirements Checklist:")
    print(f"  - Min full UTC days: {protocol.gates.gate_a.min_calendar_days}")
    print(f"  - Min quote-hours: {protocol.gates.gate_a.min_quote_hours}")
    print(f"  - Min active markets: {protocol.gates.gate_a.min_active_markets}")
    print(f"  - Min coverage ratio: {float(protocol.gates.gate_a.min_market_coverage_ratio) * 100:.1f}%")
    print(f"  - Max single market PnL share: {float(protocol.gates.gate_a.max_single_market_pnl_share) * 100:.1f}%\n")

    trades_dir = storage_path / "public_trades"
    l2_dir = storage_path / "l2_snapshots"
    ledger_dir = storage_path / "simulated_ledger"

    has_trades = trades_dir.exists() and list(trades_dir.glob("*"))
    has_l2 = l2_dir.exists() and list(l2_dir.glob("*"))

    if not has_trades and not has_l2:
        print(f"Notice: Neither {trades_dir} nor {l2_dir} has accumulated data yet.")
        print("Data Quality Status: PENDING_DATA_ACCUMULATION")
        return

    # Check L2 markets & files
    l2_markets = list(l2_dir.glob("*")) if l2_dir.exists() else []
    trade_markets = list(trades_dir.glob("*")) if trades_dir.exists() else []
    all_market_ids = set([p.name for p in l2_markets] + [p.name for p in trade_markets])

    print(f"Total Unique Markets Recorded: {len(all_market_ids)}")
    print(f"  - L2 Snapshot Directories: {len(l2_markets)}")
    print(f"  - Public Trade Directories: {len(trade_markets)}")

    # Check date coverage and file integrity
    total_l2_files = 0
    total_trade_files = 0
    date_coverage: Dict[str, set] = {}

    for m_dir in l2_markets:
        for p_file in m_dir.glob("*.parquet"):
            total_l2_files += 1
            date_coverage.setdefault(m_dir.name, set()).add(p_file.stem)

    for m_dir in trade_markets:
        for p_file in m_dir.glob("*.parquet"):
            total_trade_files += 1

    print(f"Total L2 Parquet Files: {total_l2_files}")
    print(f"Total Trade Parquet Files: {total_trade_files}")

    # Coverage analysis
    coverage_ratios: Dict[str, float] = {}
    for cid, dates in date_coverage.items():
        # Coverage ratio based on 7-day target window
        ratio = min(1.0, len(dates) / float(protocol.gates.gate_a.min_calendar_days))
        coverage_ratios[cid] = ratio

    min_cov = min(coverage_ratios.values()) if coverage_ratios else 0.0
    avg_cov = (sum(coverage_ratios.values()) / len(coverage_ratios)) if coverage_ratios else 0.0

    print(f"\nCoverage Statistics across {len(coverage_ratios)} markets:")
    print(f"  - Min Coverage Ratio: {min_cov * 100:.1f}%")
    print(f"  - Avg Coverage Ratio: {avg_cov * 100:.1f}%")

    # Overall Audit Status
    req_markets = protocol.gates.gate_a.min_active_markets
    req_cov = float(protocol.gates.gate_a.min_market_coverage_ratio)

    if len(all_market_ids) >= req_markets and min_cov >= req_cov:
        status = "DATA_QUALITY_VERIFIED_PASS"
    elif len(all_market_ids) > 0:
        status = "DATA_QUALITY_ACCUMULATING"
    else:
        status = "PENDING_DATA_ACCUMULATION"

    print(f"\nOverall Audit Status: {status}")
    print("=" * 60)


if __name__ == "__main__":
    main()
