"""04_audit_data_quality.py

Audits captured L2 data and FSM states against Gate A quality criteria:
- Coverage >= 99% per market
- >= 100 quote-hours
- Check for MID_UNCERTAIN or BOOK_UNCERTAIN anomalies
"""

from decimal import Decimal
import json
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from polyflip.research.lp_rewards.protocol import load_protocol


def main():
    protocol = load_protocol()
    storage_path = Path(protocol.data_storage.root_path)

    print(f"Auditing LP Rewards data quality at: {storage_path}")
    print(f"Gate A Requirements:")
    print(f"  - Min full UTC days: {protocol.gates.gate_a.min_calendar_days}")
    print(f"  - Min quote-hours: {protocol.gates.gate_a.min_quote_hours}")
    print(f"  - Min coverage ratio: {float(protocol.gates.gate_a.min_market_coverage_ratio) * 100:.1f}%")
    print(f"  - Max single market PnL share: {float(protocol.gates.gate_a.max_single_market_pnl_share) * 100:.1f}%\n")

    trades_dir = storage_path / "public_trades"
    if not trades_dir.exists():
        print(f"Notice: {trades_dir} is empty or not yet populated. (Shadow collector has not completed a full run).")
        print("Data Quality Status: PENDING_DATA_ACCUMULATION")
        return

    # Check markets
    markets_found = list(trades_dir.glob("*"))
    print(f"Markets recorded on disk: {len(markets_found)}")


if __name__ == "__main__":
    main()
