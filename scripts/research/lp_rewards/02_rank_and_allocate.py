"""02_rank_and_allocate.py

Ranks active universe by reward density and computes capital commitments under $100 working capital.
"""

from decimal import Decimal
import json
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from polyflip.research.lp_rewards.capital_allocator import CapitalAllocator
from polyflip.research.lp_rewards.models import MarketRewardConfig, VirtualOrder, OrderSide
from polyflip.research.lp_rewards.protocol import load_protocol


def main():
    protocol = load_protocol()
    storage_path = Path(protocol.data_storage.root_path)
    universe_file = storage_path / "universe_active.json"

    if not universe_file.exists():
        print(f"Error: {universe_file} not found. Run 01_fetch_universe.py first.")
        sys.exit(1)

    with open(universe_file, "r", encoding="utf-8") as f:
        raw_list = json.load(f)

    markets = [MarketRewardConfig(**m) for m in raw_list]
    allocator = CapitalAllocator(
        allocated_working_capital=protocol.capital_allocation.allocated_working_capital,
        max_unhedged_per_market=protocol.capital_allocation.max_unhedged_per_market,
        max_unhedged_total=protocol.capital_allocation.max_unhedged_total,
    )

    print(f"Allocating working capital (${allocator.allocated_working_capital}) across {len(markets)} active markets...")
    print(f"Constraints: Max unhedged/market = ${allocator.max_unhedged_per_market}, Max total = ${allocator.max_unhedged_total}\n")

    positions = {}
    open_orders = {}

    allocated_count = 0
    for m in markets:
        # Generate hypothetical test orders of $10 each
        test_orders = [
            VirtualOrder(
                order_id=f"test_yes_{m.condition_id[:6]}",
                condition_id=m.condition_id,
                asset_id=m.yes_token_id,
                side=OrderSide.BUY,
                price=Decimal("0.48"),
                size=Decimal("20.0"),
                placed_at_ns=0,
            ),
            VirtualOrder(
                order_id=f"test_no_{m.condition_id[:6]}",
                condition_id=m.condition_id,
                asset_id=m.no_token_id,
                side=OrderSide.BUY,
                price=Decimal("0.48"),
                size=Decimal("20.0"),
                placed_at_ns=0,
            ),
        ]

        allowed, reason = allocator.can_allocate_orders(m.condition_id, positions, open_orders, test_orders)
        if allowed:
            open_orders[m.condition_id] = test_orders
            allocated_count += 1
            print(f"  [ALLOCATED] {m.question[:50]}... -> $19.20 committed")
        else:
            print(f"  [SKIPPED] {m.question[:50]}... -> {reason}")

    print(f"\nSuccessfully allocated capital to {allocated_count} markets concurrently.")


if __name__ == "__main__":
    main()
