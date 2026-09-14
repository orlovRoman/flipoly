from decimal import Decimal, getcontext
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .models import MarketPosition, OrderbookLevel, VirtualFill

getcontext().prec = 28


class PortfolioLedger:
    """Maintains conservative double-entry accounting for simulated and live LP operations."""

    def __init__(self, allocated_capital: Decimal = Decimal("100.00")):
        self.allocated_capital = allocated_capital
        self.daily_rewards_accrued: Dict[str, Decimal] = {}  # date_str -> Decimal
        self.trades: List[VirtualFill] = []
        self.total_taker_fees: Decimal = Decimal("0.0")

    def record_fill(self, fill: VirtualFill) -> None:
        self.trades.append(fill)
        self.total_taker_fees += fill.taker_fee_paid

    def record_daily_rewards(self, date_str: str, reward_amount: Decimal) -> None:
        self.daily_rewards_accrued[date_str] = self.daily_rewards_accrued.get(date_str, Decimal("0.0")) + reward_amount

    def calculate_executable_mtm(
        self,
        positions: Dict[str, MarketPosition],
        current_bids_by_token: Dict[str, List[OrderbookLevel]],
        taker_fee_rates: Optional[Dict[str, Decimal]] = None,
    ) -> Decimal:
        """Calculate inventory MTM strictly by executable liquidation value against bids."""
        if taker_fee_rates is None:
            taker_fee_rates = {}

        total_liquidation_value = Decimal("0.0")

        for cid, pos in positions.items():
            fee_rate = taker_fee_rates.get(cid, Decimal("0.0"))

            # Liquidate YES inventory
            if pos.yes_inventory > Decimal("0.0"):
                bids = current_bids_by_token.get(f"{cid}_YES", [])
                val = self._walk_bids_liquidation(bids, pos.yes_inventory, fee_rate)
                total_liquidation_value += val

            # Liquidate NO inventory
            if pos.no_inventory > Decimal("0.0"):
                bids = current_bids_by_token.get(f"{cid}_NO", [])
                val = self._walk_bids_liquidation(bids, pos.no_inventory, fee_rate)
                total_liquidation_value += val

        return total_liquidation_value

    def _walk_bids_liquidation(
        self,
        bids: List[OrderbookLevel],
        size: Decimal,
        fee_rate: Decimal,
    ) -> Decimal:
        sorted_bids = sorted(bids, key=lambda b: b.price, reverse=True)
        filled = Decimal("0.0")
        cash = Decimal("0.0")

        for b in sorted_bids:
            needed = size - filled
            qty = min(needed, b.size)
            cash += qty * b.price
            filled += qty
            if filled >= size:
                break

        # Taker fee on liquidation
        fee = cash * fee_rate
        net_cash = cash - fee
        return max(Decimal("0.0"), net_cash)

    def calculate_net_pnl(
        self,
        positions: Dict[str, MarketPosition],
        executable_mtm: Decimal,
    ) -> Decimal:
        """Net PnL = Cumulative Rewards + Realized Trading PnL + Executable MTM - Cash Invested in Inventory - Fees."""
        cumulative_rewards = sum(self.daily_rewards_accrued.values(), Decimal("0.0"))
        realized_trading_pnl = sum((p.realized_trading_pnl for p in positions.values()), Decimal("0.0"))
        cash_invested = sum((p.cash_invested for p in positions.values()), Decimal("0.0"))

        net_pnl = (
            cumulative_rewards
            + realized_trading_pnl
            + executable_mtm
            - cash_invested
            - self.total_taker_fees
        )
        return net_pnl

    def export_trades_parquet(self, output_file: Path) -> None:
        """Export executed fills to Parquet format."""
        if not self.trades:
            return

        records = [
            {
                "fill_id": t.fill_id,
                "order_id": t.order_id,
                "condition_id": t.condition_id,
                "asset_id": t.asset_id,
                "side": t.side.value,
                "price": float(t.price),
                "size": float(t.size),
                "timestamp_ns": t.timestamp_ns,
                "queue_depletion_ratio": float(t.queue_depletion_ratio),
                "taker_fee_paid": float(t.taker_fee_paid),
            }
            for t in self.trades
        ]
        df = pd.DataFrame(records)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_file, compression="zstd", index=False)
