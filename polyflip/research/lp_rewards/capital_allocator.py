from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Tuple
from .models import MarketPosition, VirtualOrder

getcontext().prec = 28


class CapitalAllocator:
    """Manages working capital allocation under strict $100.00 capital constraints.
    
    Rules:
    - Base allocated working capital: $100.00
    - Max unhedged exposure per market: $25.00
    - Max total unhedged exposure across all markets: $50.00
    - Reserved collateral + inventory MTM + exit reserve <= $100.00
    """

    def __init__(
        self,
        allocated_working_capital: Decimal = Decimal("100.00"),
        max_unhedged_per_market: Decimal = Decimal("25.00"),
        max_unhedged_total: Decimal = Decimal("50.00"),
        exit_reserve_ratio: Decimal = Decimal("0.10"),
    ):
        self.allocated_working_capital = allocated_working_capital
        self.max_unhedged_per_market = max_unhedged_per_market
        self.max_unhedged_total = max_unhedged_total
        self.exit_reserve_ratio = exit_reserve_ratio

        # Time-weighted extra buffer tracking (for wallet safety reserve)
        self.extra_buffer_time_weighted_seconds: Decimal = Decimal("0.0")
        self.total_elapsed_seconds: Decimal = Decimal("0.0")

    def can_allocate_orders(
        self,
        condition_id: str,
        positions: Dict[str, MarketPosition],
        open_orders: Dict[str, List[VirtualOrder]],
        new_orders: List[VirtualOrder],
        current_market_prices: Optional[Dict[str, Tuple[Decimal, Decimal]]] = None,  # (mid_yes, mid_no)
    ) -> Tuple[bool, str]:
        """Check if placing new_orders would violate capital rules."""
        # 1. Calculate current market unhedged exposure
        pos = positions.get(condition_id, MarketPosition(condition_id=condition_id))
        net_inventory_cost = abs(pos.yes_inventory * Decimal("0.5") - pos.no_inventory * Decimal("0.5"))

        # Calculate new order cost commitment
        new_order_commitment = Decimal("0.0")
        for o in new_orders:
            remaining_size = o.size - o.filled_size
            new_order_commitment += o.price * remaining_size

        # Check market unhedged limit ($25)
        if (net_inventory_cost + new_order_commitment) > self.max_unhedged_per_market:
            return (
                False,
                f"Exceeds per-market unhedged limit of ${self.max_unhedged_per_market}: "
                f"current={net_inventory_cost}, requested={new_order_commitment}",
            )

        # 2. Calculate total portfolio unhedged exposure
        total_unhedged = Decimal("0.0")
        for cid, p in positions.items():
            total_unhedged += abs(p.yes_inventory * Decimal("0.5") - p.no_inventory * Decimal("0.5"))

        if (total_unhedged + new_order_commitment) > self.max_unhedged_total:
            return (
                False,
                f"Exceeds total portfolio unhedged limit of ${self.max_unhedged_total}: "
                f"current={total_unhedged}, requested={new_order_commitment}",
            )

        # 3. Calculate total capital commitment across all markets
        total_open_orders_committed = Decimal("0.0")
        for cid, orders in open_orders.items():
            for o in orders:
                total_open_orders_committed += o.price * (o.size - o.filled_size)

        total_inventory_invested = sum((p.cash_invested for p in positions.values()), Decimal("0.0"))
        exit_reserve = (total_unhedged + new_order_commitment) * self.exit_reserve_ratio

        total_needed = (
            total_open_orders_committed
            + total_inventory_invested
            + new_order_commitment
            + exit_reserve
        )

        if total_needed > self.allocated_working_capital:
            return (
                False,
                f"Exceeds allocated working capital ${self.allocated_working_capital}: "
                f"total needed=${total_needed}",
            )

        return True, "OK"

    def record_extra_buffer_usage(self, extra_buffer_amount: Decimal, duration_seconds: Decimal) -> None:
        """Track time-weighted extra buffer capital during recovery."""
        if extra_buffer_amount > Decimal("0.0") and duration_seconds > Decimal("0.0"):
            self.extra_buffer_time_weighted_seconds += extra_buffer_amount * duration_seconds
        self.total_elapsed_seconds += duration_seconds

    def compute_effective_capital(self) -> Decimal:
        """Compute time-weighted effective capital base."""
        if self.total_elapsed_seconds <= Decimal("0.0"):
            return self.allocated_working_capital
        avg_extra = self.extra_buffer_time_weighted_seconds / self.total_elapsed_seconds
        return self.allocated_working_capital + avg_extra

    def calculate_r100_calendar(self, net_pnl: Decimal, completed_full_utc_days: int) -> Decimal:
        """Calculate primary hypothesis metric R_100_calendar:
        R_100_calendar = (Net PnL / completed_days) * (100 / effective_capital)
        """
        if completed_full_utc_days <= 0:
            return Decimal("0.0")

        daily_rate = net_pnl / Decimal(str(completed_full_utc_days))
        effective_cap = self.compute_effective_capital()
        normalization_factor = Decimal("100.00") / effective_cap
        return daily_rate * normalization_factor
