from decimal import Decimal, getcontext
import time
from typing import Dict, List, Optional, Tuple
import uuid

from .models import (
    MarketPosition,
    MarketRewardConfig,
    OrderbookLevel,
    OrderbookSnapshot,
    OrderSide,
    QuotingState,
    VirtualFill,
    VirtualOrder,
)

getcontext().prec = 28


class MarketQuotingFSM:
    """Finite State Machine for market making a single Polymarket binary condition.
    
    States: FLAT -> QUOTING_BOTH -> LONG_YES / LONG_NO -> COMPLETE_SET -> MERGING -> FLAT
    Timeout: 15 minutes in LONG_YES/LONG_NO -> EXITING (VWAP bids + taker fee) -> FLAT
    Safety: HALTED if inventory > $25 or stale data.
    """

    def __init__(
        self,
        config: MarketRewardConfig,
        hedging_timeout_sec: float = 900.0,  # 15 minutes
        max_combined_fill_cost: Decimal = Decimal("0.98"),
    ):
        self.config = config
        self.hedging_timeout_sec = hedging_timeout_sec
        self.max_combined_fill_cost = max_combined_fill_cost

        self.position = MarketPosition(condition_id=config.condition_id)
        self.open_orders: Dict[str, VirtualOrder] = {}  # order_id -> VirtualOrder
        self.state_timer_ns: int = 0
        self.exit_insufficient_liquidity_count: int = 0

    @property
    def state(self) -> QuotingState:
        return self.position.state

    def reset_orders(self) -> List[VirtualOrder]:
        """Cancel all open orders and return them."""
        cancelled = []
        for o in self.open_orders.values():
            o.status = "CANCELLED"
            cancelled.append(o)
        self.open_orders.clear()
        return cancelled

    def generate_quote_orders(
        self,
        midpoint: Decimal,
        timestamp_ns: int,
        target_size: Decimal = Decimal("20.0"),
    ) -> List[VirtualOrder]:
        """Generate symmetric BUY YES and BUY NO orders inside the reward max spread."""
        if self.position.state != QuotingState.FLAT:
            return []

        spread = self.config.rewards_max_spread
        # Place BUY YES near mid - spread * 0.5
        # Place BUY NO near (1 - mid) - spread * 0.5
        p_yes = (midpoint - (spread * Decimal("0.3"))).quantize(Decimal("0.01"))
        p_no = ((Decimal("1.0") - midpoint) - (spread * Decimal("0.3"))).quantize(Decimal("0.01"))

        # Sanity check: min price 0.01, max price 0.99
        p_yes = max(Decimal("0.01"), min(Decimal("0.99"), p_yes))
        p_no = max(Decimal("0.01"), min(Decimal("0.99"), p_no))

        # Combined cost must not exceed max_combined_fill_cost
        if (p_yes + p_no) > self.max_combined_fill_cost:
            adjustment = (p_yes + p_no - self.max_combined_fill_cost) / Decimal("2.0")
            p_yes = (p_yes - adjustment).quantize(Decimal("0.01"))
            p_no = (p_no - adjustment).quantize(Decimal("0.01"))

        order_yes = VirtualOrder(
            order_id=f"order_yes_{uuid.uuid4().hex[:8]}",
            condition_id=self.config.condition_id,
            asset_id=self.config.yes_token_id,
            side=OrderSide.BUY,
            price=p_yes,
            size=target_size,
            placed_at_ns=timestamp_ns,
        )

        order_no = VirtualOrder(
            order_id=f"order_no_{uuid.uuid4().hex[:8]}",
            condition_id=self.config.condition_id,
            asset_id=self.config.no_token_id,
            side=OrderSide.BUY,
            price=p_no,
            size=target_size,
            placed_at_ns=timestamp_ns,
        )

        self.open_orders[order_yes.order_id] = order_yes
        self.open_orders[order_no.order_id] = order_no
        self.position.state = QuotingState.QUOTING_BOTH
        self.position.state_entered_at_ns = timestamp_ns
        return [order_yes, order_no]

    def on_fill(self, fill: VirtualFill) -> List[VirtualOrder]:
        """Process a fill on an active order and trigger FSM state transitions."""
        order = self.open_orders.get(fill.order_id)
        if not order:
            return []

        order.filled_size += fill.size
        if order.filled_size >= order.size:
            order.status = "FILLED"
            del self.open_orders[fill.order_id]
        else:
            order.status = "PARTIALLY_FILLED"

        # Update position and inventory
        is_yes = (fill.asset_id == self.config.yes_token_id)
        if is_yes:
            self.position.yes_inventory += fill.size
            self.position.yes_fill_cost += fill.price * fill.size
            self.position.cash_invested += fill.price * fill.size
        else:
            self.position.no_inventory += fill.size
            self.position.no_fill_cost += fill.price * fill.size
            self.position.cash_invested += fill.price * fill.size

        # Check if we now have matching sets to merge
        matching_sets = min(self.position.yes_inventory, self.position.no_inventory)
        if matching_sets > Decimal("0.0"):
            return self._transition_to_complete_set(fill.timestamp_ns)

        # Transition into one-sided inventory state
        if is_yes:
            return self._handle_yes_fill(fill)
        else:
            return self._handle_no_fill(fill)

    def _handle_yes_fill(self, fill: VirtualFill) -> List[VirtualOrder]:
        # Cancel remaining BUY YES orders
        cancelled: List[VirtualOrder] = []
        to_remove = []
        for oid, o in self.open_orders.items():
            if o.asset_id == self.config.yes_token_id:
                o.status = "CANCELLED"
                cancelled.append(o)
                to_remove.append(oid)
        for oid in to_remove:
            del self.open_orders[oid]

        # In LONG_YES state: keep or reprice opposite leg (BUY NO)
        self.position.state = QuotingState.LONG_YES
        if self.state_timer_ns == 0:
            self.state_timer_ns = fill.timestamp_ns
        self.position.state_entered_at_ns = fill.timestamp_ns

        # Reprice BUY NO so that p_yes_avg + p_no <= max_combined_fill_cost
        avg_yes_price = self.position.yes_fill_cost / self.position.yes_inventory
        max_allowed_no_price = (self.max_combined_fill_cost - avg_yes_price).quantize(Decimal("0.01"))
        max_allowed_no_price = max(Decimal("0.01"), min(Decimal("0.99"), max_allowed_no_price))

        # Adjust existing BUY NO order if needed
        for o in self.open_orders.values():
            if o.asset_id == self.config.no_token_id:
                if o.price > max_allowed_no_price:
                    o.price = max_allowed_no_price
                # Target exact remaining hedge size
                o.size = self.position.yes_inventory

        return cancelled

    def _handle_no_fill(self, fill: VirtualFill) -> List[VirtualOrder]:
        # Cancel remaining BUY NO orders
        cancelled: List[VirtualOrder] = []
        to_remove = []
        for oid, o in self.open_orders.items():
            if o.asset_id == self.config.no_token_id:
                o.status = "CANCELLED"
                cancelled.append(o)
                to_remove.append(oid)
        for oid in to_remove:
            del self.open_orders[oid]

        # In LONG_NO state: keep or reprice opposite leg (BUY YES)
        self.position.state = QuotingState.LONG_NO
        if self.state_timer_ns == 0:
            self.state_timer_ns = fill.timestamp_ns
        self.position.state_entered_at_ns = fill.timestamp_ns

        # Reprice BUY YES so that p_no_avg + p_yes <= max_combined_fill_cost
        avg_no_price = self.position.no_fill_cost / self.position.no_inventory
        max_allowed_yes_price = (self.max_combined_fill_cost - avg_no_price).quantize(Decimal("0.01"))
        max_allowed_yes_price = max(Decimal("0.01"), min(Decimal("0.99"), max_allowed_yes_price))

        # Adjust existing BUY YES order if needed
        for o in self.open_orders.values():
            if o.asset_id == self.config.yes_token_id:
                if o.price > max_allowed_yes_price:
                    o.price = max_allowed_yes_price
                o.size = self.position.no_inventory

        return cancelled

    def _transition_to_complete_set(self, timestamp_ns: int) -> List[VirtualOrder]:
        """Cancel open orders and execute complete set merge."""
        cancelled = self.reset_orders()
        self.position.state = QuotingState.COMPLETE_SET
        self.position.state_entered_at_ns = timestamp_ns

        # Immediately transition to MERGING and resolve
        return self._execute_merge(timestamp_ns)

    def _execute_merge(self, timestamp_ns: int) -> List[VirtualOrder]:
        """Merge complete sets: 1 YES + 1 NO -> 1 USDC pUSD."""
        self.position.state = QuotingState.MERGING
        sets_to_merge = min(self.position.yes_inventory, self.position.no_inventory)

        if sets_to_merge <= Decimal("0.0"):
            self.position.state = QuotingState.FLAT
            return []

        # PnL on merged set: sets_to_merge * $1.00 - cost basis
        avg_cost_yes = (self.position.yes_fill_cost / self.position.yes_inventory) if self.position.yes_inventory > 0 else Decimal("0.0")
        avg_cost_no = (self.position.no_fill_cost / self.position.no_inventory) if self.position.no_inventory > 0 else Decimal("0.0")
        combined_cost = (avg_cost_yes + avg_cost_no) * sets_to_merge

        pnl_realized = (sets_to_merge * Decimal("1.00")) - combined_cost
        self.position.realized_trading_pnl += pnl_realized
        self.position.complete_sets_merged += sets_to_merge
        self.position.cash_invested -= combined_cost

        # Deduct merged inventory
        self.position.yes_inventory -= sets_to_merge
        self.position.no_inventory -= sets_to_merge
        self.position.yes_fill_cost -= avg_cost_yes * sets_to_merge
        self.position.no_fill_cost -= avg_cost_no * sets_to_merge

        # If residual inventory remains, return to appropriate one-sided state
        if self.position.yes_inventory > Decimal("0.0"):
            self.position.state = QuotingState.LONG_YES
        elif self.position.no_inventory > Decimal("0.0"):
            self.position.state = QuotingState.LONG_NO
        else:
            self.position.state = QuotingState.FLAT
            self.state_timer_ns = 0

        return []

    def check_timeout_and_exit(
        self,
        current_timestamp_ns: int,
        orderbook: OrderbookSnapshot,
    ) -> Tuple[bool, Optional[VirtualFill]]:
        """Check if 15-minute hedging timeout has elapsed, and if so, perform forced exit."""
        if self.position.state not in (QuotingState.LONG_YES, QuotingState.LONG_NO):
            return False, None

        if self.state_timer_ns == 0:
            return False, None

        elapsed_sec = (current_timestamp_ns - self.state_timer_ns) / 1e9
        if elapsed_sec < self.hedging_timeout_sec:
            return False, None

        # Timeout reached! Transition to EXITING
        self.position.state = QuotingState.EXITING
        self.reset_orders()

        # Execute VWAP dump against orderbook bids
        is_yes = (self.position.state == QuotingState.LONG_YES or self.position.yes_inventory > 0)
        unhedged_size = self.position.yes_inventory if is_yes else self.position.no_inventory
        cost_basis = self.position.yes_fill_cost if is_yes else self.position.no_fill_cost

        if unhedged_size <= Decimal("0.0"):
            self.position.state = QuotingState.FLAT
            self.state_timer_ns = 0
            return True, None

        # Compute VWAP from bids
        vwap, filled_qty, taker_fee = self._calculate_vwap_exit(orderbook.bids, unhedged_size)

        if filled_qty < unhedged_size:
            # Insufficient depth to absorb inventory
            self.exit_insufficient_liquidity_count += 1
            # Liquidate remainder at penalty 0.01
            penalty_size = unhedged_size - filled_qty
            vwap = ((vwap * filled_qty) + (Decimal("0.01") * penalty_size)) / unhedged_size
            filled_qty = unhedged_size

        exit_revenue = (vwap * unhedged_size) - taker_fee
        pnl = exit_revenue - cost_basis

        self.position.realized_trading_pnl += pnl
        self.position.taker_fees_paid += taker_fee
        self.position.cash_invested -= cost_basis

        fill = VirtualFill(
            fill_id=f"forced_exit_{uuid.uuid4().hex[:8]}",
            order_id="forced_exit",
            condition_id=self.config.condition_id,
            asset_id=self.config.yes_token_id if is_yes else self.config.no_token_id,
            side=OrderSide.SELL,
            price=vwap,
            size=unhedged_size,
            timestamp_ns=current_timestamp_ns,
            queue_depletion_ratio=Decimal("1.0"),
            taker_fee_paid=taker_fee,
        )

        if is_yes:
            self.position.yes_inventory = Decimal("0.0")
            self.position.yes_fill_cost = Decimal("0.0")
        else:
            self.position.no_inventory = Decimal("0.0")
            self.position.no_fill_cost = Decimal("0.0")

        self.position.state = QuotingState.FLAT
        self.state_timer_ns = 0
        return True, fill

    def _calculate_vwap_exit(
        self,
        bids: List[OrderbookLevel],
        target_size: Decimal,
    ) -> Tuple[Decimal, Decimal, Decimal]:
        """Walk bids to calculate execution VWAP and taker fees."""
        # Sort bids descending
        sorted_bids = sorted(bids, key=lambda b: b.price, reverse=True)
        filled = Decimal("0.0")
        total_cash = Decimal("0.0")

        for b in sorted_bids:
            needed = target_size - filled
            fill_from_level = min(needed, b.size)
            total_cash += fill_from_level * b.price
            filled += fill_from_level
            if filled >= target_size:
                break

        if filled == Decimal("0.0"):
            return Decimal("0.01"), Decimal("0.0"), Decimal("0.0")

        vwap = total_cash / filled
        taker_fee = total_cash * self.config.taker_fee_rate
        return vwap, filled, taker_fee
