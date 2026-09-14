from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Tuple
import uuid

from .models import OrderSide, VirtualFill, VirtualOrder

getcontext().prec = 28


class QueuePositionTracker:
    """Tracks queue position ahead of an active order at a specific price level."""

    def __init__(self, order: VirtualOrder, existing_depth_ahead: Decimal, min_order_age_sec: Decimal):
        self.order = order
        self.depth_ahead = existing_depth_ahead
        self.min_order_age_sec = min_order_age_sec

    def process_public_trade(
        self,
        trade_price: Decimal,
        trade_size: Decimal,
        trade_side: OrderSide,  # Aggressor side
        trade_timestamp_ns: int,
    ) -> Optional[VirtualFill]:
        """Evaluate if public trade depletes the queue and fills our order.

        Rules:
        1. Order must be at least min_order_age_sec old.
        2. Aggressor side must match:
           - For BUY orders, aggressor must be SELL.
           - For SELL orders, aggressor must be BUY.
        3. If trade_price penetrates our price (SELL price < our BUY price): fill remaining size.
        4. If trade_price == our price: deplete queue_ahead first, then fill remainder.
        """
        # 1. Order age check
        order_age_sec = Decimal(str(trade_timestamp_ns - self.order.placed_at_ns)) / Decimal("1e9")
        if order_age_sec < self.min_order_age_sec:
            return None

        # 2. Aggressor matching check
        is_buy_order = (self.order.side == OrderSide.BUY)
        expected_aggressor = OrderSide.SELL if is_buy_order else OrderSide.BUY
        if trade_side != expected_aggressor:
            return None

        remaining_order_size = self.order.size - self.order.filled_size
        if remaining_order_size <= Decimal("0.0"):
            return None

        # 3. Through-price penetration
        penetrated = (trade_price < self.order.price) if is_buy_order else (trade_price > self.order.price)
        if penetrated:
            # Immediate full fill of remaining
            fill_size = min(remaining_order_size, trade_size)
            return VirtualFill(
                fill_id=f"fill_{uuid.uuid4().hex[:8]}",
                order_id=self.order.order_id,
                condition_id=self.order.condition_id,
                asset_id=self.order.asset_id,
                side=self.order.side,
                price=self.order.price,
                size=fill_size,
                timestamp_ns=trade_timestamp_ns,
                queue_depletion_ratio=Decimal("1.0"),
            )

        # 4. At-the-money trade
        if trade_price == self.order.price:
            if self.depth_ahead > Decimal("0.0"):
                depleted_from_queue = min(self.depth_ahead, trade_size)
                self.depth_ahead -= depleted_from_queue
                trade_remainder = trade_size - depleted_from_queue
            else:
                trade_remainder = trade_size

            if trade_remainder > Decimal("0.0"):
                fill_size = min(remaining_order_size, trade_remainder)
                return VirtualFill(
                    fill_id=f"fill_{uuid.uuid4().hex[:8]}",
                    order_id=self.order.order_id,
                    condition_id=self.order.condition_id,
                    asset_id=self.order.asset_id,
                    side=self.order.side,
                    price=self.order.price,
                    size=fill_size,
                    timestamp_ns=trade_timestamp_ns,
                    queue_depletion_ratio=fill_size / remaining_order_size,
                )

        return None
