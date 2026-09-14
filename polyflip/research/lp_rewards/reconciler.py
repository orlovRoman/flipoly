import asyncio
from decimal import Decimal, getcontext
import logging
from typing import Any, Dict, List, Optional, Tuple
import httpx

from .models import OrderbookLevel, OrderbookSnapshot

getcontext().prec = 28
logger = logging.getLogger(__name__)

CLOB_BOOK_URL = "https://clob.polymarket.com/book"


class BookReconciler:
    """Verifies local WS orderbook integrity against REST /book snapshots."""

    def __init__(
        self,
        reconciliation_interval_sec: float = 45.0,
        max_tick_discrepancy: Decimal = Decimal("0.01"),
    ):
        self.reconciliation_interval_sec = reconciliation_interval_sec
        self.max_tick_discrepancy = max_tick_discrepancy
        self.uncertain_markets: Dict[str, bool] = {}  # token_id -> is_uncertain

    async def fetch_rest_book(
        self,
        token_id: str,
        client: httpx.AsyncClient,
    ) -> Optional[Tuple[Optional[Decimal], Optional[Decimal]]]:
        """Fetch best bid and best ask from REST /book endpoint."""
        try:
            resp = await client.get(CLOB_BOOK_URL, params={"token_id": token_id})
            if resp.status_code != 200:
                return None
            data = resp.json()

            bids = data.get("bids", [])
            asks = data.get("asks", [])

            best_bid = Decimal(str(bids[0]["price"])) if bids else None
            best_ask = Decimal(str(asks[0]["price"])) if asks else None
            return best_bid, best_ask
        except Exception as e:
            logger.warning(f"Failed to fetch REST book for token {token_id}: {e}")
            return None

    def reconcile_book(
        self,
        token_id: str,
        local_bids: List[OrderbookLevel],
        local_asks: List[OrderbookLevel],
        rest_best_bid: Optional[Decimal],
        rest_best_ask: Optional[Decimal],
    ) -> bool:
        """Compare local best bid/ask with REST best bid/ask.

        Returns True if reconciled within tolerance, False if mismatch (BOOK_UNCERTAIN).
        """
        local_best_bid = max((b.price for b in local_bids), default=None)
        local_best_ask = min((a.price for a in local_asks), default=None)

        # Check bid mismatch
        if rest_best_bid is not None and local_best_bid is not None:
            if abs(local_best_bid - rest_best_bid) > self.max_tick_discrepancy:
                self.uncertain_markets[token_id] = True
                return False

        # Check ask mismatch
        if rest_best_ask is not None and local_best_ask is not None:
            if abs(local_best_ask - rest_best_ask) > self.max_tick_discrepancy:
                self.uncertain_markets[token_id] = True
                return False

        self.uncertain_markets[token_id] = False
        return True
