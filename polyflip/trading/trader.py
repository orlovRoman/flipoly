import os
import structlog
from typing import Optional, Dict, Any

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.constants import POLYGON

logger = structlog.get_logger(__name__)

class PolyTrader:
    def __init__(self):
        self.host = "https://clob.polymarket.com"
        self.chain_id = POLYGON
        self.private_key = os.getenv("POLYGON_PRIVATE_KEY")
        self.address = os.getenv("POLYGON_ADDRESS")
        
        self.client = None
        if self.private_key and self.address:
            try:
                self.client = ClobClient(
                    self.host,
                    key=self.private_key,
                    chain_id=self.chain_id
                )
                self.client.set_creds(self.client.create_or_derive_creds())
            except Exception as e:
                logger.error("failed_to_init_clob_client", error=str(e))
        else:
            logger.warning("missing_polygon_credentials_paper_trading_mode")

    def execute_trade(
        self, 
        market_id: str, 
        token_id: str, 
        side: str, # "BUY" or "SELL"
        price: float, 
        size: float
    ) -> Dict[str, Any]:
        """
        Отправляет лимитный ордер (FOK - Fill or Kill) в стакан.
        """
        logger.info("executing_trade", market_id=market_id, side=side, price=price, size=size)
        
        if not self.client:
            logger.info("paper_trade_executed", market_id=market_id, side=side, price=price, size=size)
            return {"status": "SUCCESS", "mode": "PAPER", "error_msg": None}
            
        try:
            order_args = OrderArgs(
                price=price,
                size=size,
                side=side,
                token_id=token_id
            )
            
            # Размещаем ордер FOK, чтобы он исполнился сразу или отменился
            resp = self.client.create_and_post_order(order_args, order_type=OrderType.FOK)
            
            if resp and resp.get("success"):
                logger.info("trade_success", order_id=resp.get("orderID"))
                return {"status": "SUCCESS", "mode": "LIVE", "error_msg": None}
            else:
                err = resp.get("errorMsg") if resp else "Unknown error"
                logger.error("trade_failed", error=err)
                return {"status": "FAILED", "mode": "LIVE", "error_msg": err}
                
        except Exception as e:
            logger.exception("trade_exception", error=str(e))
            return {"status": "FAILED", "mode": "LIVE", "error_msg": str(e)}
