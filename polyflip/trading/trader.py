import os
import structlog
import time
from typing import Optional, Dict, Any

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.constants import POLYGON

logger = structlog.get_logger(__name__)

class PolyTrader:
    def __init__(self):
        self.host = "https://clob.polymarket.com"
        self.chain_id = POLYGON

    def get_client(self) -> Optional[ClobClient]:
        # BUG-T01 FIX: Читаем ключ перед каждой сделкой
        private_key = os.getenv("POLYGON_PRIVATE_KEY")
        address = os.getenv("POLYGON_ADDRESS")
        
        if not private_key or not address:
            return None
            
        try:
            client = ClobClient(
                self.host,
                key=private_key,
                chain_id=self.chain_id
            )
            client.set_creds(client.create_or_derive_creds())
            return client
        except Exception as e:
            logger.error("failed_to_init_clob_client", error=str(e))
            return None

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
        
        client = self.get_client()
        
        if not client:
            logger.info("paper_trade_executed", market_id=market_id, side=side, price=price, size=size)
            return {"status": "SUCCESS", "mode": "PAPER", "error_msg": None}
            
        # BUG-T02 FIX: Retry logic с fallback на size/2
        max_retries = 3
        current_size = size
        
        for attempt in range(1, max_retries + 1):
            try:
                order_args = OrderArgs(
                    price=price,
                    size=current_size,
                    side=side,
                    token_id=token_id
                )
                
                resp = client.create_and_post_order(order_args, order_type=OrderType.FOK)
                
                if resp and resp.get("success"):
                    logger.info("trade_success", order_id=resp.get("orderID"), attempt=attempt, size=current_size)
                    return {"status": "SUCCESS", "mode": "LIVE", "error_msg": None}
                
                err = resp.get("errorMsg") if resp else "Unknown error"
                logger.warning("trade_failed_attempt", attempt=attempt, error=err, size=current_size)
                
                if attempt < max_retries:
                    time.sleep(0.5)
                    # Если первый фейл — пробуем уменьшить размер в 2 раза
                    if attempt == 1:
                        current_size = round(current_size / 2, 2)
                        logger.info("fallback_trade_size", new_size=current_size)
                        
            except Exception as e:
                logger.warning("trade_exception_attempt", attempt=attempt, error=str(e))
                if attempt < max_retries:
                    time.sleep(0.5)
                    
        return {"status": "FAILED", "mode": "LIVE", "error_msg": "Max retries exceeded"}
