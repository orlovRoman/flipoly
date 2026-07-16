import dataclasses
from dataclasses import dataclass
from typing import Optional, Any
import structlog

from polyflip.db.models import LiveMarket
from polyflip.trading.trading_config import TradingConfig
from polyflip.trading.decision_logic import TradeDecision
from polyflip.trading.position_sizing import compute_edge, compute_bet_size_edge_scaled
from polyflip.constants import TRADING_MODE_CRYPTO, TRADING_MODE_ML, TRADING_MODE_FAVORITE

logger = structlog.get_logger(__name__)


@dataclass
class PreTradeValidation:
    valid: bool
    buy_price: float
    actual_bet_size: float
    edge: float
    skip_reason: Optional[str]


async def validate_pre_trade(
    api_client: Any,
    market: LiveMarket,
    decision_obj: Optional[TradeDecision],
    cfg: TradingConfig,
    asset_mode: str,
    asset_min_edge: float,
    asset_max_price: float,
    p_flip: float,
    model_ver: Optional[int],
) -> PreTradeValidation:
    """
    Финальная проверка сделки (Pre-Trade): 
    запрос актуальной цены, проверка drift, edge, лимитов цен и размера ставки.
    """
    if decision_obj is None:
        return PreTradeValidation(valid=False, buy_price=0.0, actual_bet_size=0.0, edge=0.0, skip_reason="Decision is None")
    if decision_obj.action == "SKIP":
        return PreTradeValidation(valid=False, buy_price=0.0, actual_bet_size=0.0, edge=0.0, skip_reason=decision_obj.reason)

    decision = decision_obj.action.replace("BUY_", "")
    buy_price = decision_obj.buy_price
    actual_bet_size = decision_obj.bet_size_usdc
    token_to_buy = market.yes_token_id if decision == "YES" else market.no_token_id
    
    fresh_prices = await api_client.get_market_prices(token_to_buy)
    if not fresh_prices or fresh_prices.get("best_ask") is None:
        return PreTradeValidation(
            valid=False, buy_price=buy_price, actual_bet_size=actual_bet_size, edge=decision_obj.edge or 0.0,
            skip_reason=f"No fresh prices from API for {asset_mode} ({decision})"
        )

    fresh_ask = fresh_prices["best_ask"]
    price_drift = abs(fresh_ask - buy_price)
    
    if price_drift > cfg.max_price_drift:
        return PreTradeValidation(
            valid=False, buy_price=buy_price, actual_bet_size=actual_bet_size, edge=decision_obj.edge or 0.0,
            skip_reason=f"Price drift too large: {price_drift:.3f}"
        )

    buy_price = fresh_ask
    
    # Пересчет edge по реальной цене
    if asset_mode == TRADING_MODE_CRYPTO:
        edge = decision_obj.edge or 0.0
        actual_bet_size = decision_obj.bet_size_usdc
    else:
        if asset_mode == TRADING_MODE_ML:
            p_win = 1.0 - p_flip if decision_obj.strategy_type == "ML_TREND" else p_flip
            current_min_edge = asset_min_edge
        else: # FAVORITE
            p_win = market.current_yes_price if decision == "YES" else (1.0 - market.current_yes_price)
            current_min_edge = cfg.favorite_min_edge if cfg.favorite_min_edge is not None else asset_min_edge
            
        edge = compute_edge(p_win, buy_price)
        
        if edge < current_min_edge or edge > cfg.max_edge_filter:
            return PreTradeValidation(
                valid=False, buy_price=buy_price, actual_bet_size=actual_bet_size, edge=edge,
                skip_reason=f"Edge out of bounds (edge={edge:.4f})"
            )

    if not (cfg.trade_min_price <= buy_price <= asset_max_price):
        return PreTradeValidation(
            valid=False, buy_price=buy_price, actual_bet_size=actual_bet_size, edge=edge,
            skip_reason=f"Price out of bounds: {buy_price:.3f} [{cfg.trade_min_price}, {asset_max_price}]"
        )
        
    if asset_mode != TRADING_MODE_CRYPTO:
        if cfg.bet_sizing_mode == "fixed":
            actual_bet_size = cfg.bet_size
        else:
            actual_bet_size = compute_bet_size_edge_scaled(
                edge=edge,
                min_bet_usdc=cfg.bet_size,
                max_bet_usdc=cfg.max_bet_size_usdc,
                min_edge=current_min_edge,
                max_edge=cfg.max_bet_edge
            )
            
        if asset_mode == TRADING_MODE_FAVORITE and actual_bet_size < cfg.bet_size:
            actual_bet_size = cfg.bet_size

    if actual_bet_size <= 0:
        return PreTradeValidation(
            valid=False, buy_price=buy_price, actual_bet_size=actual_bet_size, edge=edge,
            skip_reason="Bet size <= 0"
        )

    return PreTradeValidation(
        valid=True,
        buy_price=buy_price,
        actual_bet_size=actual_bet_size,
        edge=edge,
        skip_reason=None
    )
