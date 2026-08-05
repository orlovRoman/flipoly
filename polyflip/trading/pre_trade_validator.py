import dataclasses
from dataclasses import dataclass
from typing import Optional, Any
import structlog

from polyflip.db.models import LiveMarket
from polyflip.trading.trading_config import TradingConfig
from polyflip.trading.decision_logic import TradeDecision
from polyflip.trading.position_sizing import compute_bet_size_edge_scaled
from polyflip.crypto.edge import compute_economic_edge
from polyflip.constants import TRADING_MODE_FAVORITE, TRADING_MODE_COMBINED
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from polyflip.db.models import TradeHistory

logger = structlog.get_logger(__name__)


@dataclass
class PreTradeValidation:
    valid: bool
    buy_price: float
    actual_bet_size: float
    edge: float
    skip_reason: Optional[str]
    market_role: Optional[str] = None


async def validate_pre_trade(
    db_session: AsyncSession,
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
        return PreTradeValidation(valid=False, buy_price=0.0, actual_bet_size=0.0, edge=0.0, skip_reason="validation: Decision is None")
    if decision_obj.action == "SKIP":
        return PreTradeValidation(valid=False, buy_price=0.0, actual_bet_size=0.0, edge=0.0, skip_reason=decision_obj.reason)

    edge: float = decision_obj.edge or 0.0
    current_min_edge: float = asset_min_edge

    decision = decision_obj.action.replace("BUY_", "")
    buy_price = decision_obj.buy_price
    actual_bet_size = decision_obj.bet_size_usdc
    token_to_buy = market.yes_token_id if decision == "YES" else market.no_token_id
    
    # Fetch fresh prices first
    fresh_prices = await api_client.get_market_prices(token_to_buy)
    if not fresh_prices or fresh_prices.get("best_ask") is None:
        return PreTradeValidation(
            valid=False, buy_price=buy_price, actual_bet_size=actual_bet_size, edge=decision_obj.edge or 0.0,
            skip_reason=f"validation: No fresh prices from API for {asset_mode} ({decision})"
        )

    fresh_ask = float(fresh_prices["best_ask"])
    fresh_bid = fresh_prices.get("best_bid")

    if fresh_bid is not None:
        fresh_mid = (float(fresh_bid) + fresh_ask) / 2
    else:
        fresh_mid = fresh_ask

    actual_role = "OUTSIDER" if fresh_mid < 0.50 else "FAVORITE"

    # Validate Market Role invariants using fresh prices
    if decision_obj.strategy_type == "OUTSIDER" and actual_role != "OUTSIDER":
        return PreTradeValidation(
            valid=False, buy_price=buy_price, actual_bet_size=actual_bet_size, edge=0.0,
            skip_reason="validation: OUTSIDER strategy selected a favorite token"
        )

    if decision_obj.strategy_type in {"ML_TREND", "PURE_FAVORITE"}:
        if actual_role != "FAVORITE":
            return PreTradeValidation(
                valid=False, buy_price=buy_price, actual_bet_size=actual_bet_size, edge=0.0,
                skip_reason=f"validation: {decision_obj.strategy_type} selected an outsider token"
            )
    # Note: p_flip threshold check removed (was legacy ML trend-specific).
    # Combined mode enforces p_flip validation inside evaluate_combined_entry (step 4+).
    price_drift = abs(fresh_ask - buy_price)
    
    if price_drift > cfg.max_price_drift:
        return PreTradeValidation(
            valid=False, buy_price=buy_price, actual_bet_size=actual_bet_size, edge=decision_obj.edge or 0.0,
            skip_reason=f"validation: Price drift too large: {price_drift:.3f}"
        )

    # Проверка max_acceptable_price для COMBINED режима
    max_acc_price = decision_obj.decision_details.get("max_acceptable_price") if decision_obj.decision_details else None
    if max_acc_price is not None and fresh_ask > float(max_acc_price):
        return PreTradeValidation(
            valid=False, buy_price=buy_price, actual_bet_size=actual_bet_size, edge=decision_obj.edge or 0.0,
            skip_reason=f"validation: Fresh ask {fresh_ask:.3f} exceeded max_acceptable_price {float(max_acc_price):.3f}"
        )

    buy_price = fresh_ask
    
    # Пересчет edge по реальной цене
    if decision_obj.p_win_effective is None:
        return PreTradeValidation(
            valid=False, buy_price=buy_price, actual_bet_size=actual_bet_size, edge=decision_obj.edge or 0.0,
            skip_reason="validation: Missing p_win_effective in TradeDecision"
        )
    p_win = decision_obj.p_win_effective

    if asset_mode == TRADING_MODE_COMBINED or asset_mode == TRADING_MODE_FAVORITE:
        # Определяем: аутсайдер (price < 0.5) или фаворит
        is_outsider = buy_price < 0.5
        current_min_edge = cfg.get_min_edge(is_outsider=is_outsider)
    else:
        current_min_edge = asset_min_edge
    
    # Считаем новый edge
    edge = compute_economic_edge(p_win, buy_price, cfg.fee_rate, cfg.slippage_rate)
    
    if edge < current_min_edge:
        return PreTradeValidation(
            valid=False, buy_price=buy_price, actual_bet_size=actual_bet_size, edge=edge,
            skip_reason=f"validation: Edge below minimum (edge={edge:.4f} < min={current_min_edge:.4f})"
        )

    ANOMALY_EDGE_WARN = 0.60
    if edge > ANOMALY_EDGE_WARN:
        derived_p_win = round((edge + 1.0) * buy_price, 4)
        logger.warning(
            "anomalous_edge_detected",
            asset=market.asset,
            edge=round(edge, 4),
            derived_p_win=derived_p_win,
            buy_price=buy_price,
            note="possible stale data or API price error"
        )

    if not (cfg.trade_min_price <= buy_price <= asset_max_price):
        return PreTradeValidation(
            valid=False, buy_price=buy_price, actual_bet_size=actual_bet_size, edge=edge,
            skip_reason=f"validation: Price out of bounds: {buy_price:.3f} [{cfg.trade_min_price}, {asset_max_price}]"
        )
        
    # Рассчитываем новую ставку (если не фикс)
    if cfg.bet_sizing_mode == "fixed":
        # P1.12: Запретить уменьшение fixed ставки. Игнорируем decision_obj.bet_size_usdc
        actual_bet_size = cfg.bet_size
    else:
        newly_calculated_bet_size = compute_bet_size_edge_scaled(
            edge=edge,
            min_bet_usdc=cfg.bet_size,
            max_bet_usdc=cfg.max_bet_size_usdc,
            min_edge=current_min_edge,
            max_edge=cfg.max_bet_edge
        )
        actual_bet_size = newly_calculated_bet_size
        if asset_mode == TRADING_MODE_FAVORITE and actual_bet_size < cfg.bet_size:
            actual_bet_size = cfg.bet_size

    if actual_bet_size <= 0:
        return PreTradeValidation(
            valid=False, buy_price=buy_price, actual_bet_size=actual_bet_size, edge=edge,
            skip_reason="validation: Bet size <= 0"
        )
        


    return PreTradeValidation(
        valid=True,
        buy_price=buy_price,
        actual_bet_size=actual_bet_size,
        edge=edge,
        skip_reason=None,
        market_role=actual_role
    )
