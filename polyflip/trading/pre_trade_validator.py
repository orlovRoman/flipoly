import dataclasses
from dataclasses import dataclass
from typing import Optional, Any
import structlog

from polyflip.db.models import LiveMarket
from polyflip.trading.trading_config import TradingConfig
from polyflip.trading.decision_logic import TradeDecision
from polyflip.trading.position_sizing import compute_bet_size_edge_scaled
from polyflip.crypto.edge import compute_economic_edge
from polyflip.constants import TRADING_MODE_LIGHTGBM, TRADING_MODE_ML, TRADING_MODE_FAVORITE, TRADING_MODE_COMBINED
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
        return PreTradeValidation(valid=False, buy_price=0.0, actual_bet_size=0.0, edge=0.0, skip_reason="Decision is None")
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
            skip_reason=f"No fresh prices from API for {asset_mode} ({decision})"
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
            skip_reason="OUTSIDER strategy selected a favorite token"
        )

    if decision_obj.strategy_type in {"ML_TREND", "PURE_FAVORITE"}:
        if actual_role != "FAVORITE":
            return PreTradeValidation(
                valid=False, buy_price=buy_price, actual_bet_size=actual_bet_size, edge=0.0,
                skip_reason=f"{decision_obj.strategy_type} selected an outsider token"
            )

    if decision_obj.strategy_type == "LIGHTGBM_TREND" and actual_role == "OUTSIDER" and p_flip < cfg.flip_threshold:
        return PreTradeValidation(
            valid=False, buy_price=buy_price, actual_bet_size=actual_bet_size, edge=0.0,
            skip_reason=f"LightGBM outsider blocked: p_flip={p_flip:.3f} < {cfg.flip_threshold:.3f}"
        )

    price_drift = abs(fresh_ask - buy_price)
    
    if price_drift > cfg.max_price_drift:
        return PreTradeValidation(
            valid=False, buy_price=buy_price, actual_bet_size=actual_bet_size, edge=decision_obj.edge or 0.0,
            skip_reason=f"Price drift too large: {price_drift:.3f}"
        )

    buy_price = fresh_ask
    
    # Пересчет edge по реальной цене
    if decision_obj.p_win_effective is None:
        return PreTradeValidation(
            valid=False, buy_price=buy_price, actual_bet_size=actual_bet_size, edge=decision_obj.edge or 0.0,
            skip_reason="Missing p_win_effective in TradeDecision"
        )
    p_win = decision_obj.p_win_effective

    current_min_edge = cfg.favorite_min_edge if (asset_mode == TRADING_MODE_FAVORITE and cfg.favorite_min_edge is not None) else asset_min_edge
    
    # Считаем новый edge
    edge = compute_economic_edge(p_win, buy_price, cfg.fee_rate, cfg.slippage_rate)
    
    if edge < current_min_edge:
        return PreTradeValidation(
            valid=False, buy_price=buy_price, actual_bet_size=actual_bet_size, edge=edge,
            skip_reason=f"Edge below minimum (edge={edge:.4f} < min={current_min_edge:.4f})"
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
            skip_reason=f"Price out of bounds: {buy_price:.3f} [{cfg.trade_min_price}, {asset_max_price}]"
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
            skip_reason="Bet size <= 0"
        )
        
    # Check max exposure (only for LIVE trades)
    from polyflip.db.execution_models import ExposureReservation
    from sqlalchemy import and_, or_
    import datetime

    # Global max exposure
    exposure_res = await db_session.execute(
        select(func.sum(TradeHistory.amount_usdc)).where(
            TradeHistory.mode == 'LIVE',
            TradeHistory.position_status.in_(['OPEN', 'CLOSING', 'PARTIALLY_CLOSED'])
        )
    )
    current_global_exposure = exposure_res.scalar() or 0.0
    max_global_exposure = cfg.capital * (cfg.max_exposure_pct / 100.0)
    
    if current_global_exposure + actual_bet_size > max_global_exposure:
        return PreTradeValidation(
            valid=False, buy_price=buy_price, actual_bet_size=actual_bet_size, edge=edge,
            skip_reason=f"Global max exposure exceeded: current {current_global_exposure:.2f} + new {actual_bet_size:.2f} > max {max_global_exposure:.2f}"
        )

    # Per-market max exposure (limit 50 USDC per market)
    MAX_MARKET_EXPOSURE = 50.0

    market_exposure_res = await db_session.execute(
        select(func.sum(TradeHistory.entry_cost_usdc)).where(
            TradeHistory.market_id == market.market_id,
            TradeHistory.mode == 'LIVE',
            TradeHistory.position_status.in_(['OPEN', 'CLOSING', 'PARTIALLY_CLOSED'])
        )
    )
    current_market_exposure = market_exposure_res.scalar() or 0.0

    # Also sum active reservations for this market
    reservations_res = await db_session.execute(
        select(func.sum(ExposureReservation.amount_usdc)).where(
            ExposureReservation.market_id == market.market_id,
            ExposureReservation.expires_at > datetime.datetime.now(datetime.timezone.utc)
        )
    )
    reserved_market_exposure = float(reservations_res.scalar() or 0.0)

    total_market_exposure = float(current_market_exposure) + reserved_market_exposure
    if total_market_exposure + actual_bet_size > MAX_MARKET_EXPOSURE:
        return PreTradeValidation(
            valid=False, buy_price=buy_price, actual_bet_size=actual_bet_size, edge=edge,
            skip_reason=f"Market max exposure exceeded: current {float(current_market_exposure):.2f} + reserved {reserved_market_exposure:.2f} + new {actual_bet_size:.2f} > max {MAX_MARKET_EXPOSURE:.2f}"
        )

    return PreTradeValidation(
        valid=True,
        buy_price=buy_price,
        actual_bet_size=actual_bet_size,
        edge=edge,
        skip_reason=None,
        market_role=actual_role
    )
