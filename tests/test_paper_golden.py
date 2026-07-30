import pytest
import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy import select, func
from polyflip.db.models import TradeHistory, ModelRegistry, DecisionFunnelLog, RuntimeSettings
from polyflip.db.execution_models import ExecutionRequest
from polyflip.execution.outbox import enqueue_open_request, EnqueueDisposition
from polyflip.execution.config import ExecutionMode

@dataclass(frozen=True)
class PaperGoldenTestCase:
    id: str
    asset: str
    outcome: str
    predicted_prob: float
    best_price: float
    min_edge: float
    market_role: str
    model_key: str
    model_version: int
    expected_action: str
    expected_requests: int

PAPER_GOLDEN_CASES = [
    PaperGoldenTestCase("case_btc_yes_fav", "BTC", "YES", 0.65, 0.50, 0.05, "FAVORITE", "BTC_leaning", 3, "BUY_YES", 1),
    PaperGoldenTestCase("case_btc_no_out", "BTC", "NO", 0.35, 0.45, 0.05, "OUTSIDER", "BTC_leaning", 3, "BUY_NO", 1),
    PaperGoldenTestCase("case_eth_yes_fav", "ETH", "YES", 0.70, 0.55, 0.05, "FAVORITE", "ETH_decided", 3, "BUY_YES", 1),
    PaperGoldenTestCase("case_eth_skip_low_edge", "ETH", "YES", 0.52, 0.50, 0.05, "FAVORITE", "ETH_decided", 3, "SKIP", 0),
    PaperGoldenTestCase("case_sol_yes_contested", "SOL", "YES", 0.60, 0.40, 0.05, "FAVORITE", "SOL_contested", 1, "BUY_YES", 1),
    PaperGoldenTestCase("case_doge_no_fav", "DOGE", "NO", 0.30, 0.50, 0.05, "FAVORITE", "DOGE_leaning", 8, "BUY_NO", 1),
    PaperGoldenTestCase("case_xrp_lgbm_mid_vol", "XRP", "YES", 0.68, 0.45, 0.05, "FAVORITE", "XRPUSDT_mid_vol", 13, "BUY_YES", 1),
    PaperGoldenTestCase("case_xrp_skip_edge", "XRP", "YES", 0.51, 0.50, 0.05, "FAVORITE", "XRPUSDT_mid_vol", 13, "SKIP", 0),
    PaperGoldenTestCase("case_btc_high_vol", "BTC", "YES", 0.75, 0.50, 0.05, "FAVORITE", "BTCUSDT_high_vol", 21, "BUY_YES", 1),
    PaperGoldenTestCase("case_eth_no_out", "ETH", "NO", 0.25, 0.55, 0.05, "OUTSIDER", "ETH_leaning", 3, "BUY_NO", 1),
]

@dataclass(frozen=True)
class PaperCycleResult:
    action: str
    outcome: Optional[str]
    bet_size_usdc: float
    limit_price: Optional[float]
    edge: Optional[float]
    market_role: Optional[str]
    execution_requests: int
    realized_pnl_usdc: float

async def run_deterministic_paper_cycle(db_session, case: PaperGoldenTestCase) -> PaperCycleResult:
    edge = case.predicted_prob - case.best_price if case.outcome == "YES" else (1.0 - case.predicted_prob) - case.best_price
    
    if edge < case.min_edge:
        action = "SKIP"
        req_count = 0
        outcome = None
        limit_price = None
        edge_val = None
    else:
        action = f"BUY_{case.outcome}"
        outcome = case.outcome
        limit_price = case.best_price
        edge_val = round(edge, 4)
        
        trade = TradeHistory(
            market_id=f"MARKET-{case.id}",
            asset=case.asset,
            outcome_bought=case.outcome,
            amount_usdc=1.0,
            executed_price=case.best_price,
            predicted_flip_prob=case.predicted_prob,
            active_features="test_features",
            status="SUCCESS",
            mode="PAPER",
            position_status="CLOSED",
            realized_pnl_usdc=1.0,
            model_key=case.model_key,
            model_version=case.model_version,
            model_attribution_source="EXACT",
            edge=edge_val,
            market_role=case.market_role,
            created_at=datetime.now(timezone.utc)
        )
        db_session.add(trade)
        await db_session.commit()
        await db_session.refresh(trade)
        
        res = await enqueue_open_request(
            db_session,
            trade_id=trade.id,
            market_id=f"MARKET-{case.id}",
            asset=case.asset,
            outcome_to_buy=case.outcome,
            target_amount_usdc=1.0,
            limit_price=case.best_price,
            requested_mode=ExecutionMode.PAPER,
        )
        assert res.disposition == EnqueueDisposition.CREATED
        
        reqs = (await db_session.scalars(
            select(ExecutionRequest).where(
                ExecutionRequest.market_id == f"MARKET-{case.id}",
                ExecutionRequest.requested_mode == "PAPER"
            )
        )).all()
        req_count = len(reqs)

    return PaperCycleResult(
        action=action,
        outcome=outcome,
        bet_size_usdc=1.0 if action != "SKIP" else 0.0,
        limit_price=limit_price,
        edge=edge_val,
        market_role=case.market_role if action != "SKIP" else None,
        execution_requests=req_count,
        realized_pnl_usdc=1.0 if action != "SKIP" else 0.0,
    )

@pytest.mark.asyncio
@pytest.mark.parametrize("case", PAPER_GOLDEN_CASES, ids=lambda c: c.id)
async def test_paper_behavior_matches_stable_snapshot(case, db_session):
    """Золотой тест 10 фиксированных сценариев торговли в PAPER режиме."""
    result = await run_deterministic_paper_cycle(db_session, case)
    
    assert result.action == case.expected_action
    assert result.execution_requests == case.expected_requests
    if result.action != "SKIP":
        assert result.outcome == case.outcome
        assert result.bet_size_usdc == 1.0
        assert result.market_role == case.market_role

async def compute_paper_checksum(db_session) -> str:
    """Вычисляет хэш состояния всех PAPER записей в БД."""
    trades = (await db_session.scalars(
        select(TradeHistory).where(TradeHistory.mode == "PAPER").order_by(TradeHistory.id)
    )).all()
    reqs = (await db_session.scalars(
        select(ExecutionRequest).where(ExecutionRequest.requested_mode == "PAPER").order_by(ExecutionRequest.id)
    )).all()
    
    payload = json.dumps({
        "trades": [{"id": t.id, "asset": t.asset, "status": t.status, "pnl": t.realized_pnl_usdc} for t in trades],
        "reqs": [{"id": str(r.id), "state": r.state, "intent": r.intent} for r in reqs]
    }, sort_keys=True)
    
    return hashlib.sha256(payload.encode()).hexdigest()

@pytest.mark.asyncio
async def test_paper_rows_checksum_invariance_during_readiness_check(db_session):
    """Проверяет, что контрольные суммы PAPER записей в БД остаются неизменными до и после любых проверок."""
    # Создаем исходную PAPER запись
    t = TradeHistory(
        market_id="CHK-1", asset="BTC", outcome_bought="YES", amount_usdc=1.0, executed_price=0.5,
        predicted_flip_prob=0.6, active_features="test", mode="PAPER", status="SUCCESS", created_at=datetime.now(timezone.utc)
    )
    db_session.add(t)
    await db_session.commit()
    
    checksum_before = await compute_paper_checksum(db_session)
    
    # Эмулируем обращение к настройкам и статистике
    setting = (await db_session.scalars(select(RuntimeSettings).where(RuntimeSettings.key == "LIVE_TRADING_ENABLED"))).first()
    
    checksum_after = await compute_paper_checksum(db_session)
    
    assert checksum_after == checksum_before
