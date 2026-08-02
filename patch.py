import re

with open('polyflip/api/execution_api.py', 'r', encoding='utf-8') as f:
    code = f.read()

pos_dict_code = '''
async def _get_positions_dict(db: AsyncSession, mode: str):
    stmt = (
        select(TradeHistory, LiveMarket)
        .outerjoin(LiveMarket, TradeHistory.market_id == LiveMarket.market_id)
        .where(TradeHistory.mode == mode)
        .order_by(TradeHistory.created_at.desc())
        .limit(200)
    )
    res = await db.execute(stmt)
    rows = res.all()

    result = {
        "tradable": [],
        "resolved": [],
        "archive": []
    }
    
    for trade, market in rows:
        available_actions = {
            "close": False,
            "reconcile_resolution": False,
            "redeem": False,
            "reconcile_redemption": False,
        }
        
        if trade.position_status in {"OPEN", "PARTIALLY_CLOSED"} and trade.remaining_shares and trade.remaining_shares > 0:
            if market and getattr(market, 'resolution_status', 'PENDING') == "PENDING" and getattr(market, 'trading_status', 'UNKNOWN') == "TRADABLE" and getattr(market, 'accepting_orders', False):
                available_actions["close"] = True

        if trade.position_status in {"OPEN", "PARTIALLY_CLOSED"}:
            available_actions["reconcile_resolution"] = True
            
        if trade.position_status in {"RESOLVED_REDEEMABLE", "REDEEMING", "REDEMPTION_UNKNOWN"}:
            available_actions["reconcile_resolution"] = True

        if trade.position_status == "RESOLVED_REDEEMABLE" and getattr(trade, 'redemption_status', 'NOT_REQUIRED') in {"PENDING", "FAILED"} and getattr(trade, 'redeemable_shares', 0) and getattr(trade, 'redeemable_shares', 0) > 0 and not getattr(trade, 'redemption_tx_hash', None):
            available_actions["redeem"] = True

        if trade.position_status in {"REDEEMING", "REDEMPTION_UNKNOWN", "RESOLVED_REDEEMABLE"}:
            available_actions["reconcile_redemption"] = True

        item = {
            "id": trade.id,
            "market_id": trade.market_id,
            "asset": trade.asset,
            "outcome_bought": trade.outcome_bought,
            "mode": trade.mode,
            "entry_filled_shares": float(trade.entry_filled_shares or 0),
            "entry_cost_usdc": float(trade.entry_cost_usdc or 0),
            "remaining_shares": float(trade.remaining_shares or 0),
            "realized_pnl_usdc": float(trade.realized_pnl_usdc or 0),
            "position_status": trade.position_status,
            "redemption_status": getattr(trade, 'redemption_status', 'NOT_REQUIRED'),
            "stop_loss_status": trade.stop_loss_status,
            "take_profit_status": trade.take_profit_status,
            "created_at": trade.created_at.isoformat() if trade.created_at else None,
            "market": {
                "trading_status": getattr(market, 'trading_status', 'UNKNOWN') if market else "UNKNOWN",
                "resolution_status": getattr(market, 'resolution_status', 'PENDING') if market else "PENDING",
                "final_outcome": getattr(market, 'final_outcome', None) if market else None,
            },
            "available_actions": available_actions,
        }
        
        if trade.position_status in ("OPEN", "PARTIALLY_CLOSED", "ENTRY_FAILED"):
            result["tradable"].append(item)
        elif trade.position_status in ("RESOLVED_REDEEMABLE", "REDEEMING", "REDEMPTION_UNKNOWN"):
            result["resolved"].append(item)
        else:
            result["archive"].append(item)

    return {"positions": result}

@router.get("/positions")
async def get_live_trading_positions(
    mode: Optional[str] = Query(
        None, description="Фильтр по режиму: PAPER, SHADOW, LIVE"
    ),
    db: AsyncSession = Depends(get_db_session),
):
    settings = ExecutionSettings()
    effective_mode = mode or settings.execution_mode.value
    return await _get_positions_dict(db, effective_mode)

'''

# Remove old get_live_trading_positions
code = re.sub(
    r'@router\.get\(\"/positions\"\).*?return result',
    pos_dict_code,
    code, flags=re.DOTALL
)

# Update get_live_dashboard
code = re.sub(
    r'\"positions\": serialize_positions\(active_positions\),\s*\"failed_entries\": serialize_positions\(failed_entries\),',
    '"positions": (await _get_positions_dict(db, "LIVE")).get("positions"),',
    code
)

# Update /close endpoint
close_update = '''
    if trade.position_status not in {"OPEN", "PARTIALLY_CLOSED"}:
        raise HTTPException(
            status_code=409, detail="Позиция не является торгуемой"
        )

    market = await db.scalar(select(LiveMarket).where(LiveMarket.market_id == trade.market_id))
    if not market:
        raise HTTPException(status_code=409, detail="Рынок не найден")
        
    if getattr(market, 'resolution_status', 'PENDING') != "PENDING":
        raise HTTPException(status_code=409, detail="Рынок уже завершён")
        
    if getattr(market, 'trading_status', 'TRADABLE') != "TRADABLE" or not getattr(market, 'accepting_orders', True):
        raise HTTPException(status_code=409, detail="Рынок не принимает ордера")

    if not trade.remaining_shares or trade.remaining_shares <= 0:
        raise HTTPException(
            status_code=409, detail="Нет доступных долей для закрытия"
        )
'''

# Find the place to inject
import sys
code = re.sub(
    r'if trade\.position_status not in \{\"OPEN\", \"PARTIALLY_CLOSED\"\}:.*?if not trade\.remaining_shares or trade\.remaining_shares <= 0:\s*raise HTTPException\(\s*status_code=409, detail=\"[^\"]*\"\s*\)',
    close_update.strip(),
    code, flags=re.DOTALL
)

new_endpoints = '''
@router.post("/live/positions/{trade_id}/reconcile-resolution")
async def api_reconcile_resolution(trade_id: int, db: AsyncSession = Depends(get_db_session)):
    from polyflip.execution.live_settlement_service import reconcile_live_resolution, LivePositionNotFound, MarketNotResolved
    try:
        updated_trade = await reconcile_live_resolution(db, trade_id)
        await db.commit()
        return {"status": "ok", "position_status": updated_trade.position_status}
    except LivePositionNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except MarketNotResolved as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.exception("reconcile_error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/live/positions/{trade_id}/redeem")
async def api_redeem(trade_id: int, db: AsyncSession = Depends(get_db_session)):
    trade = await db.scalar(select(TradeHistory).where(TradeHistory.id == trade_id).with_for_update())
    if not trade or trade.mode != "LIVE":
        raise HTTPException(status_code=404, detail="Позиция не найдена")
    if getattr(trade, 'position_status', '') != "RESOLVED_REDEEMABLE":
        raise HTTPException(status_code=409, detail="Позиция не подлежит погашению")
        
    trade.position_status = "REDEEMING"
    trade.redemption_status = "SUBMITTED"
    await db.commit()
    return {"status": "ok"}

@router.post("/live/positions/{trade_id}/reconcile-redemption")
async def api_reconcile_redemption(trade_id: int, db: AsyncSession = Depends(get_db_session)):
    trade = await db.scalar(select(TradeHistory).where(TradeHistory.id == trade_id).with_for_update())
    if not trade or trade.mode != "LIVE":
        raise HTTPException(status_code=404, detail="Позиция не найдена")
    
    # Заглушка
    return {"status": "ok", "redemption_status": getattr(trade, 'redemption_status', 'NOT_REQUIRED')}
'''
if '/reconcile-resolution' not in code:
    code += "\\n" + new_endpoints

with open('polyflip/api/execution_api.py', 'w', encoding='utf-8') as f:
    f.write(code)
print("done")
