import re

with open(r'C:\Users\orlov\.gemini\antigravity\scratch\flipoly\polyflip\api\trading_dashboard.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add _pnl_expr
pnl_expr_code = """
# COALESCE: новые записи имеют realized_pnl_usdc, старые — только pnl
_pnl_expr = sa.func.coalesce(
    TradeHistory.realized_pnl_usdc,
    TradeHistory.pnl,
)
"""
content = content.replace('_stats_cache = {}', pnl_expr_code + '\n_stats_cache = {}')

# 2. Add _base_conds inside get_trading_stats
base_conds_code = """
    def _base_conds(mode: str, with_cutoff: bool = True) -> list:
        conds = [
            TradeHistory.status == "SUCCESS",
            TradeHistory.position_status == "CLOSED",
            sa.or_(
                TradeHistory.realized_pnl_usdc.is_not(None),
                TradeHistory.pnl.is_not(None),
            ),
            TradeHistory.mode == mode,
        ]
        if with_cutoff and cutoff_dt:
            conds.append(TradeHistory.created_at >= cutoff_dt)
        return conds
"""
content = content.replace('        cutoff_dt = _utc_cutoff(timedelta(days=30))', '        cutoff_dt = _utc_cutoff(timedelta(days=30))\n' + base_conds_code)

# 3. Replace fetch_assets
content = re.sub(
    r'    async def fetch_assets\(\):[\s\S]*?group_by\(TradeHistory\.asset\)\n            return \(await s\.execute\(stmt\)\)\.all\(\)',
    r'''    async def fetch_assets():
        async with async_session() as s:
            conds = _base_conds(requested_mode)
            stmt = select(
                TradeHistory.asset,
                func.count(TradeHistory.id).label("total_trades"),
                func.sum(_pnl_expr).label("total_pnl"),
                func.sum(sa_case((_pnl_expr > 0, 1), else_=0)).label("wins")
            ).where(*conds).group_by(TradeHistory.asset)
            return (await s.execute(stmt)).all()''',
    content
)

# 4. Replace fetch_daily
content = re.sub(
    r'    async def fetch_daily\(\):[\s\S]*?group_by\(local_date\)\n            return \(await s\.execute\(stmt\)\)\.all\(\)',
    r'''    async def fetch_daily():
        async with async_session() as s:
            conds = _base_conds(requested_mode, with_cutoff=False)
            date_col = sa.func.coalesce(TradeHistory.closed_at, TradeHistory.created_at)
            local_date = cast(func.timezone('Asia/Ho_Chi_Minh', date_col), Date)
            if cutoff_dt:
                conds.append(
                    sa.or_(
                        TradeHistory.closed_at >= cutoff_dt,
                        sa.and_(
                            TradeHistory.closed_at.is_(None),
                            TradeHistory.created_at >= cutoff_dt
                        )
                    )
                )
            stmt = select(
                local_date.label("day"),
                func.sum(_pnl_expr).label("daily_pnl"),
                func.sum(sa_case((_pnl_expr > 0, 1), else_=0)).label("wins"),
                func.sum(sa_case((_pnl_expr <= 0, 1), else_=0)).label("losses")
            ).where(*conds).group_by(local_date)
            return (await s.execute(stmt)).all()''',
    content
)

# 5. Replace fetch_params
content = re.sub(
    r'    async def fetch_params\(\):[\s\S]*?where\(\*conds\)\n            return \(await s\.execute\(stmt\)\)\.first\(\)',
    r'''    async def fetch_params():
        async with async_session() as s:
            conds = _base_conds(requested_mode)
            stmt = select(
                func.avg(sa_case((_pnl_expr > 0, TradeHistory.executed_price), else_=None)).label("avg_win_price"),
                func.avg(sa_case((_pnl_expr <= 0, TradeHistory.executed_price), else_=None)).label("avg_loss_price"),
                func.avg(sa_case((_pnl_expr > 0, TradeHistory.predicted_flip_prob), else_=None)).label("avg_win_prob"),
                func.avg(sa_case((_pnl_expr <= 0, TradeHistory.predicted_flip_prob), else_=None)).label("avg_loss_prob")
            ).where(*conds)
            return (await s.execute(stmt)).first()''',
    content
)

# 6. Replace fetch_all_time_totals
content = re.sub(
    r'    async def fetch_all_time_totals\(\):[\s\S]*?TradeHistory\.mode == requested_mode\n            \)\n            return \(await s\.execute\(stmt\)\)\.first\(\)',
    r'''    async def fetch_all_time_totals():
        async with async_session() as s:
            conds = _base_conds(requested_mode, with_cutoff=False)
            stmt = select(
                func.count(TradeHistory.id).label("total_trades"),
                func.sum(_pnl_expr).label("total_pnl"),
                func.sum(sa_case((_pnl_expr > 0, 1), else_=0)).label("wins")
            ).where(*conds)
            return (await s.execute(stmt)).first()''',
    content
)

with open(r'C:\Users\orlov\.gemini\antigravity\scratch\flipoly\polyflip\api\trading_dashboard.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Patched!")
