import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from sqlalchemy import text
from polyflip.db.connection import async_session


def get_model_type(asset: str) -> str:
    """Classifies model by asset name. Temporary: no model_type column in DB yet."""
    return "lgbm" if "USDT" in asset else "logreg"


def fmt(v, digits=4):
    if v is None:
        return "-"
    if isinstance(v, (Decimal, float)):
        return f"{float(v):.{digits}f}"
    if isinstance(v, datetime):
        return v.strftime("%Y-%m-%d %H:%M UTC")
    return str(v)


def fmt_pnl(v):
    if v is None:
        return "-"
    f = float(v)
    sign = "+" if f > 0 else ""
    return f"{sign}{f:.2f}"


def wr_str(wins, trades):
    return "-" if not trades else f"{100 * wins / trades:.1f}%"


def print_header(title):
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def print_subheader(title):
    print(f"\n  -- {title} {'-' * max(1, 65 - len(title))}")


def print_registry_table(models):
    print(
        f"  {'Asset':<22} {'Ver':>4}  {'Act':>5}  {'Accuracy':>9}  "
        f"{'Baseline':>9}  {'Thr':>6}  {'F1':>6}  {'BkPnL':>8}  Trained"
    )
    print(
        f"  {'-'*22} {'-'*4}  {'-'*5}  {'-'*9}  {'-'*9}  "
        f"{'-'*6}  {'-'*6}  {'-'*8}  {'-'*16}"
    )
    for m in models:
        act = "* ACT" if m["is_active"] else "     "
        print(
            f"  {m['asset']:<22} {fmt(m['version']):>4}  {act}  "
            f"{fmt(m['accuracy'], 4):>9}  {fmt(m['baseline'], 4):>9}  "
            f"{fmt(m['decision_threshold'], 3):>6}  {fmt(m['f1_at_threshold'], 3):>6}  "
            f"{fmt_pnl(m['backtest_pnl']):>8}  {fmt(m['trained_at'])}"
        )
        qgr = m.get("quality_gate_reasons")
        if qgr and qgr not in ("[]", None):
            print(f"  {'':>22}  ! QGate: {str(qgr)[:55]}")


def print_trades_table(rows, label_24h=False):
    pc = "PnL(24h)" if label_24h else "PnL(all)"
    print(
        f"  {'Asset':<22} {'Model':>6}  {'Records':>8}  {'Trades':>7}  "
        f"{'Wins':>5}  {'Loss':>5}  {'WR':>7}  {pc:>10}  {'Volume':>10}"
    )
    print(
        f"  {'-'*22} {'-'*6}  {'-'*8}  {'-'*7}  {'-'*5}  "
        f"{'-'*5}  {'-'*7}  {'-'*10}  {'-'*10}"
    )
    for r in rows:
        et = int(r["executed_trades"] or 0)
        w  = int(r["win_count"] or 0)
        l  = int(r["loss_count"] or 0)
        vol = fmt(r.get("total_volume"), 2) if not label_24h else "-"
        print(
            f"  {r['asset']:<22} {r['model_version']:>6}  "
            f"{r['total_records']:>8}  {et:>7}  {w:>5}  {l:>5}  "
            f"{wr_str(w, et):>7}  {fmt_pnl(r['total_pnl']):>10}  {vol:>10}"
        )


async def main():
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{'='*72}\n  ML MODEL AUDIT REPORT  |  {now_utc}\n{'='*72}")

    async with async_session() as s:
        models = [dict(x) for x in (await s.execute(text("""
            SELECT id, asset, version, is_active,
                   accuracy, baseline,
                   decision_threshold, f1_at_threshold,
                   backtest_pnl, backtest_wr,
                   quality_gate_passed, quality_gate_reasons,
                   activation_source, trained_at
            FROM model_registry
            ORDER BY asset, version DESC;
        """))).mappings()]

        trades = [dict(x) for x in (await s.execute(text("""
            SELECT
                asset,
                COALESCE(CAST(model_version AS text), 'N/A') as model_version,
                COUNT(*) as total_records,
                COUNT(CASE WHEN status = 'SUCCESS' THEN 1 END) as executed_trades,
                COUNT(CASE WHEN pnl > 0 THEN 1 END) as win_count,
                COUNT(CASE WHEN pnl < 0 THEN 1 END) as loss_count,
                ROUND(CAST(COALESCE(SUM(pnl), 0) AS numeric), 2) as total_pnl,
                ROUND(CAST(COALESCE(SUM(amount_usdc), 0) AS numeric), 2) as total_volume
            FROM trade_history
            GROUP BY asset, COALESCE(CAST(model_version AS text), 'N/A')
            ORDER BY asset, total_pnl DESC;
        """))).mappings()]

        trades24 = [dict(x) for x in (await s.execute(text("""
            SELECT
                asset,
                COALESCE(CAST(model_version AS text), 'N/A') as model_version,
                COUNT(*) as total_records,
                COUNT(CASE WHEN status = 'SUCCESS' THEN 1 END) as executed_trades,
                COUNT(CASE WHEN pnl > 0 THEN 1 END) as win_count,
                COUNT(CASE WHEN pnl < 0 THEN 1 END) as loss_count,
                ROUND(CAST(COALESCE(SUM(pnl), 0) AS numeric), 2) as total_pnl
            FROM trade_history
            WHERE created_at >= NOW() - INTERVAL '24 hours'
            GROUP BY asset, COALESCE(CAST(model_version AS text), 'N/A')
            ORDER BY asset, total_pnl DESC;
        """))).mappings()]

    logreg_models   = [m for m in models   if get_model_type(m["asset"]) == "logreg"]
    lgbm_models     = [m for m in models   if get_model_type(m["asset"]) == "lgbm"]
    logreg_trades   = [r for r in trades   if get_model_type(r["asset"]) == "logreg"]
    lgbm_trades     = [r for r in trades   if get_model_type(r["asset"]) == "lgbm"]
    logreg_trades24 = [r for r in trades24 if get_model_type(r["asset"]) == "logreg"]
    lgbm_trades24   = [r for r in trades24 if get_model_type(r["asset"]) == "lgbm"]

    active_logreg = [m for m in logreg_models if m["is_active"]]
    active_lgbm   = [m for m in lgbm_models   if m["is_active"]]

    # === BLOCK 1: LogReg ===
    print_header(f"BLOCK 1 -- LogReg (Polymarket)  |  active: {len(active_logreg)}")
    print_subheader("1.1 Active models")
    print_registry_table(active_logreg) if active_logreg else print("  None.")
    print_subheader("1.2 All versions")
    print_registry_table(logreg_models)
    print_subheader("1.3 All-Time trades")
    print_trades_table(logreg_trades) if logreg_trades else print("  No data.")
    print_subheader("1.4 Last 24h trades")
    print_trades_table(logreg_trades24, label_24h=True) if logreg_trades24 else print("  No trades in last 24h.")

    # === BLOCK 2: LightGBM ===
    print_header(f"BLOCK 2 -- LightGBM (Crypto OHLCV)  |  active: {len(active_lgbm)}")
    print_subheader("2.1 Active models")
    print_registry_table(active_lgbm) if active_lgbm else print("  None.")
    print_subheader("2.2 All versions")
    print_registry_table(lgbm_models) if lgbm_models else print("  No LGBM models.")
    print_subheader("2.3 All-Time trades")
    print_trades_table(lgbm_trades) if lgbm_trades else print("  No data.")
    print_subheader("2.4 Last 24h trades")
    print_trades_table(lgbm_trades24, label_24h=True) if lgbm_trades24 else print("  No trades in last 24h.")

    print(f"\n{'='*72}\n")


if __name__ == "__main__":
    asyncio.run(main())
