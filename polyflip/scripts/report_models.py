"""
Аналитика по моделям (ML Model Audit Report).
Выводит:
 1. Реестр моделей (без blob) — параметры, качество, активность
 2. Торговая статистика по версиям модели (all-time)
 3. Торговая статистика за последние 24 часа
"""
import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from sqlalchemy import text
from polyflip.db.connection import async_session


def fmt(v, digits=4):
    if v is None:
        return "—"
    if isinstance(v, (Decimal, float)):
        return f"{float(v):.{digits}f}"
    if isinstance(v, datetime):
        return v.strftime("%Y-%m-%d %H:%M UTC")
    return str(v)


def fmt_pnl(v):
    if v is None:
        return "—"
    f = float(v)
    sign = "+" if f > 0 else ""
    return f"{sign}{f:.2f}"


def print_section(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


async def main():
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{'='*70}")
    print(f"  ML MODEL AUDIT REPORT  |  {now_utc}")
    print(f"{'='*70}")

    async with async_session() as s:

        # ── 1. Реестр моделей ─────────────────────────────────────────────
        m_stmt = text("""
            SELECT
                id, asset, version, is_active,
                accuracy, baseline,
                decision_threshold, decision_threshold_down,
                train_samples, validation_samples, positive_rate,
                precision_at_threshold, recall_at_threshold,
                f1_at_threshold, brier_score, ece,
                backtest_pnl, backtest_trades, backtest_wr,
                quality_gate_passed, quality_gate_reasons,
                activation_source, activated_at, trained_at, features
            FROM model_registry
            ORDER BY asset, version DESC;
        """)
        models_res = await s.execute(m_stmt)
        models = [dict(x) for x in models_res.mappings()]

        # ── 2. Торговая статистика (all-time) ─────────────────────────────
        t_stmt = text("""
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
        """)
        trades_res = await s.execute(t_stmt)
        trades = [dict(x) for x in trades_res.mappings()]

        # ── 3. Торговая статистика (24h) ──────────────────────────────────
        t24_stmt = text("""
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
        """)
        t24_res = await s.execute(t24_stmt)
        trades24 = [dict(x) for x in t24_res.mappings()]

    # ── Вывод: Реестр моделей ─────────────────────────────────────────────
    print_section("1. РЕЕСТР МОДЕЛЕЙ")

    # Группируем по asset для читаемости
    assets = sorted(set(m["asset"] for m in models))
    for asset in assets:
        asset_models = [m for m in models if m["asset"] == asset]
        active = [m for m in asset_models if m["is_active"]]
        inactive = [m for m in asset_models if not m["is_active"]]

        print(f"\n▶ {asset}  (всего моделей: {len(asset_models)}, активных: {len(active)})")
        print(f"  {'Ver':>4}  {'Active':>6}  {'Accuracy':>9}  {'Baseline':>9}  "
              f"{'Thr↑':>6}  {'Thr↓':>6}  {'F1':>6}  {'BkPnL':>8}  {'QGate':>6}  Trained")

        for m in asset_models:
            qg = "✓" if m["quality_gate_passed"] else ("✗" if m["quality_gate_passed"] is False else "—")
            active_mark = "✓ ACT" if m["is_active"] else "      "
            trained = fmt(m["trained_at"])
            print(
                f"  {fmt(m['version']):>4}  {active_mark:>6}  "
                f"{fmt(m['accuracy'], 4):>9}  {fmt(m['baseline'], 4):>9}  "
                f"{fmt(m['decision_threshold'], 3):>6}  {fmt(m['decision_threshold_down'], 3):>6}  "
                f"{fmt(m['f1_at_threshold'], 3):>6}  {fmt_pnl(m['backtest_pnl']):>8}  "
                f"{qg:>6}  {trained}"
            )
            # Фичи
            if m["features"]:
                feats = m["features"]
                if len(feats) > 60:
                    feats = feats[:57] + "..."
                print(f"         features: {feats}")
            # Причины quality gate
            if m["quality_gate_reasons"] and m["quality_gate_reasons"] != "[]":
                print(f"         QGate reasons: {m['quality_gate_reasons']}")

    # ── Вывод: All-time торговля ──────────────────────────────────────────
    print_section("2. ТОРГОВАЯ СТАТИСТИКА ПО МОДЕЛЯМ (ALL-TIME)")

    if not trades:
        print("  Нет данных.")
    else:
        print(f"\n  {'Asset':>8}  {'Model':>6}  {'Records':>8}  {'Trades':>7}  "
              f"{'Wins':>5}  {'Loss':>5}  {'WR%':>6}  {'PnL':>9}  {'Volume':>10}")
        for r in trades:
            exec_t = int(r["executed_trades"] or 0)
            wins = int(r["win_count"] or 0)
            wr = f"{100*wins/exec_t:.1f}" if exec_t > 0 else "—"
            print(
                f"  {r['asset']:>8}  {r['model_version']:>6}  "
                f"{r['total_records']:>8}  {exec_t:>7}  "
                f"{wins:>5}  {int(r['loss_count'] or 0):>5}  {wr:>6}  "
                f"{fmt_pnl(r['total_pnl']):>9}  {fmt(r['total_volume'], 2):>10}"
            )

    # ── Вывод: 24h торговля ───────────────────────────────────────────────
    print_section("3. ТОРГОВАЯ СТАТИСТИКА ЗА ПОСЛЕДНИЕ 24 ЧАСА")

    if not trades24:
        print("  Нет сделок за последние 24 часа.")
    else:
        print(f"\n  {'Asset':>8}  {'Model':>6}  {'Records':>8}  {'Trades':>7}  "
              f"{'Wins':>5}  {'Loss':>5}  {'WR%':>6}  {'PnL(24h)':>10}")
        for r in trades24:
            exec_t = int(r["executed_trades"] or 0)
            wins = int(r["win_count"] or 0)
            wr = f"{100*wins/exec_t:.1f}" if exec_t > 0 else "—"
            print(
                f"  {r['asset']:>8}  {r['model_version']:>6}  "
                f"{r['total_records']:>8}  {exec_t:>7}  "
                f"{wins:>5}  {int(r['loss_count'] or 0):>5}  {wr:>6}  "
                f"{fmt_pnl(r['total_pnl']):>10}"
            )

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    asyncio.run(main())
