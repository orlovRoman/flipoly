"""
Favorite vs Outsider Analytical Report - Last 24 Hours.
"""
import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from sqlalchemy import text
from polyflip.db.connection import async_session


def fmt_pnl(v):
    if v is None:
        return "-"
    f = float(v)
    sign = "+" if f > 0 else ""
    return f"{sign}{f:.2f}"


def fmt(v, digits=2):
    if v is None:
        return "-"
    if isinstance(v, (Decimal, float)):
        return f"{float(v):.{digits}f}"
    return str(v)


def wr_str(wins, closed):
    if not closed:
        return "-"
    return f"{100 * wins / closed:.1f}%"


def print_header(title):
    print()
    print("=" * 76)
    print(f"  {title}")
    print("=" * 76)


def print_subheader(title):
    print(f"\n  -- {title} {'-' * max(1, 69 - len(title))}")


async def main():
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{'='*76}\n  FAVORITE VS OUTSIDER ANALYTICS (LAST 24 HOURS)  |  {now_utc}\n{'='*76}")

    async with async_session() as s:

        # ── 1. Общая сводка (24h) ───────────────────────────────────────
        summary = (await s.execute(text("""
            SELECT
                COALESCE(market_role, 'UNKNOWN') as role,
                COUNT(*) as total_records,
                COUNT(CASE WHEN status = 'SUCCESS' THEN 1 END) as executed_trades,
                COUNT(CASE WHEN status = 'SUCCESS' AND pnl > 0 THEN 1 END) as win_count,
                COUNT(CASE WHEN status = 'SUCCESS' AND pnl < 0 THEN 1 END) as loss_count,
                COUNT(CASE WHEN status = 'SUCCESS' AND pnl = 0 THEN 1 END) as zero_count,
                COUNT(CASE WHEN status = 'SUCCESS' AND position_status = 'OPEN' THEN 1 END) as open_count,
                ROUND(CAST(COALESCE(SUM(CASE WHEN status = 'SUCCESS' THEN pnl ELSE 0 END), 0) AS numeric), 2) as total_pnl,
                ROUND(CAST(COALESCE(SUM(CASE WHEN status = 'SUCCESS' THEN amount_usdc ELSE 0 END), 0) AS numeric), 2) as total_volume,
                ROUND(CAST(AVG(CASE WHEN status = 'SUCCESS' THEN executed_price END) AS numeric), 4) as avg_entry_price,
                ROUND(CAST(AVG(CASE WHEN status = 'SUCCESS' THEN amount_usdc END) AS numeric), 2) as avg_bet_size
            FROM trade_history
            WHERE created_at >= NOW() - INTERVAL '24 hours'
            GROUP BY COALESCE(market_role, 'UNKNOWN')
            ORDER BY role;
        """))).mappings().all()

        print_subheader("1. ОБЩАЯ СВОДКА ЗА 24 ЧАСА")
        print(f"\n  {'Role':<12} {'Records':>8} {'Trades':>8} {'Wins':>6} {'Loss':>6} {'Open':>6} {'WR%':>7} {'AvgPrice':>9} {'AvgBet':>8} {'Volume':>10} {'PnL (USDC)':>12}")
        print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*9} {'-'*8} {'-'*10} {'-'*12}")

        total_all_trades = 0
        total_all_wins = 0
        total_all_losses = 0
        total_all_pnl = 0.0
        total_all_vol = 0.0

        for r in summary:
            exec_t = int(r["executed_trades"] or 0)
            wins = int(r["win_count"] or 0)
            losses = int(r["loss_count"] or 0)
            closed_t = wins + losses
            wr = wr_str(wins, closed_t)
            pnl = float(r["total_pnl"] or 0)
            vol = float(r["total_volume"] or 0)

            total_all_trades += exec_t
            total_all_wins += wins
            total_all_losses += losses
            total_all_pnl += pnl
            total_all_vol += vol

            print(
                f"  {r['role']:<12} "
                f"{r['total_records']:>8} "
                f"{exec_t:>8} "
                f"{wins:>6} "
                f"{losses:>6} "
                f"{r['open_count']:>6} "
                f"{wr:>7} "
                f"{fmt(r['avg_entry_price'], 4):>9} "
                f"{fmt(r['avg_bet_size'], 2):>8} "
                f"{fmt(vol, 2):>10} "
                f"{fmt_pnl(pnl):>12}"
            )

        print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*9} {'-'*8} {'-'*10} {'-'*12}")
        total_closed = total_all_wins + total_all_losses
        print(
            f"  {'ИТОГО':<12} "
            f"{sum(r['total_records'] for r in summary):>8} "
            f"{total_all_trades:>8} "
            f"{total_all_wins:>6} "
            f"{total_all_losses:>6} "
            f"{sum(r['open_count'] for r in summary):>6} "
            f"{wr_str(total_all_wins, total_closed):>7} "
            f"{'-':>9} "
            f"{'-':>8} "
            f"{fmt(total_all_vol, 2):>10} "
            f"{fmt_pnl(total_all_pnl):>12}"
        )

        # ── 2. Разбивка по активам (Asset x Role) ───────────────────────
        by_asset = (await s.execute(text("""
            SELECT
                asset,
                COALESCE(market_role, 'UNKNOWN') as role,
                COUNT(CASE WHEN status = 'SUCCESS' THEN 1 END) as executed_trades,
                COUNT(CASE WHEN status = 'SUCCESS' AND pnl > 0 THEN 1 END) as win_count,
                COUNT(CASE WHEN status = 'SUCCESS' AND pnl < 0 THEN 1 END) as loss_count,
                COUNT(CASE WHEN status = 'SUCCESS' AND position_status = 'OPEN' THEN 1 END) as open_count,
                ROUND(CAST(COALESCE(SUM(CASE WHEN status = 'SUCCESS' THEN pnl ELSE 0 END), 0) AS numeric), 2) as total_pnl,
                ROUND(CAST(COALESCE(SUM(CASE WHEN status = 'SUCCESS' THEN amount_usdc ELSE 0 END), 0) AS numeric), 2) as total_volume,
                ROUND(CAST(AVG(CASE WHEN status = 'SUCCESS' THEN executed_price END) AS numeric), 4) as avg_entry_price
            FROM trade_history
            WHERE created_at >= NOW() - INTERVAL '24 hours'
            GROUP BY asset, COALESCE(market_role, 'UNKNOWN')
            ORDER BY asset, role;
        """))).mappings().all()

        print_subheader("2. РАЗБИВКА ПО АКТИВАМ (ASSET x ROLE)")
        print(f"\n  {'Asset':<10} {'Role':<10} {'Trades':>7} {'Wins':>5} {'Loss':>5} {'Open':>5} {'WR%':>7} {'AvgPrice':>9} {'Volume':>9} {'PnL (USDC)':>11}")
        print(f"  {'-'*10} {'-'*10} {'-'*7} {'-'*5} {'-'*5} {'-'*5} {'-'*7} {'-'*9} {'-'*9} {'-'*11}")

        for r in by_asset:
            exec_t = int(r["executed_trades"] or 0)
            wins = int(r["win_count"] or 0)
            losses = int(r["loss_count"] or 0)
            closed_t = wins + losses
            wr = wr_str(wins, closed_t)
            print(
                f"  {r['asset']:<10} "
                f"{r['role']:<10} "
                f"{exec_t:>7} "
                f"{wins:>5} "
                f"{losses:>5} "
                f"{r['open_count']:>5} "
                f"{wr:>7} "
                f"{fmt(r['avg_entry_price'], 4):>9} "
                f"{fmt(r['total_volume'], 2):>9} "
                f"{fmt_pnl(r['total_pnl']):>11}"
            )

        # ── 3. Стратегии и Режимы Торговли (Strategy x Role) ─────────────
        by_strat = (await s.execute(text("""
            SELECT
                COALESCE(strategy_type, mode, 'N/A') as strategy,
                COALESCE(market_role, 'UNKNOWN') as role,
                COUNT(CASE WHEN status = 'SUCCESS' THEN 1 END) as executed_trades,
                COUNT(CASE WHEN status = 'SUCCESS' AND pnl > 0 THEN 1 END) as win_count,
                COUNT(CASE WHEN status = 'SUCCESS' AND pnl < 0 THEN 1 END) as loss_count,
                ROUND(CAST(COALESCE(SUM(CASE WHEN status = 'SUCCESS' THEN pnl ELSE 0 END), 0) AS numeric), 2) as total_pnl,
                ROUND(CAST(COALESCE(SUM(CASE WHEN status = 'SUCCESS' THEN amount_usdc ELSE 0 END), 0) AS numeric), 2) as total_volume
            FROM trade_history
            WHERE created_at >= NOW() - INTERVAL '24 hours'
            GROUP BY COALESCE(strategy_type, mode, 'N/A'), COALESCE(market_role, 'UNKNOWN')
            ORDER BY strategy, role;
        """))).mappings().all()

        print_subheader("3. СТРАТЕГИИ И РЕЖИМЫ (STRATEGY x ROLE)")
        print(f"\n  {'Strategy':<20} {'Role':<10} {'Trades':>7} {'Wins':>5} {'Loss':>5} {'WR%':>7} {'Volume':>9} {'PnL (USDC)':>11}")
        print(f"  {'-'*20} {'-'*10} {'-'*7} {'-'*5} {'-'*5} {'-'*7} {'-'*9} {'-'*11}")

        for r in by_strat:
            exec_t = int(r["executed_trades"] or 0)
            wins = int(r["win_count"] or 0)
            losses = int(r["loss_count"] or 0)
            closed_t = wins + losses
            wr = wr_str(wins, closed_t)
            print(
                f"  {r['strategy']:<20} "
                f"{r['role']:<10} "
                f"{exec_t:>7} "
                f"{wins:>5} "
                f"{losses:>5} "
                f"{wr:>7} "
                f"{fmt(r['total_volume'], 2):>9} "
                f"{fmt_pnl(r['total_pnl']):>11}"
            )

        # ── 4. Причины выхода / SL / TP / Settlement ─────────────────────
        by_exit = (await s.execute(text("""
            SELECT
                COALESCE(market_role, 'UNKNOWN') as role,
                COALESCE(exit_reason, position_status, 'UNKNOWN') as exit_type,
                COUNT(*) as cnt,
                ROUND(CAST(COALESCE(SUM(pnl), 0) AS numeric), 2) as pnl
            FROM trade_history
            WHERE created_at >= NOW() - INTERVAL '24 hours'
              AND status = 'SUCCESS'
            GROUP BY COALESCE(market_role, 'UNKNOWN'), COALESCE(exit_reason, position_status, 'UNKNOWN')
            ORDER BY role, cnt DESC;
        """))).mappings().all()

        print_subheader("4. ПРИЧИНЫ ЗАКРЫТИЯ ПОЗИЦИЙ (EXIT REASONS)")
        print(f"\n  {'Role':<12} {'Exit Reason / Status':<28} {'Count':>7} {'PnL (USDC)':>12}")
        print(f"  {'-'*12} {'-'*28} {'-'*7} {'-'*12}")

        for r in by_exit:
            print(
                f"  {r['role']:<12} "
                f"{r['exit_type']:<28} "
                f"{r['cnt']:>7} "
                f"{fmt_pnl(r['pnl']):>12}"
            )

    print(f"\n{'='*76}\n")


if __name__ == "__main__":
    asyncio.run(main())
