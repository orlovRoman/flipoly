"""
LGBM Direction Accuracy Report - за последние 24 часа.
Секция 2 использует только funnel_log (без тяжёлого JOIN с crypto_candles).
"""
import asyncio
from datetime import datetime, timezone
from sqlalchemy import text
from polyflip.db.connection import async_session


def pct(n, d):
    return f"{100*n/d:.1f}%" if d else "-"


def print_header(title):
    print()
    print("=" * 76)
    print(f"  {title}")
    print("=" * 76)


def print_subheader(title):
    print(f"\n  -- {title} {'-' * max(1, 69 - len(title))}")


async def main():
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{'='*76}\n  LGBM DIRECTION ACCURACY REPORT  |  {now_utc}\n{'='*76}")

    async with async_session() as s:

        # ── 0. Общая статистика ───────────────────────────────────────────
        stats = (await s.execute(text("""
            SELECT
                COUNT(*) as total,
                COUNT(direction_value) as with_signal,
                COUNT(CASE WHEN direction_value IN ('UP','DOWN') THEN 1 END) as clear_signal,
                COUNT(CASE WHEN direction_value = 'UP'   THEN 1 END) as up_cnt,
                COUNT(CASE WHEN direction_value = 'DOWN' THEN 1 END) as down_cnt,
                COUNT(CASE WHEN direction_status = 'DIRECTION_NONE_FALLBACK_LR' THEN 1 END) as fallback_cnt,
                MIN(created_at) as first_log,
                MAX(created_at) as last_log
            FROM decision_funnel_log
            WHERE direction_model_key IS NOT NULL
              AND created_at >= NOW() - INTERVAL '24 hours';
        """))).mappings().one()

        print_subheader("0. Общая статистика сигналов (24h)")
        total = stats['with_signal']
        print(f"\n  Всего записей с LGBM-моделью  : {total:>8}")
        print(f"  Чёткий сигнал (UP / DOWN)      : {stats['clear_signal']:>8}  ({pct(stats['clear_signal'], total)})")
        print(f"    из них UP                    : {stats['up_cnt']:>8}  ({pct(stats['up_cnt'], stats['clear_signal'])})")
        print(f"    из них DOWN                  : {stats['down_cnt']:>8}  ({pct(stats['down_cnt'], stats['clear_signal'])})")
        print(f"  Fallback на LogReg (NONE)      : {stats['fallback_cnt']:>8}  ({pct(stats['fallback_cnt'], total)})")
        print(f"  Период                         : {stats['first_log']} → {stats['last_log']}")

        # ── 1. Сигналы по монете + режиму ────────────────────────────────
        by_coin = (await s.execute(text("""
            SELECT
                direction_model_key,
                direction_regime,
                direction_value,
                COUNT(*) as cnt,
                ROUND(AVG(direction_p_up)::numeric, 3)   as avg_p_up,
                ROUND(AVG(direction_p_down)::numeric, 3) as avg_p_down,
                -- сколько сигналов с высокой уверенностью (>0.60)
                COUNT(CASE WHEN direction_value = 'UP'   AND direction_p_up   > 0.60 THEN 1 END) as high_conf_up,
                COUNT(CASE WHEN direction_value = 'DOWN' AND direction_p_down > 0.60 THEN 1 END) as high_conf_down,
                -- g7 = crypto_confirm gate: прошёл ли LGBM сигнал в воронку
                COUNT(CASE WHEN g7_crypto_confirm = true  THEN 1 END) as g7_passed,
                COUNT(CASE WHEN g7_crypto_confirm = false THEN 1 END) as g7_blocked
            FROM decision_funnel_log
            WHERE direction_model_key IS NOT NULL
              AND direction_value IS NOT NULL
              AND created_at >= NOW() - INTERVAL '24 hours'
            GROUP BY direction_model_key, direction_regime, direction_value
            ORDER BY direction_model_key, direction_regime, direction_value;
        """))).mappings().all()

        print_subheader("1. Сигналы по модели / режиму / направлению")
        print(f"\n  {'Model key':<22} {'Regime':<10} {'Dir':<5} {'Count':>7}  {'p_up':>6}  {'p_dn':>6}  {'HighConf':>9}  {'G7pass':>7}  {'G7block':>8}")
        print(f"  {'-'*22} {'-'*10} {'-'*5} {'-'*7}  {'-'*6}  {'-'*6}  {'-'*9}  {'-'*7}  {'-'*8}")
        for r in by_coin:
            hc = r['high_conf_up'] + r['high_conf_down']
            print(
                f"  {str(r['direction_model_key'] or '-'):<22} "
                f"{str(r['direction_regime'] or '-'):<10} "
                f"{str(r['direction_value'] or '-'):<5} "
                f"{r['cnt']:>7}  "
                f"{str(r['avg_p_up'] or '-'):>6}  "
                f"{str(r['avg_p_down'] or '-'):>6}  "
                f"{pct(hc, r['cnt']):>9}  "
                f"{r['g7_passed']:>7}  "
                f"{r['g7_blocked']:>8}"
            )

        # ── 2. Итог по каждой монете: уверенность и G7 ───────────────────
        print_subheader("2. Итог по монете: uверенность модели и G7-фильтр")
        by_asset = (await s.execute(text("""
            SELECT
                SPLIT_PART(direction_model_key, '_', 1) as coin,
                COUNT(CASE WHEN direction_value IN ('UP','DOWN') THEN 1 END) as clear,
                COUNT(CASE WHEN direction_value = 'UP'   THEN 1 END) as up_cnt,
                COUNT(CASE WHEN direction_value = 'DOWN' THEN 1 END) as down_cnt,
                ROUND(AVG(CASE WHEN direction_value = 'UP'   THEN direction_p_up   END)::numeric, 3) as avg_conf_up,
                ROUND(AVG(CASE WHEN direction_value = 'DOWN' THEN direction_p_down END)::numeric, 3) as avg_conf_down,
                COUNT(CASE WHEN g7_crypto_confirm = true  THEN 1 END) as g7_ok,
                COUNT(CASE WHEN g7_crypto_confirm = false THEN 1 END) as g7_no,
                COUNT(CASE WHEN final_action = 'BUY' THEN 1 END) as buys
            FROM decision_funnel_log
            WHERE direction_model_key IS NOT NULL
              AND created_at >= NOW() - INTERVAL '24 hours'
            GROUP BY SPLIT_PART(direction_model_key, '_', 1)
            ORDER BY coin;
        """))).mappings().all()

        print(f"\n  {'Coin':<10} {'UP':>6} {'DOWN':>6}  {'AvgConf UP':>11} {'AvgConf DN':>11}  {'G7 ok':>6} {'G7 no':>6}  {'BUYs':>6}")
        print(f"  {'-'*10} {'-'*6} {'-'*6}  {'-'*11} {'-'*11}  {'-'*6} {'-'*6}  {'-'*6}")
        for r in by_asset:
            print(
                f"  {r['coin']:<10} {r['up_cnt']:>6} {r['down_cnt']:>6}  "
                f"{str(r['avg_conf_up'] or '-'):>11} {str(r['avg_conf_down'] or '-'):>11}  "
                f"{r['g7_ok']:>6} {r['g7_no']:>6}  "
                f"{r['buys']:>6}"
            )

        # ── 3. Проверка: когда LGBM дал сигнал, торговля шла лучше? ──────
        print_subheader("3. Эффективность: G7 passed vs итог сделки")
        effectiveness = (await s.execute(text("""
            SELECT
                g7_crypto_confirm,
                final_action,
                COUNT(*) as cnt
            FROM decision_funnel_log
            WHERE direction_model_key IS NOT NULL
              AND direction_value IN ('UP', 'DOWN')
              AND created_at >= NOW() - INTERVAL '24 hours'
            GROUP BY g7_crypto_confirm, final_action
            ORDER BY g7_crypto_confirm, final_action;
        """))).mappings().all()

        print(f"\n  {'G7 passed':<12} {'Final action':<15} {'Count':>8}")
        print(f"  {'-'*12} {'-'*15} {'-'*8}")
        for r in effectiveness:
            g7 = "YES" if r['g7_crypto_confirm'] else ("NO" if r['g7_crypto_confirm'] is False else "NULL")
            print(f"  {g7:<12} {str(r['final_action']):<15} {r['cnt']:>8}")

    print(f"\n{'='*76}\n")


if __name__ == "__main__":
    asyncio.run(main())
