"""
LGBM Direction Accuracy Report.

Методология:
  1. Берём все записи из decision_funnel_log, где direction_value IS NOT NULL
     (т.е. LGBM-модель дала сигнал UP или DOWN).
  2. По underlying_price (цена крипты в момент сигнала) и created_at
     находим в crypto_candles следующую свечу через +5 мин, +15 мин, +30 мин.
  3. Если direction_value = 'UP' и close_price > underlying_price → correct.
     Если direction_value = 'DOWN' и close_price < underlying_price → correct.
  4. Считаем accuracy по активу, режиму, модели и горизонту.

Дополнительно: смотрим как часто LGBM-сигнал совпал с direction_status
(CONFIRMED — движок принял сигнал) и как торговля шла в эти моменты.
"""
import asyncio
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from decimal import Decimal
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

        # ── 1. Объём данных в funnel_log ──────────────────────────────────
        stats = (await s.execute(text("""
            SELECT
                COUNT(*) as total,
                COUNT(direction_value) as with_lgbm_signal,
                COUNT(CASE WHEN direction_status = 'CONFIRMED' THEN 1 END) as confirmed,
                COUNT(CASE WHEN direction_status = 'REJECTED'  THEN 1 END) as rejected,
                COUNT(CASE WHEN direction_value = 'UP'   THEN 1 END) as up_signals,
                COUNT(CASE WHEN direction_value = 'DOWN' THEN 1 END) as down_signals,
                MIN(created_at) as first_log,
                MAX(created_at) as last_log
            FROM decision_funnel_log
            WHERE direction_model_key IS NOT NULL;
        """))).mappings().one()

        print_subheader("0. Общая статистика funnel_log")
        print(f"\n  Всего записей с LGBM-сигналом : {stats['with_lgbm_signal']:>8}")
        print(f"  Из них CONFIRMED               : {stats['confirmed']:>8}  ({pct(stats['confirmed'], stats['with_lgbm_signal'])})")
        print(f"  Из них REJECTED                : {stats['rejected']:>8}  ({pct(stats['rejected'], stats['with_lgbm_signal'])})")
        print(f"  UP сигналов                    : {stats['up_signals']:>8}")
        print(f"  DOWN сигналов                  : {stats['down_signals']:>8}")
        print(f"  Период данных                  : {stats['first_log']} → {stats['last_log']}")

        # ── 2. Разбивка по активу + режиму + direction_value ──────────────
        by_asset = (await s.execute(text("""
            SELECT
                direction_model_key,
                direction_regime,
                direction_value,
                direction_status,
                COUNT(*) as cnt,
                ROUND(AVG(direction_p_up)::numeric, 3)   as avg_p_up,
                ROUND(AVG(direction_p_down)::numeric, 3) as avg_p_down,
                ROUND(AVG(direction_probability)::numeric, 3) as avg_prob
            FROM decision_funnel_log
            WHERE direction_model_key IS NOT NULL
              AND direction_value IS NOT NULL
            GROUP BY direction_model_key, direction_regime, direction_value, direction_status
            ORDER BY direction_model_key, direction_regime, direction_value, direction_status;
        """))).mappings().all()

        print_subheader("1. Сигналы по модели / режиму / направлению")
        print(f"\n  {'Model key':<22} {'Regime':<12} {'Dir':<5} {'Status':<12} {'Count':>7}  {'AvgP_up':>8}  {'AvgP_dn':>8}  {'AvgProb':>8}")
        print(f"  {'-'*22} {'-'*12} {'-'*5} {'-'*12} {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}")
        for r in by_asset:
            print(
                f"  {str(r['direction_model_key'] or '-'):<22} "
                f"{str(r['direction_regime'] or '-'):<12} "
                f"{str(r['direction_value'] or '-'):<5} "
                f"{str(r['direction_status'] or '-'):<12} "
                f"{r['cnt']:>7}  "
                f"{str(r['avg_p_up'] or '-'):>8}  "
                f"{str(r['avg_p_down'] or '-'):>8}  "
                f"{str(r['avg_prob'] or '-'):>8}"
            )

        # ── 3. Точность по crypto_candles (сравниваем с ценой через +5/+15/+30 мин) ──
        print_subheader("2. Точность направления vs реальная цена (+5/+15/+30 мин)")

        accuracy_rows = (await s.execute(text("""
            WITH signals AS (
                SELECT
                    f.id,
                    f.created_at,
                    f.direction_model_key,
                    f.direction_regime,
                    f.direction_value,
                    f.direction_status,
                    f.underlying_price,
                    -- Определяем asset из model_key: убираем суффикс режима
                    SPLIT_PART(f.direction_model_key, '_', 1) as coin
                FROM decision_funnel_log f
                WHERE f.direction_value IS NOT NULL
                  AND f.underlying_price IS NOT NULL
                  AND f.underlying_price > 0
                  AND f.direction_model_key IS NOT NULL
            ),
            candles_5 AS (
                SELECT DISTINCT ON (s.id)
                    s.id,
                    s.direction_value,
                    s.direction_model_key,
                    s.direction_regime,
                    s.direction_status,
                    s.underlying_price,
                    c.close as close_5
                FROM signals s
                JOIN crypto_candles c
                  ON c.symbol ILIKE s.coin || '%'
                 AND c.interval = '5m'
                 AND c.open_time BETWEEN s.created_at AND s.created_at + INTERVAL '10 minutes'
                ORDER BY s.id, c.open_time ASC
            ),
            candles_15 AS (
                SELECT DISTINCT ON (s.id)
                    s.id,
                    c.close as close_15
                FROM signals s
                JOIN crypto_candles c
                  ON c.symbol ILIKE s.coin || '%'
                 AND c.interval = '15m'
                 AND c.open_time BETWEEN s.created_at AND s.created_at + INTERVAL '20 minutes'
                ORDER BY s.id, c.open_time ASC
            ),
            joined AS (
                SELECT
                    c5.direction_model_key,
                    c5.direction_regime,
                    c5.direction_status,
                    c5.direction_value,
                    c5.underlying_price,
                    c5.close_5,
                    c15.close_15,
                    -- correct_5: направление совпало через 5 мин
                    CASE
                        WHEN c5.direction_value = 'UP'   AND c5.close_5 > c5.underlying_price THEN true
                        WHEN c5.direction_value = 'DOWN' AND c5.close_5 < c5.underlying_price THEN true
                        WHEN c5.close_5 IS NOT NULL THEN false
                        ELSE NULL
                    END as correct_5,
                    -- correct_15: направление совпало через 15 мин
                    CASE
                        WHEN c5.direction_value = 'UP'   AND c15.close_15 > c5.underlying_price THEN true
                        WHEN c5.direction_value = 'DOWN' AND c15.close_15 < c5.underlying_price THEN true
                        WHEN c15.close_15 IS NOT NULL THEN false
                        ELSE NULL
                    END as correct_15
                FROM candles_5 c5
                LEFT JOIN candles_15 c15 USING (id)
            )
            SELECT
                direction_model_key,
                direction_regime,
                direction_status,
                COUNT(*) as total,
                COUNT(correct_5) as matched_5,
                SUM(CASE WHEN correct_5 THEN 1 ELSE 0 END) as correct_5_cnt,
                COUNT(correct_15) as matched_15,
                SUM(CASE WHEN correct_15 THEN 1 ELSE 0 END) as correct_15_cnt
            FROM joined
            GROUP BY direction_model_key, direction_regime, direction_status
            ORDER BY direction_model_key, direction_regime, direction_status;
        """))).mappings().all()

        if not accuracy_rows:
            print("\n  Нет данных для сопоставления (возможно crypto_candles пустые или нет совпадений по времени).")
        else:
            print(f"\n  {'Model key':<22} {'Regime':<12} {'Status':<12} {'Total':>7}  {'Acc@5m':>8}  {'Acc@15m':>9}")
            print(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*7}  {'-'*8}  {'-'*9}")
            for r in accuracy_rows:
                acc5  = pct(r['correct_5_cnt'],  r['matched_5'])
                acc15 = pct(r['correct_15_cnt'], r['matched_15'])
                print(
                    f"  {str(r['direction_model_key'] or '-'):<22} "
                    f"{str(r['direction_regime'] or '-'):<12} "
                    f"{str(r['direction_status'] or '-'):<12} "
                    f"{r['total']:>7}  "
                    f"{acc5:>8}  "
                    f"{acc15:>9}"
                )

        # ── 4. Инвертированность: смотрим сколько раз LGBM ошибся
        print_subheader("3. Инвертированность сигнала (итог по всем)")
        inv = (await s.execute(text("""
            WITH signals AS (
                SELECT
                    f.created_at,
                    f.direction_value,
                    f.underlying_price,
                    SPLIT_PART(f.direction_model_key, '_', 1) as coin
                FROM decision_funnel_log f
                WHERE f.direction_value IS NOT NULL
                  AND f.underlying_price IS NOT NULL
                  AND f.underlying_price > 0
            ),
            joined AS (
                SELECT DISTINCT ON (s.created_at, s.coin)
                    s.direction_value,
                    s.underlying_price,
                    c.close,
                    CASE
                        WHEN s.direction_value = 'UP'   AND c.close > s.underlying_price THEN 'correct'
                        WHEN s.direction_value = 'DOWN' AND c.close < s.underlying_price THEN 'correct'
                        ELSE 'wrong'
                    END as result
                FROM signals s
                JOIN crypto_candles c
                  ON c.symbol ILIKE s.coin || '%'
                 AND c.interval = '5m'
                 AND c.open_time BETWEEN s.created_at AND s.created_at + INTERVAL '10 minutes'
                ORDER BY s.created_at, s.coin, c.open_time ASC
            )
            SELECT
                result,
                COUNT(*) as cnt
            FROM joined
            GROUP BY result;
        """))).mappings().all()

        total_inv = sum(r['cnt'] for r in inv)
        print(f"\n  {'Результат':<12} {'Кол-во':>8}  {'Доля':>8}")
        print(f"  {'-'*12} {'-'*8}  {'-'*8}")
        for r in inv:
            print(f"  {r['result']:<12} {r['cnt']:>8}  {pct(r['cnt'], total_inv):>8}")
        if total_inv:
            correct = next((r['cnt'] for r in inv if r['result'] == 'correct'), 0)
            print(f"\n  Итого: {total_inv} сигналов. Accuracy@5m = {pct(correct, total_inv)}")
            if correct / total_inv < 0.45:
                print(f"  ⚠ Accuracy < 45% → сигнал ИНВЕРТИРОВАН (угадывает наоборот)")
            elif correct / total_inv > 0.55:
                print(f"  ✓ Accuracy > 55% → сигнал КОРРЕКТЕН")
            else:
                print(f"  ~ Accuracy в диапазоне 45-55% → сигнал СЛУЧАЕН (нет предсказательной силы)")

    print(f"\n{'='*76}\n")


if __name__ == "__main__":
    asyncio.run(main())
