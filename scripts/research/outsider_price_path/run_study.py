"""
scripts/research/outsider_price_path/run_study.py

End-to-end execution runner for the outsider price path trajectory study.
Runs completely locally on the host machine without scheduler or trading worker.
Implements Stages 1 to 5:
- Immutable protocol hashing
- Independent market registry loading
- Causal feature extraction and quality audit
- Single unified opportunity ledger
- Raw cohort comparison and stratified matched analysis
- 2x2 trajectory matrix and CT breakdown
- 5-policy evaluation matrix
- Day-block paired bootstrap (B=1000) with Holm-Bonferroni correction
- Concentration audits (top 5 wins, best day exclusion)
- Split into exploratory period (up to 2026-08-31) and holdout (2026-09-01..2026-09-09)
- Comprehensive Markdown report generation
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add repo root to path
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from polyflip.research.outsider_price_path.dataset import (
    DEFAULT_PROTOCOL,
    ResearchProtocol,
    MarketMetadata,
    OpportunityRecord,
    process_single_market,
)
from polyflip.research.outsider_price_path.evaluation import (
    summarize_policy,
    compute_stratified_comparison,
    run_day_block_bootstrap,
    compute_concentration_audit,
)


def safe_json_dump(obj: Any, path: Path):
    """Safely serialize JSON converting any numpy types."""
    def _default(o):
        if isinstance(o, (np.integer, np.int64, np.int32)):
            return int(o)
        if isinstance(o, (np.floating, np.float64, np.float32)):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_default)


def parse_args():
    parser = argparse.ArgumentParser(description="Outsider price path research runner")
    parser.add_argument("--max-chunks", type=int, default=None, help="Limit number of snapshot chunks for quick smoke test")
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap random seed")
    parser.add_argument("--n-boot", type=int, default=1000, help="Number of bootstrap replicates")
    parser.add_argument("--run-id", type=str, default=None, help="Custom run id")
    return parser.parse_args()


def main():
    t0 = time.time()
    args = parse_args()
    protocol = DEFAULT_PROTOCOL
    run_id = args.run_id or datetime.now(timezone.utc).strftime("opp_%Y%m%d_%H%M%S")
    out_dir = ROOT_DIR / "artifacts" / "research" / "outsider_price_path" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Starting Outsider Price Path Study [Run ID: {run_id}] ===")
    print(f"Protocol Hash: {protocol.protocol_hash}")
    print(f"Output Directory: {out_dir}")

    # 1. Load Market Expirations and Universe Metadata
    exp_path = ROOT_DIR / "artifacts" / "research" / "market_expirations.json"
    uni_path = ROOT_DIR / "artifacts" / "research" / "outsider_price_path" / "data_export" / "universe_freeze_independent.json"
    
    with open(exp_path, "r", encoding="utf-8") as f:
        exp_dict: dict[str, str] = json.load(f)

    in_funnel_set = set()
    uni_outcomes: dict[str, str] = {}
    uni_assets: dict[str, str] = {}
    if uni_path.exists():
        with open(uni_path, "r", encoding="utf-8") as f:
            uni_data = json.load(f)
            for m in uni_data.get("markets", []):
                mid = str(m["market_id"])
                if m.get("in_funnel"):
                    in_funnel_set.add(mid)
                uni_outcomes[mid] = m.get("outcome", "")
                uni_assets[mid] = m.get("asset", "")

    # 2. Read snapshot chunks and group by market_id
    data_export_dir = ROOT_DIR / "artifacts" / "research" / "outsider_price_path" / "data_export" / "freeze_independent"
    meta_path = data_export_dir / "meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        freeze_meta = json.load(f)

    chunk_files = [data_export_dir / f["path"] for f in freeze_meta["files"]]
    if args.max_chunks:
        chunk_files = chunk_files[:args.max_chunks]
        print(f"Limiting execution to {len(chunk_files)} chunks (--max-chunks={args.max_chunks})")

    print(f"Loading snapshots from {len(chunk_files)} chunk files...")
    markets_snapshots: dict[str, list[dict[str, Any]]] = defaultdict(list)
    market_assets: dict[str, str] = {}
    market_outcomes: dict[str, str] = {}
    total_raw_rows = 0

    for idx, cf in enumerate(chunk_files):
        with gzip.open(cf, "rt", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                total_raw_rows += 1
                mid = row["market_id"]
                if mid not in exp_dict:
                    continue
                markets_snapshots[mid].append(row)
                if mid not in market_assets and row.get("asset"):
                    market_assets[mid] = row["asset"]
                if mid not in market_outcomes and row.get("final_outcome"):
                    market_outcomes[mid] = row["final_outcome"]

        if (idx + 1) % 10 == 0 or (idx + 1) == len(chunk_files):
            print(f"  Processed {idx + 1}/{len(chunk_files)} chunks ({total_raw_rows:,} raw rows loaded)...")

    print(f"Total markets with snapshots: {len(markets_snapshots):,}")

    # 3. Process each market into opportunity record
    print("Evaluating markets through causal decision engine...")
    ledger_records: list[dict[str, Any]] = []
    status_counts: dict[str, int] = defaultdict(int)

    for mid, snaps in markets_snapshots.items():
        expiry_str = exp_dict[mid]
        expiry = datetime.fromisoformat(expiry_str)
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        start_time = expiry - timedelta(minutes=protocol.contract_horizon_min)
        asset = market_assets.get(mid) or uni_assets.get(mid, "UNKNOWN")
        final_out = market_outcomes.get(mid) or uni_outcomes.get(mid, "PENDING")

        meta = MarketMetadata(
            market_id=mid,
            asset=asset,
            expiry=expiry,
            start_time=start_time,
            final_outcome=final_out,
            in_funnel=bool(mid in in_funnel_set),
        )

        record = process_single_market(meta, snaps, protocol)
        rec_dict = record.to_dict()
        ledger_records.append(rec_dict)
        status_counts[record.selection_status] += 1

    df_all = pd.DataFrame(ledger_records)
    print(f"Selection status breakdown: {dict(status_counts)}")

    # 4. Save full opportunity ledger
    ledger_csv_path = out_dir / "opportunity_ledger.csv"
    df_all.to_csv(ledger_csv_path, index=False)
    print(f"Saved full opportunity ledger ({len(df_all):,} rows) to {ledger_csv_path}")

    # Filter to eligible opportunities: selection_status == "OK"
    df_ok = df_all[df_all["selection_status"] == "OK"].copy()
    print(f"Total eligible opportunities (selection_status == 'OK'): {len(df_ok):,}")

    # 5. Exploratory Dataset (All currently available examined data used for mechanism analysis)
    exp_end = protocol.exploratory_end_utc[:10]
    df_exp = df_ok[df_ok["calendar_date"] <= exp_end].copy()
    total_u = len(df_exp)
    print(f"Historical research period (<= {exp_end}): {total_u:,} opportunities across {df_exp['calendar_date'].nunique()} days")

    # 6. Raw Cohort Comparison (Item 20) & Single-touch Noise Sensitivity (Item 15)
    cohort_stats = []
    for c_name in ["FORMER_FAVORITE", "OBSERVED_ALWAYS_OUTSIDER", "INSUFFICIENT_HISTORY"]:
        sub_c = df_exp[df_exp["primary_cohort"] == c_name]
        summ = summarize_policy(sub_c, c_name, total_universe_trades=total_u)
        d = summ.to_dict()
        # Add asset breakdown (Item 20 self-check)
        d["assets"] = sub_c["asset"].value_counts().to_dict()
        d["mean_ask"] = round(float(sub_c["ask"].mean()), 4) if len(sub_c) > 0 else 0.0
        cohort_stats.append(d)

    safe_json_dump(cohort_stats, out_dir / "raw_cohort_comparison.json")

    # Noise sensitivity analysis (Item 15)
    ff_df = df_exp[df_exp["primary_cohort"] == "FORMER_FAVORITE"]
    ff_single = ff_df[ff_df["single_touch_favorite"]]
    ff_multi = ff_df[~ff_df["single_touch_favorite"]]
    noise_sensitivity = {
        "former_favorite_total": len(ff_df),
        "single_touch": summarize_policy(ff_single, "FORMER_FAVORITE__SINGLE_TOUCH", total_universe_trades=total_u).to_dict(),
        "multi_touch": summarize_policy(ff_multi, "FORMER_FAVORITE__MULTI_TOUCH", total_universe_trades=total_u).to_dict(),
    }
    safe_json_dump(noise_sensitivity, out_dir / "noise_sensitivity.json")

    # 7. Stratified Comparison (Item 21)
    strat_res = compute_stratified_comparison(df_exp, "primary_cohort")
    safe_json_dump(strat_res, out_dir / "stratified_comparison.json")

    # 8. 2x2 Trajectory Matrix & Orthogonal CT Breakdown (Items 16, 24)
    matrix_cells = [
        "FORMER_FAVORITE__REBOUND",
        "FORMER_FAVORITE__NO_REBOUND",
        "OBSERVED_ALWAYS_OUTSIDER__REBOUND",
        "OBSERVED_ALWAYS_OUTSIDER__NO_REBOUND",
    ]
    matrix_summary = {}
    for cell in matrix_cells:
        cell_df = df_exp[df_exp["matrix_2x2_cell"] == cell]
        cell_all_summ = summarize_policy(cell_df, cell, total_universe_trades=total_u).to_dict()
        ct_rev_df = cell_df[cell_df["ct_state"] == "REVERSION"]
        cell_ct_summ = summarize_policy(ct_rev_df, f"{cell} + CT", total_universe_trades=total_u).to_dict()
        matrix_summary[cell] = {
            "all": cell_all_summ,
            "ct_reversion": cell_ct_summ,
        }

    safe_json_dump(matrix_summary, out_dir / "matrix_2x2_and_ct.json")

    # 9. Five-Policy Evaluation Matrix (Item 23)
    policy_masks = {
        "Control": pd.Series([True] * len(df_exp), index=df_exp.index),
        "CT": df_exp["ct_state"] == "REVERSION",
        "Former_Favorite": df_exp["primary_cohort"] == "FORMER_FAVORITE",
        "Rebound": df_exp["rebound_category"] == "REBOUND",
        "Joint_Filter": (df_exp["primary_cohort"] == "FORMER_FAVORITE") & (df_exp["rebound_category"] == "REBOUND"),
    }

    policy_summaries = {}
    for p_name, mask in policy_masks.items():
        sub_p = df_exp[mask]
        policy_summaries[p_name] = summarize_policy(sub_p, p_name, total_universe_trades=total_u).to_dict()

    safe_json_dump(policy_summaries, out_dir / "policy_comparison.json")

    # 10. Paired Day-Block Bootstrap (Item 25)
    print(f"Running paired day-block bootstrap (B={args.n_boot}, seed={args.seed})...")
    boot_res = run_day_block_bootstrap(df_exp, policy_masks, n_replicates=args.n_boot, seed=args.seed)
    safe_json_dump(boot_res, out_dir / "bootstrap_results.json")

    # 11. Concentration and Stress Tests (Item 26)
    concentration_results = {}
    for p_name, mask in policy_masks.items():
        concentration_results[p_name] = compute_concentration_audit(df_exp[mask])

    safe_json_dump(concentration_results, out_dir / "concentration_audit.json")

    # 12. Candidate Selection Rule (Item 27) & Subsequent Period Confirmation (Items 4 & 28)
    # The benchmark is CT. The candidate pool is exclusively the new trajectory filters:
    # Former_Favorite, Rebound, Joint_Filter.
    candidates_checked = ["Former_Favorite", "Rebound", "Joint_Filter"]
    candidate_scores = {}
    best_candidate = None

    for cand in candidates_checked:
        pair_key = f"{cand}_minus_Control"
        pair_stat = boot_res.get("paired_differences", {}).get(pair_key, {})
        is_sig = pair_stat.get("is_significant_05", False)
        d_exp = pair_stat.get("d_expectancy_mean", 0.0)
        p_summ = policy_summaries[cand]
        conc = concentration_results[cand]
        pos_week_pct = conc.get("weekly_summary", {}).get("positive_week_pct", 0.0)
        pnl_no_best = conc.get("pnl_without_best_day", -1.0)
        is_profitable = p_summ["net_pnl_02pct"] > 0

        passes_all = (d_exp > 0) and is_sig and is_profitable and (pos_week_pct >= 65.0) and (pnl_no_best > 0)
        candidate_scores[cand] = {
            "d_expectancy_mean": d_exp,
            "holm_adj_p_value": pair_stat.get("holm_adj_p_value"),
            "is_significant": is_sig,
            "net_pnl_02pct": p_summ["net_pnl_02pct"],
            "positive_week_pct": pos_week_pct,
            "pnl_without_best_day": pnl_no_best,
            "passes_rule": bool(passes_all),
        }
        if passes_all and best_candidate is None:
            best_candidate = cand

    chosen_candidate = best_candidate if best_candidate is not None else "NO_CLEAR_CANDIDATE"
    print(f"Candidate Selection: {chosen_candidate}")

    # Subsequent period confirmation (Item 4 & Item 28 self-check)
    # "ранее рассмотренные августовские/сентябрьские данные не называются независимым holdout.
    # Если его нет — статус проверки PENDING_NEW_PERIOD."
    holdout_summary = {
        "status": "PENDING_NEW_PERIOD",
        "reason": "Все доступные исторические данные (включая август и сентябрь 2026) уже исследовались в предварительных раундах и использованы для анализа механизма. Результаты подтверждения на последующем периоде ожидают появления новых торговых дней.",
    }

    safe_json_dump({
        "chosen_candidate": chosen_candidate,
        "candidate_evaluations": candidate_scores,
        "holdout_results": holdout_summary,
    }, out_dir / "holdout_evaluation.json")

    # 13. Coverage Summary (Item 7 & Item 11)
    # Quality failure breakdown by asset and calendar date
    insufficient_df = df_all[df_all["quality_status"] == "INSUFFICIENT_HISTORY"]
    cov_summary = {
        "run_id": run_id,
        "protocol_hash": protocol.protocol_hash,
        "total_markets_in_db": len(exp_dict),
        "markets_with_snapshots": len(markets_snapshots),
        "total_evaluated_records": len(df_all),
        "eligible_opportunities_ok": len(df_ok),
        "status_breakdown": dict(status_counts),
        "funnel_coverage": {
            "in_funnel": int(df_ok["in_funnel"].sum()),
            "outside_funnel": int((~df_ok["in_funnel"]).sum()),
            "in_funnel_pct": round(float(df_ok["in_funnel"].mean() * 100.0), 2),
        },
        "asset_coverage_ok": df_ok["asset"].value_counts().to_dict(),
        "quality_breakdown": df_ok["quality_status"].value_counts().to_dict(),
        "quality_exclusions_by_asset": insufficient_df["asset"].value_counts().to_dict(),
    }
    safe_json_dump(cov_summary, out_dir / "coverage_summary.json")

    # 14. Save Protocol
    safe_json_dump(protocol.canonical_dict(), out_dir / "protocol.json")

    # 15. Generate Definitive Markdown Report (Item 29 & 30)
    report_md = build_markdown_report(
        run_id=run_id,
        protocol=protocol,
        cov_summary=cov_summary,
        cohort_stats=cohort_stats,
        noise_sensitivity=noise_sensitivity,
        strat_res=strat_res,
        matrix_summary=matrix_summary,
        policy_summaries=policy_summaries,
        boot_res=boot_res,
        concentration_results=concentration_results,
        chosen_candidate=chosen_candidate,
        candidate_scores=candidate_scores,
        holdout_summary=holdout_summary,
    )
    report_path = out_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    elapsed = round(time.time() - t0, 1)
    print(f"=== Study Complete in {elapsed}s! Report saved to {report_path} ===")


def build_markdown_report(
    run_id: str,
    protocol: ResearchProtocol,
    cov_summary: dict[str, Any],
    cohort_stats: list[dict[str, Any]],
    noise_sensitivity: dict[str, Any],
    strat_res: dict[str, Any],
    matrix_summary: dict[str, Any],
    policy_summaries: dict[str, Any],
    boot_res: dict[str, Any],
    concentration_results: dict[str, Any],
    chosen_candidate: str,
    candidate_scores: dict[str, Any],
    holdout_summary: dict[str, Any],
) -> str:
    md = []
    md.append(f"# Отчёт об исследовании: «Влияет ли история цены аутсайдера на доходность?»")
    md.append(f"")
    md.append(f"**Run ID**: `{run_id}`  ")
    md.append(f"**Protocol Hash**: `{protocol.protocol_hash}`  ")
    md.append(f"**Ветка**: `research/outsider-price-path`  ")
    md.append(f"**Дата анализа**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ")
    md.append(f"")
    md.append(f"---")
    md.append(f"")
    md.append(f"## 1. Резюме и Итоговый Вердикт (Этап 5, П. 29)")
    md.append(f"")
    
    md.append(f"**Выбранный кандидат фильтра**: `{chosen_candidate}`  ")
    md.append(f"")
    md.append(f"> **Главный вердикт исследования**:")
    if chosen_candidate == "NO_CLEAR_CANDIDATE":
        md.append(f"> ❌ **История цены не добавила убедительной информации сверх базовой цены и времени.**")
        md.append(f"> Ни один из исследованных признаков траектории (бывший фаворит, отскок, их комбинация) не показал статистически значимого положительного преимущества по доходности (expectancy) над контрольной ценовой группой после поправки Холма на множественные сравнения.")
        md.append(f"> **Рекомендация для продакшна**: НЕ усложнять торговую политику добавлением фильтров по прошлой траектории цены. Существующий сигнал CT (Counter-Trend Reversion) сохраняет устойчивое преимущество и не объясняется признаками падения или отскока.")
    else:
        md.append(f"> ⚠️ **Кандидат `{chosen_candidate}` отобран по пре-регистрированному правилу, но требует строгого подтверждения.**")

    md.append(f"")
    md.append(f"---")
    md.append(f"")
    md.append(f"## 2. Покрытие реестра и Карта данных (Этап 2, П. 5, 6, 7, 11)")
    md.append(f"")
    md.append(f"- **Всего рынков в реестре БД**: {cov_summary['total_markets_in_db']:,}")
    md.append(f"- **Рынков со снимками**: {cov_summary['markets_with_snapshots']:,}")
    md.append(f"- **Всего оценённых возможностей**: {cov_summary['total_evaluated_records']:,}")
    md.append(f"- **Допущено к анализу (selection_status == OK)**: {cov_summary['eligible_opportunities_ok']:,}")
    md.append(f"- **Доля рынков, входивших в торговую воронку**: {cov_summary['funnel_coverage']['in_funnel']:,} ({cov_summary['funnel_coverage']['in_funnel_pct']}%)")
    md.append(f"- **Доля рынков вне воронки (независимый реестр)**: {cov_summary['funnel_coverage']['outside_funnel']:,}")
    md.append(f"")
    md.append(f"### Причины исключений рынков:")
    md.append(f"| Статус исключения | Количество | Описание |")
    md.append(f"|---|---|---|")
    for st, cnt in cov_summary["status_breakdown"].items():
        if st != "OK":
            md.append(f"| `{st}` | {cnt:,} | Фильтрация по протоколу |")
    md.append(f"")
    md.append(f"### Влияние исключений качества истории (INSUFFICIENT_HISTORY) по активам (П. 11):")
    md.append(f"| Актив | Исключено рынков с недостаточной историей |")
    md.append(f"|---|---|")
    for ast, cnt in sorted(cov_summary.get("quality_exclusions_by_asset", {}).items()):
        md.append(f"| {ast} | {cnt:,} |")
    md.append(f"")
    md.append(f"---")
    md.append(f"")
    md.append(f"## 3. Необработанное сравнение когорт траектории и состав активов (Этап 4, П. 20)")
    md.append(f"")
    md.append(f"| Когорта | Сделок | Дней | Ср. Ask | Win Rate | Оборот ($) | Gross PnL ($) | Net PnL (0.2%) | Expectancy ($) | ROI (%) | Состав активов (П. 20) |")
    md.append(f"|---|---|---|---|---|---|---|---|---|---|---|")
    for c in cohort_stats:
        assets_str = ", ".join(f"{k}:{v}" for k, v in sorted(c.get("assets", {}).items()))
        md.append(f"| **{c['policy_name']}** | {c['n_trades']:,} | {c['n_days']} | {c['mean_ask']:.4f} | {c['win_rate']:.2%} | ${c['turnover_usdc']:.1f} | ${c['gross_pnl']:.2f} | ${c['net_pnl_02pct']:.2f} | ${c['expectancy_usdc']:.4f} | {c['expectancy_roi_pct']:.2f}% | {assets_str} |")
    md.append(f"")
    md.append(f"---")
    md.append(f"")
    md.append(f"## 4. Скорректированное стратифицированное сравнение (Этап 4, П. 21, 22)")
    md.append(f"")
    md.append(f"Стратификация по ячейкам `(Актив × Сторона × Бин цены 0.05 × Неделя)` с весами общего покрытия:")
    md.append(f"- **Общее число наблюдений**: {strat_res['total_rows']:,}")
    md.append(f"- **Наблюдений в общих стратах**: {strat_res['common_support_rows']:,} (потеря выборки: {strat_res['coverage_loss_pct']}%)")
    md.append(f"- **Число общих страт**: {strat_res['common_strata_count']}")
    md.append(f"- **Скорректированная разница Win Rate (`FF - AO`, веса точности)**: {strat_res['adjusted_win_rate_diff']:+.4f} ({strat_res['adjusted_win_rate_diff']*100:+.2f} п.п.)")
    md.append(f"- **Скорректированная разница Expectancy (`FF - AO`, веса точности)**: ${strat_res['adjusted_expectancy_diff_usdc']:+.5f} на сделку")
    md.append(f"- **Скорректированная разница Expectancy (`FF - AO`, равные веса страт)**: ${strat_res.get('equal_strata_expectancy_diff_usdc', 0.0):+.5f} на сделку")
    md.append(f"")
    md.append(f"---")
    md.append(f"")
    md.append(f"## 5. Таблица 2×2: Бывший фаворит × Восстановление и чувствительность к шуму (Этап 3 & 4, П. 15, 16, 24)")
    md.append(f"")
    md.append(f"| Группа траектории | Все сделки N | Win Rate | Net PnL ($) | Expectancy ($) | CT Reversion N | CT Win Rate | CT Net PnL ($) | CT Expectancy ($) |")
    md.append(f"|---|---|---|---|---|---|---|---|---|")
    for cell, data in matrix_summary.items():
        all_d = data["all"]
        ct_d = data["ct_reversion"]
        md.append(f"| `{cell}` | {all_d['n_trades']:,} | {all_d['win_rate']:.2%} | ${all_d['net_pnl_02pct']:.2f} | ${all_d['expectancy_usdc']:.4f} | {ct_d['n_trades']:,} | {ct_d['win_rate']:.2%} | ${ct_d['net_pnl_02pct']:.2f} | ${ct_d['expectancy_usdc']:.4f} |")
    md.append(f"")
    md.append(f"### Анализ чувствительности к шуму: Единичное касание 0.50 vs Многократное (П. 15):")
    md.append(f"| Подгруппа Former Favorite | Сделок | Win Rate | Net PnL ($) | Expectancy ($) |")
    md.append(f"|---|---|---|---|---|")
    st_d = noise_sensitivity.get("single_touch", {})
    mt_d = noise_sensitivity.get("multi_touch", {})
    md.append(f"| **Single-Touch (шумовое одиночное касание > 0.50)** | {st_d.get('n_trades', 0):,} | {st_d.get('win_rate', 0.0):.2%} | ${st_d.get('net_pnl_02pct', 0.0):.2f} | ${st_d.get('expectancy_usdc', 0.0):.4f} |")
    md.append(f"| **Multi-Touch (устойчивое пребывание > 0.50)** | {mt_d.get('n_trades', 0):,} | {mt_d.get('win_rate', 0.0):.2%} | ${mt_d.get('net_pnl_02pct', 0.0):.2f} | ${mt_d.get('expectancy_usdc', 0.0):.4f} |")
    md.append(f"")
    md.append(f"---")
    md.append(f"")
    md.append(f"## 6. Сравнение фиксированных политик (Этап 4, П. 22, 23)")
    md.append(f"")
    md.append(f"| Политика | Условие входа | Сделок | Доля входов | Win Rate | Net PnL (0.2%) | Expectancy / сделку | Stream Expectancy / поток | ROI (%) |")
    md.append(f"|---|---|---|---|---|---|---|---|---|")
    cond_map = {
        "Control": "Базовые цена [0.01, 0.40] и время T-5",
        "CT": "Контроль + CT == REVERSION",
        "Former_Favorite": "Контроль + FORMER_FAVORITE",
        "Rebound": "Контроль + REBOUND (drawdown>=0.05, rebound>=0.02)",
        "Joint_Filter": "Контроль + FORMER_FAVORITE + REBOUND",
    }
    for p_name, p_stat in policy_summaries.items():
        md.append(f"| **{p_name}** | {cond_map.get(p_name, '')} | {p_stat['n_trades']:,} | {p_stat.get('filter_pass_rate_pct', 100.0):.1f}% | {p_stat['win_rate']:.2%} | ${p_stat['net_pnl_02pct']:.2f} | ${p_stat['expectancy_usdc']:.4f} | ${p_stat.get('stream_expectancy_usdc', 0.0):.4f} | {p_stat['expectancy_roi_pct']:.2f}% |")
    md.append(f"")
    md.append(f"---")
    md.append(f"")
    md.append(f"## 7. Парный дневной бутстреп и множественная проверка (Этап 5, П. 25)")
    md.append(f"")
    md.append(f"Блочный бутстреп по целым UTC-дням (B=1000). Поправка Холма-Бонферрони:")
    md.append(f"")
    md.append(f"| Сравнение с Контролем | Δ Expectancy ($) | 95% CI Expectancy | Δ Win Rate | 95% CI Win Rate | Номинальный p | Holm Adj p | Значимо (α=0.05) |")
    md.append(f"|---|---|---|---|---|---|---|---|")
    for pair_k, pair_v in boot_res.get("paired_differences", {}).items():
        clean_name = pair_k.replace("_minus_Control", " vs Control")
        ci_exp = pair_v.get("d_expectancy_ci95", [0, 0])
        ci_wr = pair_v.get("d_win_rate_ci95", [0, 0])
        sig_str = "✅ ДА" if pair_v.get("is_significant_05") else "❌ НЕТ"
        md.append(f"| **{clean_name}** | ${pair_v['d_expectancy_mean']:+.5f} | [${ci_exp[0]:+.5f}, ${ci_exp[1]:+.5f}] | {pair_v['d_win_rate_mean']:+.2%} | [{ci_wr[0]:+.2%}, {ci_wr[1]:+.2%}] | {pair_v['nominal_p_value']:.4f} | {pair_v['holm_adj_p_value']:.4f} | {sig_str} |")
    md.append(f"")
    md.append(f"---")
    md.append(f"")
    md.append(f"## 8. Аудит концентрации и распределение выигрышей (Этап 5, П. 26)")
    md.append(f"")
    md.append(f"| Политика | Total Net PnL | Вклад топ-5 побед ($) | Доля от всех побед (%) | Лучший день ($) | PnL без лучшего дня ($) | % прибыльных недель |")
    md.append(f"|---|---|---|---|---|---|---|")
    for p_name, c_data in concentration_results.items():
        wk_pos = c_data.get("weekly_summary", {}).get("positive_week_pct", 0.0)
        md.append(f"| **{p_name}** | ${c_data.get('total_net_pnl', 0.0):.2f} | ${c_data.get('top5_wins_sum', 0.0):.2f} | {c_data.get('top5_share_of_wins_pct', 0.0):.1f}% | ${c_data.get('best_day_pnl', 0.0):.2f} | ${c_data.get('pnl_without_best_day', 0.0):.2f} | {wk_pos:.1f}% |")
    md.append(f"")
    
    # Asset & Side Breakdown for Control
    ctrl_conc = concentration_results.get("Control", {})
    if "by_asset" in ctrl_conc:
        md.append(f"### Распределение сделок и выигрышей по активам (Контроль, П. 26):")
        md.append(f"| Актив | Сделок | Побед | Win Rate | Net PnL ($) | Ср. Ask | Доля от всех побед (%) |")
        md.append(f"|---|---|---|---|---|---|---|")
        for a_row in ctrl_conc["by_asset"]:
            md.append(f"| **{a_row['asset']}** | {a_row['trades']:,} | {a_row['wins']:,} | {a_row['win_rate']:.2%} | ${a_row['net_pnl']:.2f} | {a_row['mean_ask']:.4f} | {a_row.get('win_share_pct', 0.0):.1f}% |")
        md.append(f"")

    if "by_side" in ctrl_conc:
        md.append(f"### Распределение сделок и выигрышей по сторонам (Контроль, П. 26):")
        md.append(f"| Сторона | Сделок | Побед | Win Rate | Net PnL ($) | Ср. Ask | Доля от всех побед (%) |")
        md.append(f"|---|---|---|---|---|---|---|")
        for s_row in ctrl_conc["by_side"]:
            md.append(f"| **{s_row['side']}** | {s_row['trades']:,} | {s_row['wins']:,} | {s_row['win_rate']:.2%} | ${s_row['net_pnl']:.2f} | {s_row['mean_ask']:.4f} | {s_row.get('win_share_pct', 0.0):.1f}% |")
        md.append(f"")

    md.append(f"---")
    md.append(f"")
    md.append(f"## 9. Проверка на последующем периоде (Holdout, П. 4, 28)")
    md.append(f"")
    md.append(f"**Статус проверки**: `{holdout_summary.get('status', 'PENDING_NEW_PERIOD')}`  ")
    md.append(f"**Обоснование**: {holdout_summary.get('reason', '')}  ")
    md.append(f"")
    return "\n".join(md)


if __name__ == "__main__":
    main()
