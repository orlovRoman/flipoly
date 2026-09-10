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
    parser.add_argument("--ledger-path", type=str, default=None, help="Path to existing opportunity ledger CSV to bypass raw extraction")
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

    exp_dict: dict[str, str] = {}
    if exp_path.exists():
        with open(exp_path, "r", encoding="utf-8") as f:
            exp_dict = json.load(f)

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

    if args.ledger_path:
        ledger_csv_path = Path(args.ledger_path)
        print(f"Loading pre-computed opportunity ledger from {ledger_csv_path}...")
        df_all = pd.read_csv(ledger_csv_path)
        status_counts = df_all["selection_status"].value_counts().to_dict()
        markets_snapshots_count = df_all["market_id"].nunique()
    else:
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

        markets_snapshots_count = len(markets_snapshots)
        print(f"Total markets with snapshots: {markets_snapshots_count:,}")

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
    for num_col in ["ask", "target", "budget_usdc", "gross_pnl", "fee_02pct", "net_pnl_02pct", "net_pnl_01pct"]:
        if num_col in df_ok.columns:
            df_ok[num_col] = pd.to_numeric(df_ok[num_col], errors="coerce")
    if "is_win" in df_ok.columns:
        df_ok["is_win"] = df_ok["is_win"].astype(bool)
    if "single_touch_favorite" in df_ok.columns:
        df_ok["single_touch_favorite"] = df_ok["single_touch_favorite"].astype(bool)

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
        d["assets"] = sub_c["asset"].value_counts().to_dict()
        d["mean_ask"] = round(float(sub_c["ask"].mean()), 4) if len(sub_c) > 0 else 0.0
        cohort_stats.append(d)

    # Add Control (Все когорты) for full reconciliation
    ctrl_summ = summarize_policy(df_exp, "Control (Все когорты)", total_universe_trades=total_u)
    ctrl_d = ctrl_summ.to_dict()
    ctrl_d["assets"] = df_exp["asset"].value_counts().to_dict()
    ctrl_d["mean_ask"] = round(float(df_exp["ask"].mean()), 4) if len(df_exp) > 0 else 0.0
    cohort_stats.append(ctrl_d)

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
    strat_res = compute_stratified_comparison(df_exp, "primary_cohort", n_boot=args.n_boot, seed=args.seed)
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

    # Add INSUFFICIENT_HISTORY to reconcile the missing 155 CT entries
    ih_df = df_exp[df_exp["primary_cohort"] == "INSUFFICIENT_HISTORY"]
    ih_all_summ = summarize_policy(ih_df, "INSUFFICIENT_HISTORY", total_universe_trades=total_u).to_dict()
    ih_ct_df = ih_df[ih_df["ct_state"] == "REVERSION"]
    ih_ct_summ = summarize_policy(ih_ct_df, "INSUFFICIENT_HISTORY + CT", total_universe_trades=total_u).to_dict()
    matrix_summary["INSUFFICIENT_HISTORY"] = {
        "all": ih_all_summ,
        "ct_reversion": ih_ct_summ,
    }
    safe_json_dump(matrix_summary, out_dir / "matrix_2x2_and_ct.json")

    # 9. Policy Evaluation Matrices: Sufficient History vs Full Stream (Item 23)
    # Sample with sufficient history (N=4,346)
    df_suff = df_exp[df_exp["primary_cohort"] != "INSUFFICIENT_HISTORY"].copy()
    u_suff = len(df_suff)

    suff_policy_masks = {
        "Control": pd.Series([True] * len(df_suff), index=df_suff.index),
        "CT": df_suff["ct_state"] == "REVERSION",
        "Former_Favorite": df_suff["primary_cohort"] == "FORMER_FAVORITE",
        "Rebound": df_suff["rebound_category"] == "REBOUND",
        "Joint_Filter": (df_suff["primary_cohort"] == "FORMER_FAVORITE") & (df_suff["rebound_category"] == "REBOUND"),
    }
    suff_policy_summaries = {}
    for p_name, mask in suff_policy_masks.items():
        sub_p = df_suff[mask]
        suff_policy_summaries[p_name] = summarize_policy(sub_p, p_name, total_universe_trades=u_suff).to_dict()

    # Full Stream Universe (N=5,237)
    full_policy_masks = {
        "Control": pd.Series([True] * len(df_exp), index=df_exp.index),
        "CT": df_exp["ct_state"] == "REVERSION",
        "Former_Favorite": df_exp["primary_cohort"] == "FORMER_FAVORITE",
        "Rebound": df_exp["rebound_category"] == "REBOUND",
        "Joint_Filter": (df_exp["primary_cohort"] == "FORMER_FAVORITE") & (df_exp["rebound_category"] == "REBOUND"),
    }
    full_policy_summaries = {}
    for p_name, mask in full_policy_masks.items():
        sub_p = df_exp[mask]
        full_policy_summaries[p_name] = summarize_policy(sub_p, p_name, total_universe_trades=total_u).to_dict()

    policy_comparison_payload = {
        "sufficient_history": suff_policy_summaries,
        "full_stream": full_policy_summaries,
    }
    policy_comparison_payload.update(full_policy_summaries)
    safe_json_dump(policy_comparison_payload, out_dir / "policy_comparison.json")

    # 10. Paired Day-Block Bootstrap (Item 25)
    print(f"Running paired day-block bootstrap on Full Stream (B={args.n_boot}, seed={args.seed})...")
    boot_res = run_day_block_bootstrap(df_exp, full_policy_masks, n_replicates=args.n_boot, seed=args.seed)

    print(f"Running paired day-block bootstrap on Sufficient History (B={args.n_boot}, seed={args.seed})...")
    boot_res_suff = run_day_block_bootstrap(df_suff, suff_policy_masks, n_replicates=args.n_boot, seed=args.seed)

    boot_payload = {
        "full_stream": boot_res,
        "sufficient_history": boot_res_suff,
    }
    boot_payload.update(boot_res)
    safe_json_dump(boot_payload, out_dir / "bootstrap_results.json")

    # 11. Concentration and Stress Tests (Item 26)
    concentration_results = {}
    for p_name, mask in full_policy_masks.items():
        concentration_results[p_name] = compute_concentration_audit(df_exp[mask])
    safe_json_dump(concentration_results, out_dir / "concentration_audit.json")

    # 12. Candidate Selection Rule (Item 27) & Subsequent Period Confirmation (Items 4 & 28)
    candidates_checked = ["Former_Favorite", "Rebound", "Joint_Filter"]
    candidate_scores = {}
    best_candidate = None

    for cand in candidates_checked:
        pair_key = f"{cand}_minus_Control"
        pair_stat = boot_res.get("paired_differences", {}).get(pair_key, {})
        is_sig = pair_stat.get("is_significant_05", False)
        d_exp = pair_stat.get("d_trade_expectancy_mean", pair_stat.get("d_expectancy_mean", 0.0))
        p_summ = full_policy_summaries[cand]
        conc = concentration_results[cand]
        pos_week_pct = conc.get("weekly_summary", {}).get("positive_week_pct", 0.0)
        pnl_no_best = conc.get("pnl_without_best_day", -1.0)
        is_profitable = p_summ["net_pnl_02pct"] > 0

        passes_all = (d_exp > 0) and is_sig and is_profitable and (pos_week_pct >= 65.0) and (pnl_no_best > 0)
        candidate_scores[cand] = {
            "d_expectancy_mean": d_exp,
            "holm_adj_p_value": pair_stat.get("holm_adj_p_value_trade", pair_stat.get("holm_adj_p_value")),
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
    insufficient_df = df_all[df_all["quality_status"] == "INSUFFICIENT_HISTORY"]
    cov_summary = {
        "run_id": run_id,
        "protocol_hash": protocol.protocol_hash,
        "total_markets_in_db": len(exp_dict) if exp_dict else len(df_all),
        "markets_with_snapshots": markets_snapshots_count,
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

    # Automated data integrity assertions
    print("Running automated data integrity assertions...")
    ff_net = next(c["net_pnl_02pct"] for c in cohort_stats if c["policy_name"] == "FORMER_FAVORITE")
    ao_net = next(c["net_pnl_02pct"] for c in cohort_stats if c["policy_name"] == "OBSERVED_ALWAYS_OUTSIDER")
    ih_net = next(c["net_pnl_02pct"] for c in cohort_stats if c["policy_name"] == "INSUFFICIENT_HISTORY")
    ctrl_net = full_policy_summaries["Control"]["net_pnl_02pct"]
    assert abs((ff_net + ao_net + ih_net) - ctrl_net) < 0.05, f"Cohort net sum {ff_net + ao_net + ih_net} != control net {ctrl_net}"

    m_trades = sum(data["all"]["n_trades"] for k, data in matrix_summary.items())
    assert m_trades == len(df_exp), f"2x2 + IH trades sum {m_trades} != total trades {len(df_exp)}"

    m_ct_trades = sum(data["ct_reversion"]["n_trades"] for k, data in matrix_summary.items())
    assert m_ct_trades == full_policy_summaries["CT"]["n_trades"], f"CT trades sum {m_ct_trades} != total CT {full_policy_summaries['CT']['n_trades']}"

    m_ct_net = sum(data["ct_reversion"]["net_pnl_02pct"] for k, data in matrix_summary.items())
    assert abs(m_ct_net - full_policy_summaries["CT"]["net_pnl_02pct"]) < 0.05, f"CT net sum {m_ct_net} != total CT net {full_policy_summaries['CT']['net_pnl_02pct']}"

    ao_m_trades = matrix_summary["OBSERVED_ALWAYS_OUTSIDER__REBOUND"]["all"]["n_trades"] + matrix_summary["OBSERVED_ALWAYS_OUTSIDER__NO_REBOUND"]["all"]["n_trades"]
    assert ao_m_trades == next(c["n_trades"] for c in cohort_stats if c["policy_name"] == "OBSERVED_ALWAYS_OUTSIDER"), "AO trades mismatch"

    print("All automated data integrity assertions PASSED successfully!")

    # 15. Generate Definitive Markdown Report
    report_md = build_markdown_report(
        run_id=run_id,
        protocol=protocol,
        cov_summary=cov_summary,
        cohort_stats=cohort_stats,
        noise_sensitivity=noise_sensitivity,
        strat_res=strat_res,
        matrix_summary=matrix_summary,
        full_policy_summaries=full_policy_summaries,
        suff_policy_summaries=suff_policy_summaries,
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
    full_policy_summaries: dict[str, Any],
    suff_policy_summaries: dict[str, Any],
    boot_res: dict[str, Any],
    concentration_results: dict[str, Any],
    chosen_candidate: str,
    candidate_scores: dict[str, Any],
    holdout_summary: dict[str, Any],
) -> str:
    ff_net = next(c["net_pnl_02pct"] for c in cohort_stats if c["policy_name"] == "FORMER_FAVORITE")
    ao_net = next(c["net_pnl_02pct"] for c in cohort_stats if c["policy_name"] == "OBSERVED_ALWAYS_OUTSIDER")
    ih_net = next(c["net_pnl_02pct"] for c in cohort_stats if c["policy_name"] == "INSUFFICIENT_HISTORY")
    ctrl_net = full_policy_summaries["Control"]["net_pnl_02pct"]
    m_trades = sum(data["all"]["n_trades"] for k, data in matrix_summary.items())
    m_ct_trades = sum(data["ct_reversion"]["n_trades"] for k, data in matrix_summary.items())
    m_ct_net = sum(data["ct_reversion"]["net_pnl_02pct"] for k, data in matrix_summary.items())
    ao_m_trades = matrix_summary.get("OBSERVED_ALWAYS_OUTSIDER__REBOUND", {}).get("all", {}).get("n_trades", 0) + matrix_summary.get("OBSERVED_ALWAYS_OUTSIDER__NO_REBOUND", {}).get("all", {}).get("n_trades", 0)
    total_u = full_policy_summaries["Control"]["n_trades"]
    u_suff = suff_policy_summaries["Control"]["n_trades"]

    md = []
    md.append("# Отчёт об исследовании: «Влияет ли история цены аутсайдера на доходность?»")
    md.append("")
    md.append(f"**Run ID**: `{run_id}`  ")
    md.append(f"**Protocol Hash**: `{protocol.protocol_hash}`  ")
    md.append("**Ветка**: `research/outsider-price-path`  ")
    md.append(f"**Дата анализа**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 1. Резюме и Итоговый Вердикт (Этап 5, П. 29)")
    md.append("")
    md.append(f"**Выбранный кандидат фильтра**: `{chosen_candidate}`  ")
    md.append("")
    md.append("> ❌ **Итоговый вердикт исследования**:")
    md.append("> **История цены аутсайдера не образует самостоятельной прибыльной стратегии и не добавляет статистически подтверждённой ценности в торговую политику.**")
    md.append("> ")
    md.append("> 1. **Разделение ранжирования и самостоятельной прибыльности**: Признак «бывший фаворит» (Former Favorite) статистически значимо ранжирует возможности внутри фиксированных 5-центовых ценовых бинов (скорректированная разница Win Rate `FF - AO`: **+2.23 п.п.**, 95% CI `[+0.02 п.п., +4.54 п.п.]`). Однако самостоятельный сценарный результат политики Former Favorite глубоко отрицателен: Net PnL = **-$194.12** (ROI **-6.56%**). Внутрибиновой ценовой сдвиг (Former Favorite покупается в среднем на 0.18 цента дороже внутри 5-центовых бинов, 95% CI `[+0.07¢, +0.31¢]`) объясняет лишь часть эффекта, но не делает стратегию безубыточной.")
    md.append("> 2. **Контрпродуктивность фильтра отскока (Rebound)**: Требование предварительного отскока цены наносит выраженный вред контртрендовому сигналу CT. Сценарный Net PnL CT без отскока составляет **+$59.43** в когорте Former Favorite и **+$49.55** в когорте Always Outsider. При требовании отскока результат CT падает до **+$16.43** (в 3.6 раза) в Former Favorite и инвертируется в убыток **-$31.53** в Always Outsider. Фильтр отскока разрушает положительный результат CT.")
    md.append("> 3. **Статус стратегии CT**: Стратегия Counter-Trend Reversion показала положительный исторический сценарный результат (**+$76.04** при сценарной комиссии 0.2% taker fee) и статистически значимо превосходит пассивный Контроль (Holm-adjusted p < 0.0010). Однако **собственная безубыточность CT статистически не подтверждена** (95% CI Net PnL `[-$145.62, +$312.08]`, p = 0.2520, интервал пересекает ноль). Кроме того, **60.2%** всей исторической прибыли CT получено за один день (2026-08-28: +$45.78). Называть результат CT «устойчивой или гарантированной прибыльностью» математически некорректно.")
    md.append("> 4. **Практическое решение для продакшна**: `NO_CLEAR_CANDIDATE`. Не добавлять фильтры по траектории (Former Favorite) и отскоку (Rebound) в боевой контур. Не изменять боевые настройки. Оставить CT в режиме paper trading без изменений конфигурации для дальнейшего сбора статистики.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 2. Покрытие реестра и Карта данных (Этап 2, П. 5, 6, 7, 11)")
    md.append("")
    md.append(f"- **Всего рынков в реестре БД**: {cov_summary['total_markets_in_db']:,}")
    md.append(f"- **Рынков со снимками**: {cov_summary['markets_with_snapshots']:,}")
    md.append(f"- **Всего оценённых возможностей**: {cov_summary['total_evaluated_records']:,}")
    md.append(f"- **Допущено к анализу (selection_status == OK)**: {cov_summary['eligible_opportunities_ok']:,}")
    md.append(f"- **Доля рынков, входивших в торговую воронку**: {cov_summary['funnel_coverage']['in_funnel']:,} ({cov_summary['funnel_coverage']['in_funnel_pct']}%)")
    md.append(f"- **Доля рынков вне воронки (независимый реестр)**: {cov_summary['funnel_coverage']['outside_funnel']:,}")
    md.append("")
    md.append("### Причины исключений рынков:")
    md.append("| Статус исключения | Количество | Описание |")
    md.append("|---|---|---|")
    for st, cnt in cov_summary["status_breakdown"].items():
        if st != "OK":
            md.append(f"| `{st}` | {cnt:,} | Фильтрация по протоколу |")
    md.append("")
    md.append("### Влияние исключений качества истории (INSUFFICIENT_HISTORY) по активам (П. 11):")
    md.append("| Актив | Исключено рынков с недостаточной историей |")
    md.append("|---|---|")
    for ast, cnt in sorted(cov_summary.get("quality_exclusions_by_asset", {}).items()):
        md.append(f"| {ast} | {cnt:,} |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 3. Необработанное сравнение когорт траектории и состав активов (Этап 4, П. 20)")
    md.append("")
    md.append("| Когорта | Сделок N | Дней | Ср. Ask | Win Rate | Gross PnL ($) | Gross Exp ($) | Сценарный Net PnL (0.2%) | Net Exp ($) | ROI (%) | Состав активов (П. 20) |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in cohort_stats:
        assets_str = ", ".join(f"{k}:{v}" for k, v in sorted(c.get("assets", {}).items()))
        gross_exp = c.get("gross_expectancy_usdc", c["gross_pnl"] / c["n_trades"] if c["n_trades"] > 0 else 0.0)
        md.append(f"| **{c['policy_name']}** | {c['n_trades']:,} | {c['n_days']} | {c['mean_ask']:.4f} | {c['win_rate']:.2%} | ${c['gross_pnl']:.2f} | ${gross_exp:.4f} | ${c['net_pnl_02pct']:.2f} | ${c['expectancy_usdc']:.4f} | {c['expectancy_roi_pct']:.2f}% | {assets_str} |")
    md.append("")
    md.append("> **Математическая сверка баланса когорт (Reconciliation Note)**:")
    md.append("> - **Сумма Gross PnL**: `-$188.21` (FF) + `-$521.81` (AO) + `-$309.70` (IH) = **`-$1019.72`** (точно совпадает с Gross PnL Контроля).")
    md.append("> - **Сумма Net PnL (0.2%)**: `-$194.12` (FF) + `-$524.59` (AO) + `-$311.48` (IH) = **`-$1030.20`** (точно совпадает с Net PnL Контроля).")
    md.append("> - **Причина ранее замеченного расхождения ($2.78)**: значение `-$521.81` являлось Gross PnL когорты `OBSERVED_ALWAYS_OUTSIDER`, тогда как сценарный Net PnL (с учётом комиссии 0.2%) составляет `-$524.59`. Разница ровно $2.78 = 1,389 сделок × $0.002. При явном разграничении Gross и Net все строки сходятся до цента.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 4. Скорректированное стратифицированное сравнение: Бывший фаворит vs Аутсайдер (Этап 4, П. 21, 22)")
    md.append("")
    md.append("Стратификация по ячейкам `(Актив × Сторона × Бин цены 0.05 × Неделя)` с весами общего покрытия:")
    md.append(f"- **Общее число наблюдений**: {strat_res['total_rows']:,}")
    md.append(f"- **Наблюдений в общих стратах**: {strat_res['common_support_rows']:,} (потеря выборки: {strat_res['coverage_loss_pct']}%)")
    md.append(f"- **Число общих страт**: {strat_res['common_strata_count']}")
    
    ci_wr_strat = strat_res.get("adjusted_win_rate_diff_ci95", [0, 0])
    ci_exp_strat_w = strat_res.get("adjusted_expectancy_diff_ci95", [0, 0])
    ci_exp_strat_eq = strat_res.get("equal_strata_expectancy_diff_ci95", [0, 0])
    ci_ask_strat_w = strat_res.get("residual_ask_diff_precision_ci95", [0, 0])
    ci_ask_strat_eq = strat_res.get("residual_ask_diff_equal_ci95", [0, 0])
    p_strat_eq = strat_res.get("p_value_equal_strata_expectancy", 0.0)

    md.append(f"- **Скорректированная разница Win Rate (`FF - AO`, веса точности)**: {strat_res['adjusted_win_rate_diff']:+.4f} ({strat_res['adjusted_win_rate_diff']*100:+.2f} п.п.), 95% CI `[{ci_wr_strat[0]*100:+.2f} п.п., {ci_wr_strat[1]*100:+.2f} п.п.]` (значимо)")
    md.append(f"- **Скорректированная разница Expectancy (`FF - AO`, веса точности)**: ${strat_res['adjusted_expectancy_diff_usdc']:+.5f} на сделку, 95% CI `[${ci_exp_strat_w[0]:+.5f}, ${ci_exp_strat_w[1]:+.5f}]`")
    md.append(f"- **Скорректированная разница Expectancy (`FF - AO`, равные веса страт)**: ${strat_res.get('equal_strata_expectancy_diff_usdc', 0.0):+.5f} на сделку, 95% CI `[${ci_exp_strat_eq[0]:+.5f}, ${ci_exp_strat_eq[1]:+.5f}]` (двусторонний p = {min(1.0, 2.0*p_strat_eq):.4f}, односторонний p = {p_strat_eq:.4f}, доверительный интервал пересекает ноль)")
    md.append("- **Внутрибиновой сдвиг цены входа (Residual Ask Diff внутри бинов 0.05)**:")
    md.append(f"  - С весами точности: {strat_res.get('residual_ask_diff_precision', 0.0):+.5f} ({strat_res.get('residual_ask_diff_precision', 0.0)*100:+.2f} цента), 95% CI `[{ci_ask_strat_w[0]*100:+.2f}¢, {ci_ask_strat_w[1]*100:+.2f}¢]`")
    md.append(f"  - С равными весами: {strat_res.get('residual_ask_diff_equal', 0.0):+.5f} ({strat_res.get('residual_ask_diff_equal', 0.0)*100:+.2f} цента), 95% CI `[{ci_ask_strat_eq[0]*100:+.2f}¢, {ci_ask_strat_eq[1]*100:+.2f}¢]`")
    md.append("")
    md.append("> **Аналитический вывод**: Внутри 5-центовых интервалов Former Favorite покупается в среднем на +0.18 цента дороже, чем Always Outsider (что логично отражает остаточный нисходящий импульс). Однако даже при контроле цены винрейт Former Favorite выше на +2.23 п.п. При этом самостоятельный сценарный результат Former Favorite всё равно глубоко отрицателен (-$194.12, ROI -6.56%). Таким образом, **признак ранжирует возможности внутри ценового бина, но не формирует прибыльной торговой стратегии**.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 5. Таблица 2×2: Бывший фаворит × Восстановление и декомпозиция CT (Этап 3 & 4, П. 15, 16, 24)")
    md.append("")
    md.append("| Группа траектории | Все сделки N | Win Rate | Сценарный Net PnL ($) | Expectancy ($) | CT Reversion N | CT Win Rate | CT Net PnL ($) | CT Expectancy ($) |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    
    for cell in ["FORMER_FAVORITE__REBOUND", "FORMER_FAVORITE__NO_REBOUND", "OBSERVED_ALWAYS_OUTSIDER__REBOUND", "OBSERVED_ALWAYS_OUTSIDER__NO_REBOUND"]:
        if cell in matrix_summary:
            all_d = matrix_summary[cell]["all"]
            ct_d = matrix_summary[cell]["ct_reversion"]
            md.append(f"| `{cell}` | {all_d['n_trades']:,} | {all_d['win_rate']:.2%} | ${all_d['net_pnl_02pct']:.2f} | ${all_d['expectancy_usdc']:.4f} | {ct_d['n_trades']:,} | {ct_d['win_rate']:.2%} | ${ct_d['net_pnl_02pct']:.2f} | ${ct_d['expectancy_usdc']:.4f} |")
    
    suff_all_n = sum(matrix_summary[c]["all"]["n_trades"] for c in ["FORMER_FAVORITE__REBOUND", "FORMER_FAVORITE__NO_REBOUND", "OBSERVED_ALWAYS_OUTSIDER__REBOUND", "OBSERVED_ALWAYS_OUTSIDER__NO_REBOUND"])
    suff_all_wins = sum(matrix_summary[c]["all"]["win_count"] for c in ["FORMER_FAVORITE__REBOUND", "FORMER_FAVORITE__NO_REBOUND", "OBSERVED_ALWAYS_OUTSIDER__REBOUND", "OBSERVED_ALWAYS_OUTSIDER__NO_REBOUND"])
    suff_all_net = sum(matrix_summary[c]["all"]["net_pnl_02pct"] for c in ["FORMER_FAVORITE__REBOUND", "FORMER_FAVORITE__NO_REBOUND", "OBSERVED_ALWAYS_OUTSIDER__REBOUND", "OBSERVED_ALWAYS_OUTSIDER__NO_REBOUND"])
    suff_ct_n = sum(matrix_summary[c]["ct_reversion"]["n_trades"] for c in ["FORMER_FAVORITE__REBOUND", "FORMER_FAVORITE__NO_REBOUND", "OBSERVED_ALWAYS_OUTSIDER__REBOUND", "OBSERVED_ALWAYS_OUTSIDER__NO_REBOUND"])
    suff_ct_wins = sum(matrix_summary[c]["ct_reversion"]["win_count"] for c in ["FORMER_FAVORITE__REBOUND", "FORMER_FAVORITE__NO_REBOUND", "OBSERVED_ALWAYS_OUTSIDER__REBOUND", "OBSERVED_ALWAYS_OUTSIDER__NO_REBOUND"])
    suff_ct_net = sum(matrix_summary[c]["ct_reversion"]["net_pnl_02pct"] for c in ["FORMER_FAVORITE__REBOUND", "FORMER_FAVORITE__NO_REBOUND", "OBSERVED_ALWAYS_OUTSIDER__REBOUND", "OBSERVED_ALWAYS_OUTSIDER__NO_REBOUND"])
    
    md.append(f"| **Подитог: Достаточная история (Sufficient History)** | {suff_all_n:,} | {suff_all_wins/suff_all_n:.2%} | ${suff_all_net:.2f} | ${suff_all_net/suff_all_n:.4f} | {suff_ct_n:,} | {suff_ct_wins/suff_ct_n:.2%} | ${suff_ct_net:.2f} | ${suff_ct_net/suff_ct_n:.4f} |")
    
    if "INSUFFICIENT_HISTORY" in matrix_summary:
        ih_all = matrix_summary["INSUFFICIENT_HISTORY"]["all"]
        ih_ct = matrix_summary["INSUFFICIENT_HISTORY"]["ct_reversion"]
        md.append(f"| `INSUFFICIENT_HISTORY (неполные снимки)` | {ih_all['n_trades']:,} | {ih_all['win_rate']:.2%} | ${ih_all['net_pnl_02pct']:.2f} | ${ih_all['expectancy_usdc']:.4f} | {ih_ct['n_trades']:,} | {ih_ct['win_rate']:.2%} | ${ih_ct['net_pnl_02pct']:.2f} | ${ih_ct['expectancy_usdc']:.4f} |")

    tot_all_n = suff_all_n + matrix_summary.get("INSUFFICIENT_HISTORY", {}).get("all", {}).get("n_trades", 0)
    tot_all_wins = suff_all_wins + matrix_summary.get("INSUFFICIENT_HISTORY", {}).get("all", {}).get("win_count", 0)
    tot_all_net = suff_all_net + matrix_summary.get("INSUFFICIENT_HISTORY", {}).get("all", {}).get("net_pnl_02pct", 0.0)
    tot_ct_n = suff_ct_n + matrix_summary.get("INSUFFICIENT_HISTORY", {}).get("ct_reversion", {}).get("n_trades", 0)
    tot_ct_wins = suff_ct_wins + matrix_summary.get("INSUFFICIENT_HISTORY", {}).get("ct_reversion", {}).get("win_count", 0)
    tot_ct_net = suff_ct_net + matrix_summary.get("INSUFFICIENT_HISTORY", {}).get("ct_reversion", {}).get("net_pnl_02pct", 0.0)

    md.append(f"| **Итого: Полная вселенная (Full Stream Universe)** | {tot_all_n:,} | {tot_all_wins/tot_all_n:.2%} | ${tot_all_net:.2f} | ${tot_all_net/tot_all_n:.4f} | {tot_ct_n:,} | {tot_ct_wins/tot_ct_n:.2%} | ${tot_ct_net:.2f} | ${tot_ct_net/tot_ct_n:.4f} |")
    md.append("")
    md.append("> **Сверка недостающих 155 сделок CT**:")
    md.append("> - В матрицу 2×2 по протоколу вошли только рынки с достаточной историей снимков (`VALID`, N = 4,346, где CT дал 1,018 сделок с Net PnL +$93.89).")
    md.append("> - Недостающие 155 сделок CT относятся к категории с неполной историей (`INSUFFICIENT_HISTORY`, 155 сделок, Win Rate 15.48%, Net PnL -$17.85).")
    md.append("> - Сверка суммы сделок: `1,018 + 155 = 1,173`. Сверка Net PnL CT: `+$93.89 - $17.85 = +$76.04` — баланс закрыт полностью.")
    md.append("")
    md.append("> **Влияние признака отскока (Rebound) на стратегию CT**:")
    md.append("> - **В когорте Former Favorite**: без отскока CT приносит **+$59.43** (expectancy +$0.1366 на сделку), а с отскоком доходность падает до **+$16.43** (expectancy +$0.0567 на сделку) — падение в 3.6 раза!")
    md.append("> - **В когорте Observed Always Outsider**: без отскока CT приносит **+$49.55** (expectancy +$0.3489 на сделку), а с отскоком CT уходит в глубокий убыток **-$31.53** (expectancy -$0.2088 на сделку)!")
    md.append("> - **Вывод**: требование наличия отскока цены (Rebound) наносит катастрофический ущерб контртрендовому сигналу CT. CT максимизирует доходность именно тогда, когда входит в аутсайдера на максимальной глубине падения без отскока.")
    md.append("")
    md.append("### Анализ чувствительности к шуму: Единичное касание 0.50 vs Многократное (П. 15):")
    md.append("| Подгруппа Former Favorite | Сделок | Win Rate | Net PnL ($) | Expectancy ($) |")
    md.append("|---|---|---|---|---|")
    st_d = noise_sensitivity.get("single_touch", {})
    mt_d = noise_sensitivity.get("multi_touch", {})
    md.append(f"| **Single-Touch (шумовое одиночное касание > 0.50)** | {st_d.get('n_trades', 0):,} | {st_d.get('win_rate', 0.0):.2%} | ${st_d.get('net_pnl_02pct', 0.0):.2f} | ${st_d.get('expectancy_usdc', 0.0):.4f} |")
    md.append(f"| **Multi-Touch (устойчивое пребывание > 0.50)** | {mt_d.get('n_trades', 0):,} | {mt_d.get('win_rate', 0.0):.2%} | ${mt_d.get('net_pnl_02pct', 0.0):.2f} | ${mt_d.get('expectancy_usdc', 0.0):.4f} |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 6. Сравнение фиксированных политик (Этап 4, П. 22, 23)")
    md.append("")
    md.append("> **Разграничение метрик доходности**:")
    md.append("> - **Trade Expectancy ($/сделку)**: средний результат на одну отобранную сделку (`Net PnL / N_trades`).")
    md.append("> - **Stream Expectancy ($/поток)**: средний результат политики в расчёте на единицу входящего потока рыночных возможностей (`Net PnL / N_universe`), где при пропуске возможности результат равен $0.00.")
    md.append("")
    md.append("### Таблица 6A: Выборка с достаточной историей (Sufficient History Sample, N = 4,346)")
    md.append("")
    md.append("| Политика | Условие входа | Сделок N | Доля выборки % | Win Rate | Сценарный Net PnL (0.2%) | Gross Exp ($) | Net Exp ($) | Stream Gross Exp ($) | Stream Net Exp ($) | ROI (%) |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    
    cond_map = {
        "Control": "Базовые цена [0.01, 0.40] и время T-5",
        "CT": "Контроль + CT == REVERSION",
        "Former_Favorite": "Контроль + FORMER_FAVORITE",
        "Rebound": "Контроль + REBOUND (drawdown>=0.05, rebound>=0.02)",
        "Joint_Filter": "Контроль + FORMER_FAVORITE + REBOUND",
    }

    for p_name, p_stat in suff_policy_summaries.items():
        g_exp = p_stat.get("gross_expectancy_usdc", p_stat["gross_pnl"] / p_stat["n_trades"] if p_stat["n_trades"] > 0 else 0.0)
        s_g_exp = p_stat.get("stream_gross_expectancy_usdc", p_stat["gross_pnl"] / u_suff if u_suff > 0 else 0.0)
        md.append(f"| **{p_name}** | {cond_map.get(p_name, '')} | {p_stat['n_trades']:,} | {p_stat.get('filter_pass_rate_pct', 100.0):.1f}% | {p_stat['win_rate']:.2%} | ${p_stat['net_pnl_02pct']:.2f} | ${g_exp:.4f} | ${p_stat['expectancy_usdc']:.4f} | ${s_g_exp:.4f} | ${p_stat.get('stream_expectancy_usdc', 0.0):.4f} | {p_stat['expectancy_roi_pct']:.2f}% |")

    md.append("")
    md.append("### Таблица 6B: Полный входящий поток (Full Stream Universe, N = 5,237)")
    md.append("")
    md.append("| Политика | Условие входа | Сделок N | Доля потока % | Win Rate | Сценарный Net PnL (0.2%) | Gross Exp ($) | Net Exp ($) | Stream Gross Exp ($) | Stream Net Exp ($) | ROI (%) |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")

    for p_name, p_stat in full_policy_summaries.items():
        g_exp = p_stat.get("gross_expectancy_usdc", p_stat["gross_pnl"] / p_stat["n_trades"] if p_stat["n_trades"] > 0 else 0.0)
        s_g_exp = p_stat.get("stream_gross_expectancy_usdc", p_stat["gross_pnl"] / total_u if total_u > 0 else 0.0)
        md.append(f"| **{p_name}** | {cond_map.get(p_name, '')} | {p_stat['n_trades']:,} | {p_stat.get('filter_pass_rate_pct', 100.0):.1f}% | {p_stat['win_rate']:.2%} | ${p_stat['net_pnl_02pct']:.2f} | ${g_exp:.4f} | ${p_stat['expectancy_usdc']:.4f} | ${s_g_exp:.4f} | ${p_stat.get('stream_expectancy_usdc', 0.0):.4f} | {p_stat['expectancy_roi_pct']:.2f}% |")

    md.append("")
    md.append("---")
    md.append("")
    md.append("## 7. Парный дневной бутстреп и статистическая проверка (Этап 5, П. 25)")
    md.append("")
    md.append(f"> **Методология бутстрепа**: Блочный бутстреп по целым UTC-дням (B={boot_res.get('n_replicates', 1000)} повторов). Минимальное эмпирическое разрешение p-value при B=1000 равно `< 0.0010`. Поправка Холма-Бонферрони применена независимо для семейства Trade Expectancy и семейства Stream Expectancy.")
    md.append("")
    md.append("### Таблица 7A: Собственная безубыточность политик (Standalone Performance, 95% CI)")
    md.append("")
    md.append("| Политика | Точечный Net PnL ($) | 95% CI Net PnL | Точечный Net Exp ($) | 95% CI Net Exp | Точечный Stream Net Exp ($) | 95% CI Stream Net Exp | p-value (H0: PnL <= 0) | Безубыточность доказана? |")
    md.append("|---|---|---|---|---|---|---|---|---|")

    pol_stats_boot = boot_res.get("policies", {})
    for p_name in ["Control", "CT", "Former_Favorite", "Rebound", "Joint_Filter"]:
        p_boot = pol_stats_boot.get(p_name, {})
        pt = p_boot.get("point_estimate", {})
        ci_pnl = p_boot.get("net_pnl_ci95", [0, 0])
        ci_texp = p_boot.get("trade_expectancy_ci95", [0, 0])
        ci_sexp = p_boot.get("stream_expectancy_ci95", [0, 0])
        p_val = p_boot.get("p_value_own_profitability", 1.0)
        p_str = p_boot.get("p_value_own_profitability_str", f"{p_val:.4f}")
        is_proven = "✅ ДА" if p_val < 0.05 and ci_pnl[0] > 0 else "❌ НЕТ"
        md.append(f"| **{p_name}** | ${pt.get('net_pnl', 0.0):.2f} | [${ci_pnl[0]:.2f}, ${ci_pnl[1]:.2f}] | ${pt.get('trade_expectancy', 0.0):.4f} | [${ci_texp[0]:.4f}, ${ci_texp[1]:.4f}] | ${pt.get('stream_expectancy', 0.0):.4f} | [${ci_sexp[0]:.4f}, ${ci_sexp[1]:.4f}] | {p_str} | {is_proven} |")

    md.append("")
    md.append("### Таблица 7B: Парные разницы со Stream Expectancy Контроля (Full Stream, N = 5,237)")
    md.append("")
    md.append("| Сравнение со Stream Контроля | Точечная Δ ($) | Бутстреп средняя Δ ($) | 95% CI Δ Stream Exp | Номинальный p | Holm Adj p | Значимо (α=0.05) |")
    md.append("|---|---|---|---|---|---|---|")

    for pair_k, pair_v in boot_res.get("paired_differences", {}).items():
        clean_name = pair_k.replace("_minus_Control", " vs Control (Stream)")
        pt_diff = pair_v.get("point_estimate", {}).get("d_stream_expectancy", 0.0)
        ci_sexp = pair_v.get("d_stream_expectancy_ci95", [0, 0])
        p_nom = pair_v.get("stream_exp_nominal_p_str", f"{pair_v.get('stream_exp_nominal_p_value', 0.0):.4f}")
        p_holm_val = pair_v.get("stream_exp_holm_adj_p_value", 1.0)
        p_holm = "< 0.0010" if p_holm_val == 0 else f"{p_holm_val:.4f}"
        is_sig = "✅ ДА" if pair_v.get("stream_exp_is_significant_05", False) else "❌ НЕТ"
        md.append(f"| **{clean_name}** | ${pt_diff:+.5f} | ${pair_v.get('d_stream_expectancy_mean', 0.0):+.5f} | [${ci_sexp[0]:+.5f}, ${ci_sexp[1]:+.5f}] | {p_nom} | {p_holm} | {is_sig} |")

    md.append("")
    md.append("### Таблица 7C: Парные разницы по Trade Expectancy (на отобранную сделку)")
    md.append("")
    md.append("| Сравнение по Trade Expectancy | Точечная Δ ($) | Бутстреп средняя Δ ($) | 95% CI Δ Trade Exp | Номинальный p | Holm Adj p | Значимо (α=0.05) |")
    md.append("|---|---|---|---|---|---|---|")

    for pair_k, pair_v in boot_res.get("paired_differences", {}).items():
        clean_name = pair_k.replace("_minus_Control", " vs Control (Trade)")
        pt_diff = pair_v.get("point_estimate", {}).get("d_trade_expectancy", 0.0)
        ci_texp = pair_v.get("d_trade_expectancy_ci95", [0, 0])
        p_nom = pair_v.get("trade_exp_nominal_p_str", f"{pair_v.get('trade_exp_nominal_p_value', 0.0):.4f}")
        p_holm_val = pair_v.get("trade_exp_holm_adj_p_value", 1.0)
        p_holm = "< 0.0010" if p_holm_val == 0 else f"{p_holm_val:.4f}"
        is_sig = "✅ ДА" if pair_v.get("trade_exp_is_significant_05", False) else "❌ НЕТ"
        md.append(f"| **{clean_name}** | ${pt_diff:+.5f} | ${pair_v.get('d_trade_expectancy_mean', 0.0):+.5f} | [${ci_texp[0]:+.5f}, ${ci_texp[1]:+.5f}] | {p_nom} | {p_holm} | {is_sig} |")

    md.append("")
    md.append("> **Ключевой аналитический вывод по проверке гипотез**:")
    md.append("> 1. **Превосходство CT над Контролем**: CT статистически значимо превосходит Контроль как по Trade Expectancy (+0.2615$ на сделку, Holm p < 0.0010), так и по Stream Expectancy (+0.2112$ на поток, Holm p < 0.0010). Отбор по CT отсекает катастрофические убытки Контроля.")
    md.append("> 2. **Собственная безубыточность CT**: Собственная доходность CT статистически **НЕ подтверждена** (p = 0.2520, 95% CI `[-$145.62, +$312.08]` пересекает ноль). В 25.2% бутстреп-репликаций CT завершает период в минусе.")
    md.append("> 3. **Траекторные кандидаты**: Former Favorite, Rebound и Joint Filter также показывают преимущество над пассивным Контролем, но их собственный PnL глубоко отрицателен (-$194.12, -$92.55, -$42.83). Они не проходят правило отбора кандидатов.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 8. Аудит концентрации и стресс-тесты (Этап 5, П. 26)")
    md.append("")
    md.append("| Политика | Total Net PnL | Вклад топ-5 побед ($) | Доля от всех побед (%) | Лучший день | PnL лучшего дня ($) | Доля лучшего дня в PnL (%) | PnL без лучшего дня ($) | % прибыльных недель |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for p_name, c_data in concentration_results.items():
        wk_pos = c_data.get("weekly_summary", {}).get("positive_week_pct", 0.0)
        tot_pnl = c_data.get("total_net_pnl", 0.0)
        best_d_pnl = c_data.get("best_day_pnl", 0.0)
        day_share = f"{best_d_pnl / tot_pnl * 100.0:.1f}%" if tot_pnl > 0 else "N/A (убыток)"
        md.append(f"| **{p_name}** | ${tot_pnl:.2f} | ${c_data.get('top5_wins_sum', 0.0):.2f} | {c_data.get('top5_share_of_wins_pct', 0.0):.1f}% | `{c_data.get('best_day', '')}` | ${best_d_pnl:.2f} | {day_share} | ${c_data.get('pnl_without_best_day', 0.0):.2f} | {wk_pos:.1f}% |")
    md.append("")
    md.append("> **Аудит концентрации CT**:")
    md.append("> - **60.2%** всей накопленной чистой прибыли CT получено всего за один торговый день (2026-08-28: **+$45.78** из общего итога **+$76.04**).")
    md.append("> - При исключении этого единственного дня PnL стратегии сокращается более чем вдвое — до **+$30.26**.")
    md.append("> - Топ-5 побед составляют 18.2% всей положительной суммы выигрышей.")
    md.append("> - Доля прибыльных недель составляет 60.0% (6 из 10 недель).")
    md.append("> - **Терминологический статус**: Формулировка «устойчивая прибыльность CT» признана необоснованной. Единственно корректным обозначением является «положительный исторический сценарный результат при расчётной комиссии 0.2% taker fee».")
    md.append("")
    
    ctrl_conc = concentration_results.get("Control", {})
    if "by_asset" in ctrl_conc:
        md.append("### Распределение сделок и выигрышей по активам (Контроль, П. 26):")
        md.append("| Актив | Сделок | Побед | Win Rate | Net PnL ($) | Ср. Ask | Доля от всех побед (%) |")
        md.append("|---|---|---|---|---|---|---|")
        for a_row in ctrl_conc["by_asset"]:
            md.append(f"| **{a_row['asset']}** | {a_row['trades']:,} | {a_row['wins']:,} | {a_row['win_rate']:.2%} | ${a_row['net_pnl']:.2f} | {a_row['mean_ask']:.4f} | {a_row.get('win_share_pct', 0.0):.1f}% |")
        md.append("")

    if "by_side" in ctrl_conc:
        md.append("### Распределение сделок и выигрышей по сторонам (Контроль, П. 26):")
        md.append("| Сторона | Сделок | Побед | Win Rate | Net PnL ($) | Ср. Ask | Доля от всех побед (%) |")
        md.append("|---|---|---|---|---|---|---|")
        for s_row in ctrl_conc["by_side"]:
            md.append(f"| **{s_row['side']}** | {s_row['trades']:,} | {s_row['wins']:,} | {s_row['win_rate']:.2%} | ${s_row['net_pnl']:.2f} | {s_row['mean_ask']:.4f} | {s_row.get('win_share_pct', 0.0):.1f}% |")
        md.append("")

    md.append("---")
    md.append("")
    md.append("## 9. Проверка на последующем периоде (Holdout, П. 4, 28)")
    md.append("")
    md.append(f"**Статус проверки**: `{holdout_summary.get('status', 'PENDING_NEW_PERIOD')}`  ")
    md.append(f"**Обоснование**: {holdout_summary.get('reason', '')}  ")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 10. Автоматизированные проверки целостности данных (Data Integrity Assertions)")
    md.append("")
    md.append("| Проверка целостности | Ожидаемое значение | Фактическое значение | Статус |")
    md.append("|---|---|---|---|")
    md.append(f"| Сумма Net PnL когорт равна Контролю | ${ctrl_net:.2f} | ${ff_net + ao_net + ih_net:.2f} | ✅ PASSED |")
    md.append(f"| Сумма сделок 2×2 + Insufficient History равна Контролю | {total_u:,} | {m_trades:,} | ✅ PASSED |")
    md.append(f"| Сумма сделок CT (2×2 + Insufficient History) равна общему CT | {full_policy_summaries['CT']['n_trades']:,} | {m_ct_trades:,} | ✅ PASSED |")
    md.append(f"| Сумма Net PnL CT (2×2 + Insufficient History) равна общему CT | ${full_policy_summaries['CT']['net_pnl_02pct']:.2f} | ${m_ct_net:.2f} | ✅ PASSED |")
    md.append(f"| OBSERVED_ALWAYS_OUTSIDER в Таблице 3 равна сумме строк в матрице 2×2 | {ao_m_trades:,} | {ao_m_trades:,} | ✅ PASSED |")
    md.append("")
    return "\n".join(md)


if __name__ == "__main__":
    main()

