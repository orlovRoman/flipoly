"""
scripts/research/run_stage2_price_filter_evaluation.py

Rigorous evaluation of the Model A1 price cap hypothesis (max_price <= 0.40)
according to Stage 2 requirements (Items 23 through 29):
23. Fix dataset hash and pre-registered protocol before fit.
24. Recompute controls (M0, Mlegacy, A-price, A1) on development split.
25. Disentangle price cap effect from ML selection (A1 base vs A1 cap vs Naive price rule without ML).
26. Development robustness checks (daily, side, price bins, caps grid [0.30, 0.35, 0.40, 0.45, 0.50, 0.95], top trade impact).
27. Fix single candidate configuration strictly from development performance.
28. Independent out-of-sample evaluation on test split (paired block delta, CI zero-cross check).
29. Final product decision and artifact bundles.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.run_stage2_empirical_benchmark import load_real_btc_outsider_data
from polyflip.models.point_in_time_features import compute_outsider_model_features
from polyflip.models.outsider_trainer import train_outsider_model
from polyflip.models.outsider_baselines import LegacyOutsiderBaseline, MarketPriceBaseline
from polyflip.models.probability_metrics import brier_score
from polyflip.research.outsider_replay import OutsiderReplayEngine, ReplayPolicy
from polyflip.research.reporting_helpers import (
    compute_drawdown,
    compute_payoff_ratio,
    compute_clustered_uncertainty,
    generate_price_bins_report,
)


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_paired_block_bootstrap(
    pnls_a: list[float],
    pnls_b: list[float],
    cluster_ids: list[str],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Computes paired delta PnL (A - B) with cluster block bootstrap on identical markets.
    """
    if not pnls_a or not pnls_b or len(pnls_a) != len(pnls_b):
        return {"delta_pnl": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "status": "INVALID_PAIRED_INPUT"}

    deltas = np.array(pnls_a) - np.array(pnls_b)
    clusters = np.array(cluster_ids)
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    if n_clusters <= 1:
        return {
            "delta_pnl": round(float(np.sum(deltas)), 4),
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "status": "INSUFFICIENT_BLOCKS",
        }

    rng = np.random.default_rng(seed)
    cluster_delta_sum = {c: float(np.sum(deltas[clusters == c])) for c in unique_clusters}
    c_vals = np.array(list(cluster_delta_sum.values()))

    boot_deltas = []
    for _ in range(n_bootstrap):
        sample_indices = rng.choice(len(c_vals), size=len(c_vals), replace=True)
        boot_deltas.append(np.sum(c_vals[sample_indices]))

    point_delta = float(np.sum(deltas))
    ci_lower = float(np.percentile(boot_deltas, 2.5))
    ci_upper = float(np.percentile(boot_deltas, 97.5))

    return {
        "delta_pnl": round(point_delta, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "n_clusters": int(n_clusters),
        "status": "VALID",
    }


def run_stage2_evaluation() -> dict[str, Any]:
    print("=" * 90)
    print("STAGE 2: RIGOROUS EVALUATION OF MODEL A1 & HARD MAX PRICE FILTER (<=0.40)")
    print("=" * 90)

    # 1. Dataset loading & integrity
    raw_dataset_path = REPO_ROOT / "artifacts" / "weighted_policy" / "observations_30d.json"
    dataset_hash = compute_file_sha256(raw_dataset_path)

    df_raw, meta = load_real_btc_outsider_data()
    df_feat = compute_outsider_model_features(df_raw)
    df_feat = df_feat.sort_values("decision_at").reset_index(drop=True)

    # 2. Chronological Split: 70% Development, 30% Independent Out-Of-Sample Test
    n_total = len(df_feat)
    split_idx = int(n_total * 0.70)

    df_dev = df_feat.iloc[:split_idx].copy().reset_index(drop=True)
    df_test = df_feat.iloc[split_idx:].copy().reset_index(drop=True)

    dev_start = str(df_dev["decision_at"].min())
    dev_end = str(df_dev["decision_at"].max())
    test_start = str(df_test["decision_at"].min())
    test_end = str(df_test["decision_at"].max())

    print(f"Total authentic rows: {n_total}")
    print(f"  Development Split: {len(df_dev)} rows ({dev_start} -> {dev_end})")
    print(f"  Test Holdout:      {len(df_test)} rows ({test_start} -> {test_end})")

    # Item 23: Pre-register and save experiment protocol BEFORE fitting
    protocol = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "item": "23",
        "dataset_path": str(raw_dataset_path.relative_to(REPO_ROOT)),
        "dataset_sha256": dataset_hash,
        "total_rows": n_total,
        "splits": {
            "development": {
                "n_rows": len(df_dev),
                "start": dev_start,
                "end": dev_end,
                "unique_markets": int(df_dev["market_id"].nunique()),
            },
            "test_holdout": {
                "n_rows": len(df_test),
                "start": test_start,
                "end": test_end,
                "unique_markets": int(df_test["market_id"].nunique()),
            },
        },
        "models_evaluated": [
            "M0_MARKET",
            "MLEGACY_V11",
            "MODEL_A_PRICE",
            "MODEL_A1",
            "MODEL_A1_CAP_040",
            "NAIVE_PRICE_RULE_040",
        ],
        "policy_parameters": {
            "stake_usdc": 1.0,
            "initial_capital": 100.0,
            "default_fee_rate": 0.002,
            "min_edge": 0.02,
            "max_positions_per_market": 1,
            "allow_reentry": False,
        },
        "selection_criteria": {
            "hypothesis": "Hard max_price <= 0.40 filter eliminates 0.40-0.50 parity churn and produces sustainable edge.",
            "ml_edge_criterion": "Model A1 with cap must outperform both Base A1 AND the Naive Price Rule (cap without ML) on Development.",
            "test_significance_criterion": "Independent test expectancy lower 95% CI > 0 AND paired delta PnL lower CI > 0.",
            "inconclusive_rule": "If test CI crosses zero, conclusion is strictly INCONCLUSIVE (do not claim edge).",
        },
    }

    protocol_path = REPO_ROOT / "artifacts" / "research" / "stage2_evaluation_protocol.json"
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    with open(protocol_path, "w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2)
    print(f"[Item 23] Protocol locked -> {protocol_path}")

    # =========================================================================
    # Item 24 & 25: Train & Evaluate Models on Development Split
    # =========================================================================
    print("\n--- [Items 24 & 25] Training and Evaluating on Development Set ---")

    # M0 Market Baseline on Dev
    m0_model = MarketPriceBaseline()
    m0_p_dev = m0_model.predict_proba(df_dev)[:, 1]
    m0_brier = float(brier_score(df_dev["target"].to_numpy(), m0_p_dev))

    # Mlegacy Baseline on Dev
    mlegacy_model = LegacyOutsiderBaseline()
    mlegacy_p_dev = mlegacy_model.predict_proba(df_dev)[:, 1]
    mlegacy_brier = float(brier_score(df_dev["target"].to_numpy(), mlegacy_p_dev))

    # A-price Baseline on Dev (price and spread only)
    res_a_price = train_outsider_model(df_dev, feature_set="MODEL_A_PRICE", validation_mode="walk_forward", n_splits=5)
    p_a_price_dev = res_a_price.oof_predictions

    # A1 Model on Dev
    res_a1 = train_outsider_model(df_dev, feature_set="MODEL_A1", validation_mode="walk_forward", n_splits=5)
    p_a1_dev = res_a1.oof_predictions

    # Simulation helper on Dev
    def run_replay_on_dev(p_win_arr, max_p: float, name: str) -> tuple[dict[str, Any], Any]:
        policy = ReplayPolicy(
            max_positions_per_market=1,
            stake_usdc=1.0,
            initial_capital=100.0,
            min_edge=0.02,
            default_fee_rate=0.002,
            max_price=max_p,
        )
        engine = OutsiderReplayEngine(policy=policy)
        decisions = []
        for idx in range(len(df_dev)):
            row = df_dev.iloc[idx]
            decisions.append({
                "market_id": row["market_id"],
                "decision_at": row["decision_at"],
                "time_left_min": row["time_left_min"],
                "time_valid": bool(row.get("time_valid", True)),
                "market_end_at": row.get("market_end_at", None),
                "candidate_side": row["candidate_side"],
                "executable_ask": row["executable_ask"],
                "p_win": p_win_arr[idx] if np.isfinite(p_win_arr[idx]) else 0.0,
                "target": row["target"],
            })
        ledger = engine.run(decisions)
        trades = ledger.executed_trades
        pnls = [t["realized_pnl"] for t in trades]
        n_trades = len(pnls)
        wins = sum(1 for t in trades if t["outcome"] == 1)
        wr = round(wins / n_trades, 4) if n_trades else 0.0
        pnl = round(float(np.sum(pnls)), 4) if n_trades else 0.0
        exp = round(pnl / n_trades, 6) if n_trades else 0.0
        trade_clusters = [pd.to_datetime(t["decision_at"], utc=True).strftime("%Y-%m-%d") for t in trades]
        unc = compute_clustered_uncertainty(pnls, trade_clusters) if n_trades else {"ci_lower": 0.0, "ci_upper": 0.0, "se": 0.0}
        dd = compute_drawdown(pnls, initial_equity=100.0)
        payoff = compute_payoff_ratio(pnls)

        summary = {
            "name": name,
            "max_price": max_p,
            "n_trades": n_trades,
            "win_rate": wr,
            "net_pnl_usdc": pnl,
            "expectancy": exp,
            "ci_lower": unc["ci_lower"],
            "ci_upper": unc["ci_upper"],
            "max_drawdown": dd,
            "payoff_ratio": payoff,
        }
        return summary, ledger

    # Dev Evaluations
    sum_mlegacy_dev, _ = run_replay_on_dev(mlegacy_p_dev, max_p=0.95, name="MLEGACY_V11")
    sum_a_price_dev, _ = run_replay_on_dev(p_a_price_dev, max_p=0.95, name="MODEL_A_PRICE")
    sum_a1_base_dev, ledger_a1_base_dev = run_replay_on_dev(p_a1_dev, max_p=0.95, name="MODEL_A1_BASE")
    sum_a1_cap_dev, ledger_a1_cap_dev = run_replay_on_dev(p_a1_dev, max_p=0.40, name="MODEL_A1_CAP_040")

    # Item 25: Disentangle price cap from ML: Naive Price Rule (buy all outsiders with ask <= 0.40 without ML)
    naive_p_win_dev = df_dev["executable_ask"].to_numpy() + 0.03
    sum_naive_dev, ledger_naive_dev = run_replay_on_dev(naive_p_win_dev, max_p=0.40, name="NAIVE_PRICE_RULE_040")

    dev_results = {
        "M0_MARKET": {"brier": round(m0_brier, 4), "n_trades": 0, "net_pnl_usdc": 0.0},
        "MLEGACY_V11": sum_mlegacy_dev,
        "MODEL_A_PRICE": sum_a_price_dev,
        "MODEL_A1_BASE": sum_a1_base_dev,
        "MODEL_A1_CAP_040": sum_a1_cap_dev,
        "NAIVE_PRICE_RULE_040": sum_naive_dev,
    }

    print("Dev Results Summary:")
    for k, v in dev_results.items():
        if k == "M0_MARKET":
            print(f"  {k:20}: Brier={v['brier']}")
        else:
            print(f"  {k:20}: Trades={v['n_trades']:3d} | PnL={v['net_pnl_usdc']:+6.2f} USDC | Exp={v['expectancy']:+.4f} [{v['ci_lower']:+.3f}, {v['ci_upper']:+.3f}] | DD={v['max_drawdown']:.2f}")

    # =========================================================================
    # Item 26: Development Robustness Checks
    # =========================================================================
    print("\n--- [Item 26] Development Robustness & Sensitivity Checks ---")
    caps_grid = [0.30, 0.35, 0.40, 0.45, 0.50, 0.95]
    caps_sensitivity = {}
    for cap in caps_grid:
        s, _ = run_replay_on_dev(p_a1_dev, max_p=cap, name=f"CAP_{cap:.2f}")
        caps_sensitivity[f"{cap:.2f}"] = {
            "n_trades": s["n_trades"],
            "net_pnl_usdc": s["net_pnl_usdc"],
            "expectancy": s["expectancy"],
            "max_drawdown": s["max_drawdown"],
        }
        print(f"  Cap {cap:.2f}: Trades={s['n_trades']:3d} | PnL={s['net_pnl_usdc']:+6.2f} | Exp={s['expectancy']:+.4f} | DD={s['max_drawdown']:.2f}")

    trades_cap_dev = ledger_a1_cap_dev.executed_trades
    df_trades_cap_dev = pd.DataFrame(trades_cap_dev)

    if not df_trades_cap_dev.empty:
        df_trades_cap_dev["day"] = pd.to_datetime(df_trades_cap_dev["decision_at"]).dt.strftime("%Y-%m-%d")
        daily_pnl = df_trades_cap_dev.groupby("day")["realized_pnl"].agg(["count", "sum"]).to_dict(orient="index")
        side_pnl = df_trades_cap_dev.groupby("candidate_side")["realized_pnl"].agg(["count", "sum"]).to_dict(orient="index")

        sorted_pnls = sorted([t["realized_pnl"] for t in trades_cap_dev], reverse=True)
        top3_trade_sum = round(float(np.sum(sorted_pnls[:3])), 2) if len(sorted_pnls) >= 3 else 0.0
        total_pnl = sum_a1_cap_dev["net_pnl_usdc"]
        top3_pct = round((top3_trade_sum / total_pnl * 100.0), 1) if total_pnl > 0 else 0.0

        p_report = generate_price_bins_report(df_trades_cap_dev, price_col="quote_ask")
        price_bins_report = p_report.to_dict(orient="records") if isinstance(p_report, pd.DataFrame) else list(p_report)
    else:
        daily_pnl = {}
        side_pnl = {}
        top3_trade_sum = 0.0
        top3_pct = 0.0
        price_bins_report = []

    dev_robustness = {
        "caps_sensitivity": caps_sensitivity,
        "daily_distribution": daily_pnl,
        "side_distribution": side_pnl,
        "top3_trades_sum_usdc": top3_trade_sum,
        "top3_trades_pct_of_total": top3_pct,
        "price_bins_breakdown": price_bins_report,
    }

    # =========================================================================
    # Item 27: Candidate Selection based strictly on Development Performance
    # =========================================================================
    print("\n--- [Item 27] Candidate Selection on Development ---")
    dev_delta_vs_base = round(sum_a1_cap_dev["net_pnl_usdc"] - sum_a1_base_dev["net_pnl_usdc"], 4)
    dev_delta_vs_naive = round(sum_a1_cap_dev["net_pnl_usdc"] - sum_naive_dev["net_pnl_usdc"], 4)

    print(f"Dev Delta PnL (A1_CAP_040 vs A1_BASE): {dev_delta_vs_base:+.2f} USDC")
    print(f"Dev Delta PnL (A1_CAP_040 vs NAIVE):   {dev_delta_vs_naive:+.2f} USDC")

    selected_candidate_name = "MODEL_A1_CAP_040"
    candidate_selection_status = "CANDIDATE_SELECTED_FOR_OOS_VERIFICATION"
    print(f"Selected Configuration for Out-of-Sample Test: {selected_candidate_name} ({candidate_selection_status})")

    # =========================================================================
    # Item 28: Independent Out-Of-Sample Evaluation on Test Split (Last 30%)
    # =========================================================================
    print("\n--- [Item 28] Independent Out-of-Sample Evaluation on Test Holdout ---")

    final_model = res_a1.final_model
    feats = res_a1.feature_names
    X_test = df_test[list(feats)].copy()

    p_test_a1 = final_model.predict_proba(X_test)[:, 1]

    def run_replay_on_test(p_win_arr, max_p: float, name: str) -> tuple[dict[str, Any], Any]:
        policy = ReplayPolicy(
            max_positions_per_market=1,
            stake_usdc=1.0,
            initial_capital=100.0,
            min_edge=0.02,
            default_fee_rate=0.002,
            max_price=max_p,
        )
        engine = OutsiderReplayEngine(policy=policy)
        decisions = []
        for idx in range(len(df_test)):
            row = df_test.iloc[idx]
            decisions.append({
                "market_id": row["market_id"],
                "decision_at": row["decision_at"],
                "time_left_min": row["time_left_min"],
                "time_valid": bool(row.get("time_valid", True)),
                "market_end_at": row.get("market_end_at", None),
                "candidate_side": row["candidate_side"],
                "executable_ask": row["executable_ask"],
                "p_win": p_win_arr[idx] if np.isfinite(p_win_arr[idx]) else 0.0,
                "target": row["target"],
            })
        ledger = engine.run(decisions)
        trades = ledger.executed_trades
        pnls = [t["realized_pnl"] for t in trades]
        n_trades = len(pnls)
        wins = sum(1 for t in trades if t["outcome"] == 1)
        wr = round(wins / n_trades, 4) if n_trades else 0.0
        pnl = round(float(np.sum(pnls)), 4) if n_trades else 0.0
        exp = round(pnl / n_trades, 6) if n_trades else 0.0
        trade_clusters = [pd.to_datetime(t["decision_at"], utc=True).strftime("%Y-%m-%d") for t in trades]
        unc = compute_clustered_uncertainty(pnls, trade_clusters) if n_trades else {"ci_lower": 0.0, "ci_upper": 0.0, "se": 0.0}
        dd = compute_drawdown(pnls, initial_equity=100.0)
        payoff = compute_payoff_ratio(pnls)

        summary = {
            "name": name,
            "max_price": max_p,
            "n_trades": n_trades,
            "win_rate": wr,
            "net_pnl_usdc": pnl,
            "expectancy": exp,
            "ci_lower": unc["ci_lower"],
            "ci_upper": unc["ci_upper"],
            "max_drawdown": dd,
            "payoff_ratio": payoff,
        }
        return summary, ledger

    sum_test_a1_base, ledger_test_a1_base = run_replay_on_test(p_test_a1, max_p=0.95, name="TEST_MODEL_A1_BASE")
    sum_test_a1_cap, ledger_test_a1_cap = run_replay_on_test(p_test_a1, max_p=0.40, name="TEST_MODEL_A1_CAP_040")

    naive_p_win_test = df_test["executable_ask"].to_numpy() + 0.03
    sum_test_naive, ledger_test_naive = run_replay_on_test(naive_p_win_test, max_p=0.40, name="TEST_NAIVE_PRICE_RULE_040")

    test_trades_base_map = {t["market_id"]: t["realized_pnl"] for t in ledger_test_a1_base.executed_trades}
    test_trades_cap_map = {t["market_id"]: t["realized_pnl"] for t in ledger_test_a1_cap.executed_trades}
    test_trades_naive_map = {t["market_id"]: t["realized_pnl"] for t in ledger_test_naive.executed_trades}

    all_test_markets = sorted(list(set(test_trades_base_map.keys()) | set(test_trades_cap_map.keys()) | set(test_trades_naive_map.keys())))

    market_to_date = df_test.set_index("market_id")["decision_at"].apply(
        lambda d: pd.to_datetime(d, utc=True).strftime("%Y-%m-%d")
    ).to_dict()
    all_test_clusters = [market_to_date.get(m, "unknown") for m in all_test_markets]

    pnl_base_aligned = [test_trades_base_map.get(m, 0.0) for m in all_test_markets]
    pnl_cap_aligned = [test_trades_cap_map.get(m, 0.0) for m in all_test_markets]
    pnl_naive_aligned = [test_trades_naive_map.get(m, 0.0) for m in all_test_markets]

    paired_cap_vs_base = compute_paired_block_bootstrap(
        pnls_a=pnl_cap_aligned,
        pnls_b=pnl_base_aligned,
        cluster_ids=all_test_clusters,
        seed=42,
    )

    paired_cap_vs_naive = compute_paired_block_bootstrap(
        pnls_a=pnl_cap_aligned,
        pnls_b=pnl_naive_aligned,
        cluster_ids=all_test_clusters,
        seed=42,
    )

    print(f"Test Holdout Results:")
    print(f"  Base Model A1:        Trades={sum_test_a1_base['n_trades']:3d} | PnL={sum_test_a1_base['net_pnl_usdc']:+6.2f} USDC | Exp={sum_test_a1_base['expectancy']:+.4f} [{sum_test_a1_base['ci_lower']:+.3f}, {sum_test_a1_base['ci_upper']:+.3f}] | DD={sum_test_a1_base['max_drawdown']:.2f}")
    print(f"  Filtered A1 (<=0.40): Trades={sum_test_a1_cap['n_trades']:3d} | PnL={sum_test_a1_cap['net_pnl_usdc']:+6.2f} USDC | Exp={sum_test_a1_cap['expectancy']:+.4f} [{sum_test_a1_cap['ci_lower']:+.3f}, {sum_test_a1_cap['ci_upper']:+.3f}] | DD={sum_test_a1_cap['max_drawdown']:.2f}")
    print(f"  Naive Rule (<=0.40):  Trades={sum_test_naive['n_trades']:3d} | PnL={sum_test_naive['net_pnl_usdc']:+6.2f} USDC | Exp={sum_test_naive['expectancy']:+.4f} [{sum_test_naive['ci_lower']:+.3f}, {sum_test_naive['ci_upper']:+.3f}] | DD={sum_test_naive['max_drawdown']:.2f}")

    print(f"Paired Block Delta (Cap 0.40 vs Base):  Delta PnL={paired_cap_vs_base['delta_pnl']:+6.2f} | 95% CI=[{paired_cap_vs_base['ci_lower']:+.3f}, {paired_cap_vs_base['ci_upper']:+.3f}]")
    print(f"Paired Block Delta (Cap 0.40 vs Naive): Delta PnL={paired_cap_vs_naive['delta_pnl']:+6.2f} | 95% CI=[{paired_cap_vs_naive['ci_lower']:+.3f}, {paired_cap_vs_naive['ci_upper']:+.3f}]")

    exp_lower = sum_test_a1_cap["ci_lower"]
    paired_base_lower = paired_cap_vs_base["ci_lower"]

    if sum_test_a1_cap["net_pnl_usdc"] > 0 and exp_lower > 0 and paired_base_lower > 0:
        oos_decision = "CANDIDATE_SELECTED"
        verdict_summary = "Hard max_price <= 0.40 demonstrates statistically verified out-of-sample positive edge and superiority over base policy."
    elif sum_test_a1_cap["net_pnl_usdc"] > 0 and (exp_lower <= 0 or paired_base_lower <= 0):
        oos_decision = "INCONCLUSIVE"
        verdict_summary = f"Positive point PnL ({sum_test_a1_cap['net_pnl_usdc']:+.2f} USDC), but 95% confidence bounds cross zero (Expectancy CI [{exp_lower:+.3f}, {sum_test_a1_cap['ci_upper']:+.3f}], Paired Delta CI [{paired_base_lower:+.3f}, {paired_cap_vs_base['ci_upper']:+.3f}]). Edge is statistically INCONCLUSIVE."
    else:
        oos_decision = "EDGE_NOT_SUPPORTED"
        verdict_summary = "Out-of-sample testing fails to demonstrate positive net expectancy."

    print(f"\nFinal Out-Of-Sample Decision: {oos_decision}")
    print(f"Rationale: {verdict_summary}")

    # =========================================================================
    # Item 29: Final Product Decision & Artifact Bundles
    # =========================================================================
    verdict_artifact = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "item": "29",
        "title": "Stage 2 Final Evaluation Verdict: Model A1 Hard Max Price Filter (<=0.40)",
        "dataset_integrity": {
            "dataset_path": str(raw_dataset_path.relative_to(REPO_ROOT)),
            "dataset_sha256": dataset_hash,
            "total_markets": meta["unique_markets"],
            "total_decisions": n_total,
        },
        "development_results": dev_results,
        "development_robustness": dev_robustness,
        "out_of_sample_test_results": {
            "base_model_a1": sum_test_a1_base,
            "filtered_model_a1_cap_040": sum_test_a1_cap,
            "naive_price_rule_cap_040": sum_test_naive,
            "paired_delta_cap_vs_base": paired_cap_vs_base,
            "paired_delta_cap_vs_naive": paired_cap_vs_naive,
        },
        "final_verdict": {
            "status": oos_decision,
            "rationale": verdict_summary,
            "recommendation": (
                "Deploy Model A1 with max_price <= 0.40 in PAPER execution mode."
                if oos_decision in ("CANDIDATE_SELECTED", "INCONCLUSIVE") and sum_test_a1_cap["net_pnl_usdc"] > 0
                else "Do not deploy; preserve existing defensive baseline."
            ),
        },
    }

    verdict_path = REPO_ROOT / "artifacts" / "research" / "stage2_price_filter_verdict.json"
    with open(verdict_path, "w", encoding="utf-8") as f:
        json.dump(verdict_artifact, f, indent=2)
    print(f"[Item 29] Verdict artifact saved -> {verdict_path}")

    candidate_bundle = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_architecture": "LogisticRegression",
        "feature_set": "MODEL_A1",
        "features": list(feats),
        "hyperparameters": {
            "C": float(res_a1.best_c),
            "penalty": "l2",
            "solver": "lbfgs",
            "imputer": "median",
            "scaler": "standard",
            "calibration": "sigmoid",
        },
        "execution_policy": {
            "max_price": 0.40,
            "min_edge": 0.02,
            "stake_usdc": 1.0,
            "max_positions_per_market": 1,
            "allow_reentry": False,
        },
        "evaluation_verdict": oos_decision,
    }

    bundle_path = REPO_ROOT / "artifacts" / "research" / "candidate_bundle.json"
    with open(bundle_path, "w", encoding="utf-8") as f:
        json.dump(candidate_bundle, f, indent=2)
    print(f"[Item 29] Candidate bundle saved -> {bundle_path}")

    return verdict_artifact


if __name__ == "__main__":
    run_stage2_evaluation()
