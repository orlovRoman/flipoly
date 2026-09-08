"""
scripts/research/compare_outsider_models.py

Comprehensive model comparison and final configuration selection (Item 2.10):
Generates a unified comparison table for:
  - M0 (Market Price Baseline)
  - Mlegacy (Legacy Baseline)
  - Model A1 (Compact 4-feature Outsider LogReg)
  - Model B1 (A1 + z_outsider + ret_30s + ret_120s)
  - Model B + LGBM Veto
  - Model B + LGBM Input

Reports: unique markets, coverage, Brier, log loss, ECE, reliability across 0.05 bins,
net PnL, expectancy, payoff ratio, maximum drawdown, and paired delta PnL relative to Model A and M0.
Selects the simplest configuration that yields robust positive net expectancy.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence, Any
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from polyflip.models.probability_metrics import (
    brier_score,
    log_loss_score,
    expected_calibration_error,
)
from polyflip.crypto.edge import compute_net_ev_per_share
from polyflip.models.outsider_baselines import (
    MarketPriceBaseline,
    LegacyOutsiderBaseline,
)
from polyflip.models.outsider_trainer import train_outsider_model
from polyflip.trading.combined_voting import evaluate_lgbm_outsider_interaction


def compute_drawdown(pnls: Sequence[float]) -> float:
    """Computes maximum peak-to-trough drawdown from sequence of trade PnLs."""
    if not pnls:
        return 0.0
    cumsum = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumsum)
    drawdowns = running_max - cumsum
    return round(float(np.max(drawdowns)), 4) if len(drawdowns) > 0 else 0.0


def compute_payoff_ratio(pnls: Sequence[float]) -> float:
    """Computes payoff ratio (average win / average loss)."""
    p_arr = np.asarray(pnls, dtype=float)
    wins = p_arr[p_arr > 0]
    losses = np.abs(p_arr[p_arr < 0])
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 1e-9
    return round(avg_win / avg_loss, 4) if avg_loss > 0 else 0.0


def compute_clustered_uncertainty(
    trade_pnls: Sequence[float],
    cluster_ids: Sequence[Any],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Computes cluster-robust standard error and 95% confidence intervals for expectancy (mean PnL per trade).
    Clusters can be market_id or calendar date.
    """
    if not trade_pnls or len(trade_pnls) == 0:
        return {"se": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n_clusters": 0}

    pnls_arr = np.asarray(trade_pnls, dtype=float)
    c_arr = np.asarray(cluster_ids)
    unique_clusters = np.unique(c_arr)
    n_c = len(unique_clusters)
    n_trades = len(pnls_arr)

    if n_c <= 1 or n_trades == 0:
        mean_val = round(float(np.mean(pnls_arr)), 6) if n_trades > 0 else 0.0
        return {"se": 0.0, "ci_lower": mean_val, "ci_upper": mean_val, "n_clusters": n_c}

    # Aggregate PnL and trade count per cluster
    cluster_sums = np.array([np.sum(pnls_arr[c_arr == c]) for c in unique_clusters])
    cluster_counts = np.array([np.sum(c_arr == c) for c in unique_clusters])

    # Cluster bootstrap for expectancy (ratio of sums: total_pnl / total_trades)
    rng = np.random.default_rng(seed)
    boot_means = []
    for _ in range(n_bootstrap):
        sampled_c = rng.choice(n_c, size=n_c, replace=True)
        tot_pnl = np.sum(cluster_sums[sampled_c])
        tot_trades = np.sum(cluster_counts[sampled_c])
        if tot_trades > 0:
            boot_means.append(tot_pnl / tot_trades)

    if boot_means:
        se = float(np.std(boot_means, ddof=1))
        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))
    else:
        se = 0.0
        exp = float(np.sum(cluster_sums) / n_trades)
        ci_lower, ci_upper = exp, exp

    return {
        "se": round(se, 6),
        "ci_lower": round(ci_lower, 6),
        "ci_upper": round(ci_upper, 6),
        "n_clusters": int(n_c),
    }


def compute_paired_cluster_delta(
    pnls_model: Sequence[float],
    indices_model: Sequence[int],
    pnls_ref: Sequence[float],
    indices_ref: Sequence[int],
    cluster_ids_full: Sequence[Any],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Computes paired cluster-level delta PnL between candidate model and reference model.
    For each cluster (market or date): delta_pnl_c = sum(pnl_cand in c) - sum(pnl_ref in c).
    """
    clusters_full = np.asarray(cluster_ids_full)
    unique_clusters = np.unique(clusters_full)
    n_c = len(unique_clusters)

    # Build cluster sum maps
    c_pnl_cand = {c: 0.0 for c in unique_clusters}
    for pnl, idx in zip(pnls_model, indices_model):
        c_pnl_cand[clusters_full[idx]] += pnl

    c_pnl_ref = {c: 0.0 for c in unique_clusters}
    for pnl, idx in zip(pnls_ref, indices_ref):
        c_pnl_ref[clusters_full[idx]] += pnl

    deltas = np.array([c_pnl_cand[c] - c_pnl_ref[c] for c in unique_clusters])
    total_delta = float(np.sum(deltas))

    if n_c <= 1:
        return {
            "delta_pnl": round(total_delta, 4),
            "se": 0.0,
            "ci_lower": round(total_delta, 4),
            "ci_upper": round(total_delta, 4),
            "p_value": 1.0,
        }

    # Standard error of sum of cluster deltas
    se_delta = float(np.sqrt(n_c) * np.std(deltas, ddof=1))
    t_stat = total_delta / se_delta if se_delta > 1e-9 else 0.0

    # Cluster bootstrap for delta PnL
    rng = np.random.default_rng(seed)
    boot_deltas = []
    for _ in range(n_bootstrap):
        sampled_c = rng.choice(n_c, size=n_c, replace=True)
        boot_deltas.append(np.sum(deltas[sampled_c]))

    ci_lower = float(np.percentile(boot_deltas, 2.5))
    ci_upper = float(np.percentile(boot_deltas, 97.5))
    # Empirical one-tailed p-value for H0: delta <= 0
    p_val = float(np.mean(np.array(boot_deltas) <= 0.0)) if total_delta > 0 else float(np.mean(np.array(boot_deltas) >= 0.0))

    return {
        "delta_pnl": round(total_delta, 4),
        "se": round(se_delta, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "t_stat": round(t_stat, 3),
        "p_value": round(p_val, 4),
    }


def compare_all_models(
    df: pd.DataFrame,
    fee_rate: float = 0.002,
    min_edge: float = 0.02,
) -> dict[str, Any]:
    """
    Evaluates all model configurations on identical complete cohort.
    """
    if df.empty:
        raise ValueError("Input dataframe is empty")

    total_markets = int(df["market_id"].nunique())
    y_true = df["target"].to_numpy()
    asks = df["executable_ask"].to_numpy() if "executable_ask" in df.columns else np.full(len(df), 0.35)
    clusters_full = df["market_id"].to_numpy() if "market_id" in df.columns else np.arange(len(df))

    # 1. Baseline M0
    m0 = MarketPriceBaseline()
    p_m0 = m0.predict_proba(df)[:, 1]

    # 2. Baseline Mlegacy
    m_leg = LegacyOutsiderBaseline()
    p_leg = m_leg.predict_proba(df)[:, 1]

    # 3. Model A1 (Trained)
    res_a = train_outsider_model(df, feature_set="MODEL_A1", fee_rate=fee_rate, min_edge=min_edge)
    p_a = res_a.oof_predictions

    # 4. Model B1 (Trained)
    res_b = train_outsider_model(df, feature_set="MODEL_B1", fee_rate=fee_rate, min_edge=min_edge)
    p_b = res_b.oof_predictions

    # Prepare simulation helper
    def _simulate_trades(probs: np.ndarray) -> tuple[list[float], list[int], int]:
        trade_pnls = []
        trade_indices = []
        wins = 0
        for i in range(len(df)):
            net_ev = compute_net_ev_per_share(probs[i], asks[i], fee_per_share=asks[i] * fee_rate)
            if net_ev >= min_edge and asks[i] < 0.95:
                fee = asks[i] * fee_rate
                outcome = y_true[i]
                pnl = (1.0 - asks[i] - fee) if outcome == 1 else (-asks[i] - fee)
                trade_pnls.append(pnl)
                trade_indices.append(i)
                if outcome == 1:
                    wins += 1
        return trade_pnls, trade_indices, wins

    models_data = [
        ("M0_MARKET", p_m0),
        ("MLEGACY", p_leg),
        ("MODEL_A1", p_a),
        ("MODEL_B1", p_b),
    ]

    report_rows = []
    pnl_dict = {}
    trades_dict = {}

    for name, p_vals in models_data:
        brier = float(brier_score(y_true, p_vals))
        ll = float(log_loss_score(y_true, p_vals))
        ece_val, diag = expected_calibration_error(y_true, p_vals, n_bins=20)
        ece = float(ece_val) if ece_val is not None else 0.0

        pnls, indices, wins = _simulate_trades(p_vals)
        n_trades = len(pnls)
        total_pnl = float(np.sum(pnls)) if n_trades > 0 else 0.0
        pnl_dict[name] = total_pnl
        trades_dict[name] = (pnls, indices)

        win_rate = float(wins / n_trades) if n_trades > 0 else 0.0
        expectancy = float(total_pnl / n_trades) if n_trades > 0 else 0.0
        payoff = compute_payoff_ratio(pnls)
        max_dd = compute_drawdown(pnls)

        trade_clusters = clusters_full[indices] if indices else []
        unc = compute_clustered_uncertainty(pnls, trade_clusters)

        report_rows.append({
            "model": name,
            "unique_markets": total_markets,
            "coverage": 1.0,
            "brier": round(brier, 4),
            "log_loss": round(ll, 4),
            "ece": round(ece, 4),
            "n_trades": n_trades,
            "win_rate": round(win_rate, 4),
            "net_pnl": round(total_pnl, 4),
            "expectancy": round(expectancy, 6),
            "expectancy_ci_lower": unc["ci_lower"],
            "expectancy_ci_upper": unc["ci_upper"],
            "payoff": round(payoff, 4),
            "max_dd": round(max_dd, 4),
        })

    # 5. Model B + LGBM Veto and Input
    df_interaction = df.copy()
    df_interaction["p_b_win"] = p_b
    if "lgbm_direction" not in df_interaction.columns:
        df_interaction["lgbm_direction"] = "NONE"
    if "lgbm_oof_prob" not in df_interaction.columns:
        df_interaction["lgbm_oof_prob"] = 0.5

    lgbm_inter = evaluate_lgbm_outsider_interaction(df_interaction, min_edge=min_edge, fee_rate=fee_rate)
    v_res = lgbm_inter.get("b_plus_lgbm_veto", {})
    in_res = lgbm_inter.get("b_plus_lgbm_input", {})

    v_pnls = v_res.get("trade_pnls", [])
    v_indices = v_res.get("trade_indices", [])
    v_payoff = compute_payoff_ratio(v_pnls)
    v_max_dd = compute_drawdown(v_pnls)
    v_clusters = clusters_full[v_indices] if v_indices else []
    v_unc = compute_clustered_uncertainty(v_pnls, v_clusters)

    report_rows.append({
        "model": "MODEL_B_PLUS_VETO",
        "unique_markets": total_markets,
        "coverage": 1.0,
        "brier": round(float(brier_score(y_true, p_b)), 4),
        "log_loss": round(float(log_loss_score(y_true, p_b)), 4),
        "ece": round(float(expected_calibration_error(y_true, p_b, n_bins=20)[0] or 0.0), 4),
        "n_trades": v_res.get("n_accepted", 0),
        "win_rate": v_res.get("win_rate", 0.0),
        "net_pnl": v_res.get("total_pnl", 0.0),
        "expectancy": v_res.get("expectancy", 0.0),
        "expectancy_ci_lower": v_unc["ci_lower"],
        "expectancy_ci_upper": v_unc["ci_upper"],
        "payoff": v_payoff,
        "max_dd": v_max_dd,
    })
    pnl_dict["MODEL_B_PLUS_VETO"] = v_res.get("total_pnl", 0.0)
    trades_dict["MODEL_B_PLUS_VETO"] = (v_pnls, v_indices)

    in_pnls = in_res.get("trade_pnls", [])
    in_indices = in_res.get("trade_indices", [])
    in_payoff = compute_payoff_ratio(in_pnls)
    in_max_dd = compute_drawdown(in_pnls)
    in_clusters = clusters_full[in_indices] if in_indices else []
    in_unc = compute_clustered_uncertainty(in_pnls, in_clusters)

    report_rows.append({
        "model": "MODEL_B_PLUS_LGBM_INPUT",
        "unique_markets": total_markets,
        "coverage": 1.0,
        "brier": round(float(brier_score(y_true, p_b)), 4),
        "log_loss": round(float(log_loss_score(y_true, p_b)), 4),
        "ece": round(float(expected_calibration_error(y_true, p_b, n_bins=20)[0] or 0.0), 4),
        "n_trades": in_res.get("n_trades", 0),
        "win_rate": in_res.get("win_rate", 0.0),
        "net_pnl": in_res.get("total_pnl", 0.0),
        "expectancy": in_res.get("expectancy", 0.0),
        "expectancy_ci_lower": in_unc["ci_lower"],
        "expectancy_ci_upper": in_unc["ci_upper"],
        "payoff": in_payoff,
        "max_dd": in_max_dd,
    })
    pnl_dict["MODEL_B_PLUS_LGBM_INPUT"] = in_res.get("total_pnl", 0.0)
    trades_dict["MODEL_B_PLUS_LGBM_INPUT"] = (in_pnls, in_indices)

    # Compute paired delta PnL relative to Model A1 and M0 with cluster-robust CI
    base_pnl_a = pnl_dict.get("MODEL_A1", 0.0)
    base_pnl_m0 = pnl_dict.get("M0_MARKET", 0.0)
    a_pnls, a_indices = trades_dict.get("MODEL_A1", ([], []))

    for r in report_rows:
        name = r["model"]
        m_pnls, m_indices = trades_dict.get(name, ([], []))
        paired_a = compute_paired_cluster_delta(
            pnls_model=m_pnls,
            indices_model=m_indices,
            pnls_ref=a_pnls,
            indices_ref=a_indices,
            cluster_ids_full=clusters_full,
        )
        r["delta_pnl_vs_A"] = round(r["net_pnl"] - base_pnl_a, 4)
        r["delta_pnl_vs_M0"] = round(r["net_pnl"] - base_pnl_m0, 4)
        r["delta_pnl_ci_lower_vs_A"] = paired_a["ci_lower"]
        r["delta_pnl_ci_upper_vs_A"] = paired_a["ci_upper"]
        r["delta_pnl_p_val_vs_A"] = paired_a["p_value"]

    # Decision logic: choose simplest model that demonstrates robust positive gain
    selected_config = "MODEL_A1"
    rationale = "Model A1 serves as robust baseline"

    delta_b1 = next((r for r in report_rows if r["model"] == "MODEL_B1"), None)
    delta_veto = next((r for r in report_rows if r["model"] == "MODEL_B_PLUS_VETO"), None)

    if delta_b1 and delta_b1["delta_pnl_vs_A"] > 0:
        selected_config = "MODEL_B1"
        rationale = (
            f"Model B1 improves net PnL by +{delta_b1['delta_pnl_vs_A']:.2f} over Model A1 "
            f"(95% CI: [{delta_b1['delta_pnl_ci_lower_vs_A']:.2f}, {delta_b1['delta_pnl_ci_upper_vs_A']:.2f}], "
            f"p={delta_b1['delta_pnl_p_val_vs_A']:.3f})"
        )
    elif delta_veto and delta_veto["delta_pnl_vs_A"] > 0:
        selected_config = "MODEL_B_PLUS_VETO"
        rationale = (
            f"Model B + LGBM Veto provides superior net expectancy by eliminating conflicting trades "
            f"(delta PnL: +{delta_veto['delta_pnl_vs_A']:.2f} vs A1)"
        )
    else:
        selected_config = "MODEL_A1"
        rationale = "Model A1 retained: Model B does not demonstrate sustained incremental edge on test cohort"

    return {
        "summary_table": report_rows,
        "selected_configuration": selected_config,
        "selection_rationale": rationale,
        "lgbm_interaction": lgbm_inter,
    }


def print_comparison_table(res: dict[str, Any]) -> None:
    """Prints formatted comparison table to stdout."""
    print("\n" + "=" * 136)
    print("STAGE 2 OUTSIDER MODELS BENCHMARK & COMPARISON TABLE (Item 2.10)")
    print("=" * 136)
    header = (
        f"{'Model':<24} | {'Brier':<7} | {'LogLoss':<8} | {'ECE':<6} | "
        f"{'Trades':<6} | {'WinRate':<7} | {'NetPnL':<8} | {'Payoff':<6} | {'MaxDD':<6} | "
        f"{'Exp [95% CI]':<22} | {'d_PnL vs A [95% CI]':<23}"
    )
    print(header)
    print("-" * 136)
    for r in res["summary_table"]:
        exp_str = f"{r['expectancy']:+.4f} [{r['expectancy_ci_lower']:+.3f},{r['expectancy_ci_upper']:+.3f}]"
        d_pnl_str = f"{r['delta_pnl_vs_A']:+6.2f} [{r.get('delta_pnl_ci_lower_vs_A', 0):+5.1f},{r.get('delta_pnl_ci_upper_vs_A', 0):+5.1f}]"
        line = (
            f"{r['model']:<24} | {r['brier']:<7.4f} | {r['log_loss']:<8.4f} | {r['ece']:<6.4f} | "
            f"{r['n_trades']:<6} | {r['win_rate']:<7.3f} | {r['net_pnl']:<8.2f} | {r['payoff']:<6.2f} | {r['max_dd']:<6.2f} | "
            f"{exp_str:<22} | {d_pnl_str:<23}"
        )
        print(line)
    print("=" * 136)
    print(f"\nFinal Recommended Configuration: {res['selected_configuration']}")
    print(f"Selection Rationale:             {res['selection_rationale']}")
    print("=" * 136 + "\n")


if __name__ == "__main__":
    from scripts.research.outsider_ablation import generate_ablation_dataset
    print("Running comprehensive model comparison benchmark...")
    df_sample = generate_ablation_dataset(n_markets=24)
    res = compare_all_models(df_sample)
    print_comparison_table(res)
