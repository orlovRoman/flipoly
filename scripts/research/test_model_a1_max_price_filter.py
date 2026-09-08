import json
import sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

REPO_ROOT = Path(".").resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.run_stage2_empirical_benchmark import load_real_btc_outsider_data
from polyflip.models.point_in_time_features import compute_outsider_model_features
from polyflip.models.outsider_trainer import train_outsider_model
from polyflip.research.outsider_replay import OutsiderReplayEngine, ReplayPolicy
from polyflip.research.reporting_helpers import (
    compute_drawdown,
    compute_payoff_ratio,
    compute_clustered_uncertainty,
)

df, meta = load_real_btc_outsider_data()
df_feat = compute_outsider_model_features(df)
res_a = train_outsider_model(df_feat, feature_set="MODEL_A1", validation_mode="walk_forward", n_splits=5)
p_a = res_a.oof_predictions

grid = [0.30, 0.35, 0.40, 0.45, 0.50, 0.95]
results = {}

for max_p in grid:
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
    for i in range(len(df_feat)):
        decisions.append({
            "market_id": df_feat["market_id"].iloc[i],
            "decision_at": df_feat["decision_at"].iloc[i],
            "time_left_min": df_feat["time_left_min"].iloc[i],
            "candidate_side": df_feat["candidate_side"].iloc[i],
            "executable_ask": df_feat["executable_ask"].iloc[i],
            "p_win": p_a[i] if np.isfinite(p_a[i]) else 0.0,
            "target": df_feat["target"].iloc[i],
        })
    ledger = engine.run(decisions)
    trades = ledger.executed_trades
    pnls = [t["realized_pnl"] for t in trades]
    n_trades = len(pnls)
    wins = sum(1 for t in trades if t["outcome"] == 1)
    wr = round(wins / n_trades, 4) if n_trades else 0.0
    pnl = round(float(np.sum(pnls)), 4) if n_trades else 0.0
    exp = round(pnl / n_trades, 6) if n_trades else 0.0
    unc = compute_clustered_uncertainty(pnls, [t["market_id"] for t in trades]) if n_trades else {"ci_lower": 0.0, "ci_upper": 0.0, "se": 0.0}
    dd = compute_drawdown(pnls)
    payoff = compute_payoff_ratio(pnls)
    
    results[f"max_price_{max_p:.2f}"] = {
        "max_price": max_p,
        "n_trades": n_trades,
        "win_rate": wr,
        "net_pnl_usdc": pnl,
        "expectancy": exp,
        "ci_lower": unc["ci_lower"],
        "ci_upper": unc["ci_upper"],
        "se": unc["se"],
        "max_drawdown": dd,
        "payoff_ratio": payoff,
    }

out_data = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "study": "Model A1 Hard Max Price Filter Evaluation (Variant B)",
    "dataset": "artifacts/weighted_policy/observations_30d.json",
    "total_markets": meta["unique_markets"],
    "grid_results": results,
    "delta_040_vs_base": {
        "trades_delta": results["max_price_0.40"]["n_trades"] - results["max_price_0.95"]["n_trades"],
        "pnl_delta_usdc": round(results["max_price_0.40"]["net_pnl_usdc"] - results["max_price_0.95"]["net_pnl_usdc"], 4),
        "expectancy_multiple": round(results["max_price_0.40"]["expectancy"] / results["max_price_0.95"]["expectancy"], 2),
        "max_dd_reduction": round(results["max_price_0.95"]["max_drawdown"] - results["max_price_0.40"]["max_drawdown"], 2),
    }
}

out_path = REPO_ROOT / "artifacts" / "research" / "model_a1_price_filter_study.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out_data, f, indent=2)
print("Saved study JSON ->", out_path)
