"""
scripts/research/run_stage2_regime_study.py

Executes Stage 2 research pipeline and updates reproducible manifests and artifacts.
"""
from __future__ import annotations

import sys
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from polyflip.research.stage2_regime_study import run_stage2_study


def update_manifest(study_results: dict) -> None:
    manifest_path = REPO_ROOT / "artifacts" / "research" / "reproducible_data_manifest.json"
    if not manifest_path.exists():
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Recompute sha256 of updated artifacts
    for fname in [
        "stage2_comprehensive_study_results.json",
        "common_opportunity_ledger.json",
        "three_major_wins_evidence.json",
    ]:
        p = REPO_ROOT / "artifacts" / "research" / fname
        if p.exists():
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            manifest.setdefault("manifest_files", {})[fname] = {
                "path": str(p),
                "sha256": h,
                "size_bytes": p.stat().st_size,
            }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("Updated reproducible_data_manifest.json with new artifact hashes.")


def main() -> None:
    print("=" * 80)
    print("STARTING STAGE 2: COMPREHENSIVE REGIME AND STRIKE RESEARCH STUDY")
    print("=" * 80)

    snaps_csv = REPO_ROOT / "artifacts" / "research" / "btc_snapshots_4_15m.csv"
    candles_csv = REPO_ROOT / "artifacts" / "research" / "crypto_candles_5m.csv"
    exp_json = REPO_ROOT / "artifacts" / "research" / "market_expirations.json"

    res = run_stage2_study(
        snapshots_csv_path=snaps_csv,
        candles_csv_path=candles_csv,
        expirations_json_path=exp_json,
        target_asset="BTC",
        stake_usdc=1.0,
        taker_fee_rate=0.002,
        seed=42,
        n_bootstrap=1000,
    )

    update_manifest(res)

    print("\n" + "=" * 80)
    print("STAGE 2 CORE RESULTS SUMMARY")
    print("=" * 80)
    variants = res["item_25_and_28_core_variants"]
    print(f"{'Variant':<26s} | {'Trades':<6s} | {'WinRate':<7s} | {'Net PnL':<10s} | {'Exp':<8s} | {'Delta vs C0 (95% CI)':<25s}")
    print("-" * 90)
    for k, v in variants.items():
        ci = v.get("paired_bootstrap_ci_95", [0, 0])
        delta = v.get("paired_delta_vs_c0", 0.0)
        ci_str = f"[{ci[0]:+7.2f}, {ci[1]:+7.2f}]" if k != "C0" else "baseline"
        delta_str = f"{delta:+7.2f}" if k != "C0" else "  0.00"
        print(f"{v['label']:<26s} | {v['n_trades']:<6d} | {v['win_rate']*100:<6.2f}% | {v['net_pnl_usdc']:+9.2f}  | {v['expectancy']:+7.4f} | {delta_str} {ci_str}")

    print("\n" + "=" * 80)
    print("FORMER FAVORITE MECHANISM BREAKDOWN (Items 19 & 20)")
    print("=" * 80)
    ff = res["item_19_and_20_former_favorite_mechanism"]
    print(f"{'Group':<22s} | {'Trades':<6s} | {'WinRate':<7s} | {'Net PnL':<10s} | {'Expectancy':<10s}")
    print("-" * 65)
    for grp, st in ff.items():
        print(f"{grp:<22s} | {st['n_trades']:<6d} | {st['win_rate']*100:<6.2f}% | {st['net_pnl_usdc']:+9.2f}  | {st['expectancy']:+8.4f}")

    print("\n" + "=" * 80)
    print("COHORT DECOMPOSITION (Item 22)")
    print("=" * 80)
    ch = res["item_22_cohort_decomposition"]
    print(f"{'Cohort':<12s} | {'Trades':<6s} | {'WinRate':<7s} | {'AvgPrice':<8s} | {'Net PnL':<10s} | {'Expectancy':<10s}")
    print("-" * 65)
    for c_name, st in ch.items():
        print(f"{c_name:<12s} | {st['n_trades']:<6d} | {st['win_rate']*100:<6.2f}% | {st['avg_entry_price']:<8.4f} | {st['net_pnl_usdc']:+9.2f}  | {st['expectancy']:+8.4f}")

    print("\n" + "=" * 80)
    print("MULTI-BUDGET EXECUTION SIMULATION (Item 26)")
    print("=" * 80)
    mb = res["item_26_multi_budget_execution"]
    print(f"{'Budget':<8s} | {'Fills (Full/Part/Unfill)':<25s} | {'Spent USDC':<11s} | {'VWAP Drift':<10s} | {'Net PnL':<10s} | {'ROIC %':<8s}")
    print("-" * 80)
    for b_lbl, b_data in mb.items():
        fills_str = f"{b_data['full_fills']}/{b_data['partial_fills']}/{b_data['unfilled']}"
        print(f"{b_lbl:<8s} | {fills_str:<25s} | {b_data['total_spent_usdc']:<11.2f} | {b_data['vwap_drift_vs_decision_price']:+9.4f} | {b_data['net_pnl_total']:+9.2f}  | {b_data['return_on_spent_pct']:+6.2f}%")

    print("\n" + "=" * 80)
    print(f"VERDICT: {res['item_30_decision_verdict']['verdict_status']}")
    print(f"ACTION : {res['item_30_decision_verdict']['action_required']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
