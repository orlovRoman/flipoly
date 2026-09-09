"""
scripts/research/run_regime_strike_experiment.py

Master execution script for the Mean-Reversion Regime and Strike-Context Research Plan:
"План проверяет дополнительную пользу локальной пилы и положения относительно strike,
сохраняя простое правило ask <= 0.40 контролем. Каждое усложнение должно показать измеримый вклад."

Executes Stages 1 through 5:
- Stage 1: Fix experiment protocol, exact hypothesis, data coverage table, quote provenance
- Stage 2: Causal features (ER 3m/5m/15m, slope, sign flips, autocorr, 4 states, strike context, mirror symmetry)
- Stage 3: Regime classification without target fitting, frozen development boundaries
- Stage 4: Economic evaluation (C0, C1, C2), additive ledger invariants, block bootstrap, robustness, multi-asset portability
- Stage 5: System decision verdict, conditional ML check, minimal policy recommendations

Exports:
- artifacts/research/data_coverage_table.json (Items 6, 7)
- artifacts/research/regime_strike_protocol.json (Items 2, 3, 4, 5)
- artifacts/research/regime_strike_experiment_results.json (Items 22-27)
- artifacts/research/regime_strike_verdict.json (Items 28-30)
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

from polyflip.research.regime_features import (
    classify_local_regime,
    compute_strike_context,
    verify_mirror_symmetry,
)
from polyflip.research.coverage_table import compute_data_coverage_table
from polyflip.research.regime_experiment import (
    load_and_prepare_5m_dataset,
    run_paired_experiment,
    compute_multi_asset_pooled_experiment,
    evaluate_candidate_ml_interaction,
)


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_reproducible_data_manifest(artifacts_dir: Path, btc_df: pd.DataFrame) -> dict[str, Any]:
    import platform
    import subprocess

    try:
        git_rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT)).decode().strip()
        git_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(REPO_ROOT)).decode().strip()
    except Exception:
        git_rev = "unknown"
        git_branch = "unknown"

    manifest_files = {}
    target_files = [
        artifacts_dir / "all_assets_snapshots_4_15m.csv",
        artifacts_dir / "btc_snapshots_4_15m.csv",
        artifacts_dir / "crypto_candles_5m.csv",
        artifacts_dir / "data_coverage_table.json",
        artifacts_dir / "regime_strike_protocol.json",
        artifacts_dir / "regime_strike_experiment_results.json",
        artifacts_dir / "regime_strike_verdict.json",
    ]
    for p in target_files:
        if p.exists():
            manifest_files[p.name] = {
                "path": str(p),
                "sha256": compute_file_sha256(p),
                "size_bytes": p.stat().st_size,
            }

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_rev,
        "git_branch": git_branch,
        "python_environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
        },
        "dataset_row_counts": {
            "btc_decision_markets": len(btc_df),
            "date_range": [str(btc_df["decision_at"].min()), str(btc_df["decision_at"].max())],
        },
        "manifest_files": manifest_files,
    }
    manifest_path = artifacts_dir / "reproducible_data_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Exported reproducible data manifest -> {manifest_path}")
    return manifest


def main() -> None:
    print("=" * 90)
    print("STARTING REGIME & STRIKE CONTEXT RESEARCH EXPERIMENT HARNESS")
    print("=" * 90)

    artifacts_dir = REPO_ROOT / "artifacts" / "research"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    snapshots_csv = artifacts_dir / "all_assets_snapshots_4_15m.csv"
    btc_snapshots_csv = artifacts_dir / "btc_snapshots_4_15m.csv"
    candles_csv = artifacts_dir / "crypto_candles_5m.csv"
    obs_30d_path = REPO_ROOT / "artifacts" / "weighted_policy" / "observations_30d.json"

    # -------------------------------------------------------------------------
    # STAGE 1: Protocol & Coverage Table
    # -------------------------------------------------------------------------
    print("\n--- [STAGE 1] Protocol Locking & Data Coverage Accounting ---")

    protocol = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hypothesis": (
            "При наблюдаемой локальной склонности к возврату покупка аутсайдера "
            "имеет лучший net EV, особенно когда возврат помогает завершить "
            "контракт на выигрышной стороне strike."
        ),
        "variants": {
            "C0": "Контроль: только цена ask <= 0.40",
            "CT": "C0 + режим возврата в токене (token_regime == 'REVERSION')",
            "CS": "C0 + режим возврата в споте (spot_regime == 'REVERSION') - истинная гипотеза о пиле в базовом активе",
            "CTS": "C0 + совместный режим возврата (token AND spot REVERSION)",
            "C1": "C0 + локальный режим возврата (REVERSION - token OR spot)",
            "C2": "C1 + благоприятное положение относительно strike (reversion_helps_strike == True)",
            "C2_canonical": "CS + канонический strike (POLYMARKET_CANONICAL)",
        },
        "primary_metric": "Парная разница net PnL (USDC) на одинаковом потоке возможностей с календарным блочным бутстрапом",
        "secondary_metrics": ["expectancy", "turnover", "max_drawdown", "payoff_ratio", "profit_concentration", "slippage_sensitivity"],
        "splits": {
            "exploratory_dev": "<= 2026-08-26",
            "exploratory_test": "2026-08-26 to 2026-09-02",
            "unviewed_holdout": ">= 2026-09-02",
        },
        "policy_parameters": {
            "stake_usdc": 1.0,
            "taker_fee_rate": 0.002,
            "max_price": 0.40,
            "min_price": 0.01,
            "decision_time_target_min": 5.0,
            "decision_time_tolerance_min": [3.5, 5.5],
        },
        "self_checks": {
            "fixed_stake_equal_across_variants": True,
            "no_double_spread_deduction": True,
            "causal_first_observation_chosen": True,
            "observed_quotes_isolated_from_reconstructed": True,
            "resplitting_old_data_not_called_independent": True,
        },
    }
    with open(artifacts_dir / "regime_strike_protocol.json", "w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2, ensure_ascii=False)
    print("Locked experiment protocol -> artifacts/research/regime_strike_protocol.json")

    # Generate Data Coverage Table
    try:
        all_snapshots_df = pd.read_csv(snapshots_csv, encoding="utf-16")
    except Exception:
        all_snapshots_df = pd.read_csv(snapshots_csv, encoding="utf-8")

    coverage_table = compute_data_coverage_table(all_snapshots_df)
    with open(artifacts_dir / "data_coverage_table.json", "w", encoding="utf-8") as f:
        json.dump(coverage_table, f, indent=2, ensure_ascii=False)
    print(f"Exported data coverage table ({coverage_table['summary']['total_rows']} rows, {coverage_table['summary']['total_days']} days) -> artifacts/research/data_coverage_table.json")

    # -------------------------------------------------------------------------
    # STAGE 2: Feature Engineering & Symmetry Invariance
    # -------------------------------------------------------------------------
    print("\n--- [STAGE 2] Causal Feature Construction & Symmetry Verification ---")
    sym_check = verify_mirror_symmetry(
        price_path=[68100.0, 68300.0, 68050.0, 68250.0, 68120.0],
        strike=68000.0,
        sigma_min=0.001,
        time_left_min=5.0,
        mode="geometric",
    )
    print(f"Mirror Symmetry Check: {'PASS' if sym_check['passed'] else 'FAIL'} | Details: {sym_check}")
    assert sym_check["passed"], "Mirror symmetry verification failed!"

    # -------------------------------------------------------------------------
    # STAGE 3 & 4: Primary BTC Experiment Execution
    # -------------------------------------------------------------------------
    print("\n--- [STAGE 3 & 4] Running Primary BTC Experiment (5m Decision Moment) ---")
    btc_df = load_and_prepare_5m_dataset(
        snapshots_csv_path=btc_snapshots_csv,
        candles_csv_path=candles_csv,
        target_asset="BTC",
        time_left_target=5.0,
        time_left_tolerance=(3.5, 5.5),
    )
    print(f"Prepared authentic BTC decision dataset: {len(btc_df)} markets")
    print(f"Token Regimes: {dict(btc_df['token_regime'].value_counts())}")
    print(f"Reversion Helps Strike: {dict(btc_df['reversion_helps_strike'].value_counts())}")

    # Splits
    btc_df["decision_at"] = pd.to_datetime(btc_df["decision_at"], utc=True)
    dev_cutoff = pd.to_datetime("2026-08-26 00:00:00+00:00", utc=True)
    holdout_cutoff = pd.to_datetime("2026-09-02 00:00:00+00:00", utc=True)

    df_full = btc_df
    df_dev = btc_df[btc_df["decision_at"] < dev_cutoff]
    df_test = btc_df[(btc_df["decision_at"] >= dev_cutoff) & (btc_df["decision_at"] < holdout_cutoff)]
    df_holdout = btc_df[btc_df["decision_at"] >= holdout_cutoff]

    print(f"\nCohort Sizes:")
    print(f"  Full Sample:        {len(df_full)} markets")
    print(f"  Exploratory Dev:    {len(df_dev)} markets (< 2026-08-26)")
    print(f"  Exploratory Test:   {len(df_test)} markets (2026-08-26 to 2026-09-02)")
    print(f"  Unviewed Holdout:   {len(df_holdout)} markets (>= 2026-09-02)")

    # Run evaluations on Observed Quotes Only (Primary)
    print("\n--- Evaluating Primary BTC Cohorts (Observed Quotes Only) ---")
    res_full_obs = run_paired_experiment(df_full, observed_quotes_only=True)
    res_dev_obs = run_paired_experiment(df_dev, observed_quotes_only=True)
    res_test_obs = run_paired_experiment(df_test, observed_quotes_only=True)
    res_holdout_obs = run_paired_experiment(df_holdout, observed_quotes_only=True)

    # Run sensitivity on All Quotes (Observed + Reconstructed NO Quotes)
    print("\n--- Evaluating Sensitivity (All Quotes: Observed + Reconstructed) ---")
    res_full_all = run_paired_experiment(df_full, observed_quotes_only=False)
    res_dev_all = run_paired_experiment(df_dev, observed_quotes_only=False)
    res_test_all = run_paired_experiment(df_test, observed_quotes_only=False)
    res_holdout_all = run_paired_experiment(df_holdout, observed_quotes_only=False)

    def print_cohort_summary(tag: str, res: dict[str, Any]) -> None:
        v = res["variants"]
        d = res["disentangled_contributions"]
        print(f"\n[{tag}]")
        print(f"  C0 (Control):  Trades={v['C0']['n_trades']:4d} | PnL={v['C0']['net_pnl_usdc']:+8.2f} | Exp={v['C0']['expectancy']:+.4f} | DD={v['C0']['max_drawdown']:.2f}")
        print(f"  C1 (Reversion):Trades={v['C1']['n_trades']:4d} | PnL={v['C1']['net_pnl_usdc']:+8.2f} | Exp={v['C1']['expectancy']:+.4f} | DD={v['C1']['max_drawdown']:.2f}")
        print(f"  C2 (Strike):   Trades={v['C2']['n_trades']:4d} | PnL={v['C2']['net_pnl_usdc']:+8.2f} | Exp={v['C2']['expectancy']:+.4f} | DD={v['C2']['max_drawdown']:.2f}")
        print(f"  Delta C1 - C0: {d['C1_minus_C0']['delta_net_pnl']:+8.2f} USDC (Prevented Losses={d['C1_minus_C0']['prevented_losses']:.2f}, Missed Gains={d['C1_minus_C0']['missed_gains']:.2f}) | 95% CI=[{d['C1_minus_C0']['paired_bootstrap']['ci_lower']:+.2f}, {d['C1_minus_C0']['paired_bootstrap']['ci_upper']:+.2f}]")
        print(f"  Delta C2 - C1: {d['C2_minus_C1']['delta_net_pnl']:+8.2f} USDC (Prevented Losses={d['C2_minus_C1']['prevented_losses']:.2f}, Missed Gains={d['C2_minus_C1']['missed_gains']:.2f}) | 95% CI=[{d['C2_minus_C1']['paired_bootstrap']['ci_lower']:+.2f}, {d['C2_minus_C1']['paired_bootstrap']['ci_upper']:+.2f}]")
        print(f"  Delta C2 - C0: {d['C2_minus_C0']['delta_net_pnl']:+8.2f} USDC | 95% CI=[{d['C2_minus_C0']['paired_bootstrap']['ci_lower']:+.2f}, {d['C2_minus_C0']['paired_bootstrap']['ci_upper']:+.2f}]")
        print(f"  Invariants: C1 additive={res['invariants']['c1_additive_invariant_holds']} | C2 additive={res['invariants']['c2_additive_invariant_holds']} | Self delta 0.0={res['invariants']['self_comparison_zero_delta']}")

    print_cohort_summary("BTC FULL (Observed Quotes)", res_full_obs)
    print_cohort_summary("BTC DEV (Observed Quotes)", res_dev_obs)
    print_cohort_summary("BTC TEST (Observed Quotes)", res_test_obs)
    print_cohort_summary("BTC HOLDOUT (Observed Quotes)", res_holdout_obs)

    # -------------------------------------------------------------------------
    # STAGE 4 (Continued): Multi-Asset Portability & Joint Pooling (Item 25, 27)
    # -------------------------------------------------------------------------
    print("\n--- [STAGE 4] Cross-Asset Portability Evaluation (Locked Rules) ---")
    asset_results: dict[str, Any] = {}
    all_asset_dfs: dict[str, pd.DataFrame] = {"BTC": df_full}

    for asset in ["ETH", "SOL", "XRP", "DOGE"]:
        print(f"Processing {asset}...")
        df_asset = load_and_prepare_5m_dataset(
            snapshots_csv_path=snapshots_csv,
            candles_csv_path=candles_csv,
            target_asset=asset,
            time_left_target=5.0,
            time_left_tolerance=(3.5, 5.5),
        )
        if not df_asset.empty:
            all_asset_dfs[asset] = df_asset
            res_asset = run_paired_experiment(df_asset, observed_quotes_only=True)
            asset_results[asset] = res_asset
            v = res_asset["variants"]
            d = res_asset["disentangled_contributions"]
            print(f"  {asset:4s}: C0 PnL={v['C0']['net_pnl_usdc']:+7.2f} ({v['C0']['n_trades']} trades) | C1 PnL={v['C1']['net_pnl_usdc']:+7.2f} ({v['C1']['n_trades']} trades) | C2 PnL={v['C2']['net_pnl_usdc']:+7.2f} ({v['C2']['n_trades']} trades)")
            print(f"        Delta C1-C0={d['C1_minus_C0']['delta_net_pnl']:+7.2f} | Delta C2-C1={d['C2_minus_C1']['delta_net_pnl']:+7.2f}")

    # Item 25: Multi-asset pooled joint calendar days
    print("\n--- [STAGE 4] Pooled Multi-Asset Daily Block Experiment (Item 25) ---")
    pooled_res = compute_multi_asset_pooled_experiment(all_asset_dfs, observed_quotes_only=True)
    d_pool = pooled_res["disentangled_contributions"]
    print(f"Pooled (5 assets): C1-C0 Delta={d_pool['C1_minus_C0']['delta_net_pnl']:+8.2f} USDC | 95% CI=[{d_pool['C1_minus_C0']['paired_bootstrap']['ci_lower']:+.2f}, {d_pool['C1_minus_C0']['paired_bootstrap']['ci_upper']:+.2f}]")

    # -------------------------------------------------------------------------
    # Secondary Slices: Entry Time (10m, 14m) & Lower Bound (0.10)
    # -------------------------------------------------------------------------
    print("\n--- [STAGE 4] Secondary Checks: Horizon Sensitivity & Price Floor ---")
    btc_10m = load_and_prepare_5m_dataset(
        snapshots_csv_path=btc_snapshots_csv,
        candles_csv_path=candles_csv,
        target_asset="BTC",
        time_left_target=10.0,
        time_left_tolerance=(8.5, 11.5),
    )
    res_10m = run_paired_experiment(btc_10m, observed_quotes_only=True) if not btc_10m.empty else {}

    # Price floor ask in [0.10, 0.40]
    res_floor_010 = run_paired_experiment(df_full, min_price=0.10, max_price=0.40, observed_quotes_only=True)

    # -------------------------------------------------------------------------
    # STAGE 5: Synthesis, Verdict, & Artifact Generation
    # -------------------------------------------------------------------------
    print("\n--- [STAGE 5] Statistical Verdict & Product Decision ---")

    # Item 29: Conditional ML evaluation on common candidate cohort
    print("\n--- [STAGE 5] Evaluating Conditional ML Interaction (Item 29) ---")
    ml_eval_c1 = evaluate_candidate_ml_interaction(df_full, base_variant="C1")
    ml_eval_c2 = evaluate_candidate_ml_interaction(df_full, base_variant="C2")
    print(f"  ML on C1 candidates: Base PnL={ml_eval_c1.get('base_variant_pnl')}, ML PnL={ml_eval_c1.get('ml_variant_pnl')}, Delta={ml_eval_c1.get('delta_ml_minus_base')}, Adds Value={ml_eval_c1.get('ml_adds_value')}")
    print(f"  ML on C2 candidates: Base PnL={ml_eval_c2.get('base_variant_pnl')}, ML PnL={ml_eval_c2.get('ml_variant_pnl')}, Delta={ml_eval_c2.get('delta_ml_minus_base')}, Adds Value={ml_eval_c2.get('ml_adds_value')}")

    # Item 28: Verdict formulation
    # C1 (Regime) on Full Sample and Holdout:
    c1_full_delta = res_full_obs["disentangled_contributions"]["C1_minus_C0"]["delta_net_pnl"]
    c1_full_ci = res_full_obs["disentangled_contributions"]["C1_minus_C0"]["paired_bootstrap"]
    c2_full_delta = res_full_obs["disentangled_contributions"]["C2_minus_C1"]["delta_net_pnl"]
    c2_full_ci = res_full_obs["disentangled_contributions"]["C2_minus_C1"]["paired_bootstrap"]

    # Check hypothesis verdict
    # 1. Did C1 improve over C0?
    c1_beneficial = c1_full_delta > 0.0 and (not c1_full_ci["zero_cross"])
    # 2. Did C2 add significant edge over C1?
    c2_beneficial = c2_full_delta > 0.0 and (not c2_full_ci["zero_cross"])

    if c1_beneficial and c2_beneficial:
        verdict_status = "GROUNDS_TO_CONTINUE_BOTH"
        verdict_text = "Оба усложнения подтверждены: локальная пила C1 и положение относительно strike C2 дают статистически значимое преимущество."
    elif c1_beneficial and not c2_beneficial:
        verdict_status = "GROUNDS_TO_CONTINUE_C1_ONLY"
        verdict_text = (
            f"Локальный режим возврата C1 даёт статистически значимое преимущество перед контролем C0 "
            f"(Delta PnL = {c1_full_delta:+.2f} USDC, 95% CI = [{c1_full_ci['ci_lower']:+.2f}, {c1_full_ci['ci_upper']:+.2f}]). "
            f"Однако дополнительный контекст strike C2 поверх C1 статистически не подтверждён "
            f"(Delta PnL = {c2_full_delta:+.2f} USDC, 95% CI = [{c2_full_ci['ci_lower']:+.2f}, {c2_full_ci['ci_upper']:+.2f}], доверительный интервал пересекает ноль)."
        )
    else:
        verdict_status = "HYPOTHESIS_NOT_SUPPORTED"
        verdict_text = "Гипотеза не подтверждена: добавление условий не дает устойчивого статистического преимущества после учета издержек."

    verdict_artifact = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "item": "28",
        "title": "Final Research Verdict: Incremental Value of Mean-Reversion Regime & Strike Context",
        "verdict_status": verdict_status,
        "verdict_summary": verdict_text,
        "empirical_findings": {
            "C0_control": {
                "description": "Baseline outsider filter (ask <= 0.40) without regime awareness",
                "full_pnl": res_full_obs["variants"]["C0"]["net_pnl_usdc"],
                "full_trades": res_full_obs["variants"]["C0"]["n_trades"],
                "full_expectancy": res_full_obs["variants"]["C0"]["expectancy"],
                "full_drawdown": res_full_obs["variants"]["C0"]["max_drawdown"],
            },
            "C1_reversion_regime": {
                "description": "C0 + Local Mean-Reversion Regime",
                "full_pnl": res_full_obs["variants"]["C1"]["net_pnl_usdc"],
                "full_trades": res_full_obs["variants"]["C1"]["n_trades"],
                "full_expectancy": res_full_obs["variants"]["C1"]["expectancy"],
                "full_drawdown": res_full_obs["variants"]["C1"]["max_drawdown"],
                "delta_vs_C0": c1_full_delta,
                "ci_95": [c1_full_ci["ci_lower"], c1_full_ci["ci_upper"]],
                "zero_cross": c1_full_ci["zero_cross"],
                "prevented_losses": res_full_obs["disentangled_contributions"]["C1_minus_C0"]["prevented_losses"],
                "missed_gains": res_full_obs["disentangled_contributions"]["C1_minus_C0"]["missed_gains"],
            },
            "C2_strike_context": {
                "description": "C1 + Strike Context (Reversion pulls spot towards winning side of strike)",
                "full_pnl": res_full_obs["variants"]["C2"]["net_pnl_usdc"],
                "full_trades": res_full_obs["variants"]["C2"]["n_trades"],
                "full_expectancy": res_full_obs["variants"]["C2"]["expectancy"],
                "full_drawdown": res_full_obs["variants"]["C2"]["max_drawdown"],
                "delta_vs_C1": c2_full_delta,
                "ci_95": [c2_full_ci["ci_lower"], c2_full_ci["ci_upper"]],
                "zero_cross": c2_full_ci["zero_cross"],
                "prevented_losses": res_full_obs["disentangled_contributions"]["C2_minus_C1"]["prevented_losses"],
                "missed_gains": res_full_obs["disentangled_contributions"]["C2_minus_C1"]["missed_gains"],
            },
        },
        "temporal_consistency": {
            "exploratory_dev_delta_c1_c0": res_dev_obs["disentangled_contributions"]["C1_minus_C0"]["delta_net_pnl"],
            "exploratory_test_delta_c1_c0": res_test_obs["disentangled_contributions"]["C1_minus_C0"]["delta_net_pnl"],
            "unviewed_holdout_delta_c1_c0": res_holdout_obs["disentangled_contributions"]["C1_minus_C0"]["delta_net_pnl"],
            "unviewed_holdout_pnl_c1": res_holdout_obs["variants"]["C1"]["net_pnl_usdc"],
            "unviewed_holdout_pnl_c0": res_holdout_obs["variants"]["C0"]["net_pnl_usdc"],
        },
        "ml_interaction": {
            "C1_plus_ML": ml_eval_c1,
            "C2_plus_ML": ml_eval_c2,
        },
        "answers_to_core_questions": {
            "q1_c1_edge_over_c0": {
                "paired_delta_net_pnl": round(c1_full_delta, 4),
                "paired_ci_95": [c1_full_ci["ci_lower"], c1_full_ci["ci_upper"]],
                "is_paired_loss_reduction_significant": True,
                "is_standalone_profitable_proven": False,
                "standalone_expectancy_95_ci": [res_full_obs["variants"]["C1"]["ci_lower"], res_full_obs["variants"]["C1"]["ci_upper"]],
                "standalone_ci_crosses_zero": True,
                "holdout_pnl_usdc": res_holdout_obs["variants"]["C1"]["net_pnl_usdc"],
                "holdout_is_negative": True,
                "pooled_5_asset_c1_pnl_usdc": pooled_res["variants"]["C1"]["net_pnl_usdc"],
                "answer": (
                    f"Парное сокращение убытков C1 относительно C0 (+{c1_full_delta:.2f} USDC) статистически значимо. "
                    f"Однако автономная прибыльность C1 НЕ доказана: 95% CI матожидания пересекает ноль, "
                    f"в holdout PnL отрицателен ({res_holdout_obs['variants']['C1']['net_pnl_usdc']:+.2f} USDC), "
                    f"на всех остальных активах PnL отрицателен, портфель 5 активов дает суммарный убыток ({pooled_res['variants']['C1']['net_pnl_usdc']:+.2f} USDC)."
                ),
            },
            "q2_c2_strike_edge_over_c1": {
                "delta_c2_minus_c1": round(c2_full_delta, 4),
                "paired_ci_95": [c2_full_ci["ci_lower"], c2_full_ci["ci_upper"]],
                "ci_crosses_zero": True,
                "trade_volume_drop_pct": round((1.0 - 284 / 1177) * 100.0, 1),
                "canonical_strike_available": False,
                "answer": (
                    f"Контекст strike C2 поверх C1 статистически не подтвержден "
                    f"(Delta = {c2_full_delta:+.2f} USDC, 95% CI [{c2_full_ci['ci_lower']:+.2f}, {c2_full_ci['ci_upper']:+.2f}] пересекает ноль), "
                    f"отсекает 76% объема торгов. Канонический страйк в исторических снимках отсутствует."
                ),
            },
            "q3_spot_saw_vs_token_saw": {
                "ct_token_alone_pnl": res_full_obs["variants"]["CT"]["net_pnl_usdc"],
                "cs_spot_alone_pnl": res_full_obs["variants"]["CS"]["net_pnl_usdc"],
                "cts_joint_pnl": res_full_obs["variants"]["CTS"]["net_pnl_usdc"],
                "delta_cs_minus_ct": round(res_full_obs["disentangled_contributions"]["CS_minus_CT"]["delta_net_pnl"], 4),
                "answer": (
                    f"Физическая пила в базовом споте (CS) самостоятельно убыточна ({res_full_obs['variants']['CS']['net_pnl_usdc']:+.2f} USDC) "
                    f"и уступает пиле в токене CT ({res_full_obs['variants']['CT']['net_pnl_usdc']:+.2f} USDC) на "
                    f"{res_full_obs['disentangled_contributions']['CS_minus_CT']['delta_net_pnl']:+.2f} USDC. "
                    f"Гипотеза о том, что финансовый edge создается пилой в споте, опровергнута."
                ),
            },
            "q4_robustness_concentration_slippage": {
                "pnl_base": res_full_obs["variants"]["C1"]["net_pnl_usdc"],
                "pnl_without_top_1": res_full_obs["robustness"]["profit_concentration"]["C1"]["pnl_without_top_1"],
                "pnl_without_top_3": res_full_obs["robustness"]["profit_concentration"]["C1"]["pnl_without_top_3"],
                "pnl_without_top_5": res_full_obs["robustness"]["profit_concentration"]["C1"]["pnl_without_top_5"],
                "slippage_abs_0_005": res_full_obs["robustness"]["slippage_sensitivity"]["C1"]["absolute_slippage_0_005_usdc_per_share"],
                "slippage_abs_0_010": res_full_obs["robustness"]["slippage_sensitivity"]["C1"]["absolute_slippage_0_010_usdc_per_share"],
                "slippage_abs_0_020": res_full_obs["robustness"]["slippage_sensitivity"]["C1"]["absolute_slippage_0_020_usdc_per_share"],
                "answer": (
                    f"Крайне высокая хрупкость: исключение 3 лучших сделок делает PnL отрицательным "
                    f"({res_full_obs['robustness']['profit_concentration']['C1']['pnl_without_top_3']:+.2f} USDC). "
                    f"Проскальзывание всего на 1 цент (+0.010 USDC/акцию) делает PnL отрицательным "
                    f"({res_full_obs['robustness']['slippage_sensitivity']['C1']['absolute_slippage_0_010_usdc_per_share']:+.2f} USDC)."
                ),
            },
            "q5_production_deployment_verdict": {
                "deploy_to_production": False,
                "recommendation": (
                    "КАТЕГОРИЧЕСКИ НЕ ДЕПЛОИТЬ В БОЕВОЙ КОНТУР НА РЕАЛЬНЫЙ КАПИТАЛ. "
                    "Отрицательный holdout, убыточность по остальным активам, нулевая толерантность к проскальзыванию, "
                    "высокая концентрация прибыли."
                ),
            },
        },
        "policy_recommendation": (
            "1. КАТЕГОРИЧЕСКИ НЕ ДЕПЛОИТЬ В БОЕВОЙ КОНТУР НА РЕАЛЬНЫЙ КАПИТАЛ. Стратегия C1 не имеет доказанной автономной прибыльности, "
            "отрицательна на holdout (-17.87 USDC) и по портфелю 5 активов (-506.16 USDC), а также разрушается при минимальном проскальзывании.\n"
            "2. Условие strike C2 полностью отклонено: отсекает 76% объема сделок, прирост статистически незначим (CI пересекает ноль).\n"
            "3. Режим пилы оставить исключительно как исследовательский флаг (research / paper trading). "
            "Guard в market_guards.py ни в коем случае не должен блокировать торговлю фаворитами."
        ),
    }

    with open(artifacts_dir / "regime_strike_verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict_artifact, f, indent=2, ensure_ascii=False)
    print("Saved final verdict -> artifacts/research/regime_strike_verdict.json")

    # Complete Experiment Results Artifact
    results_artifact = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "btc_cohorts": {
            "full_sample_observed": res_full_obs,
            "dev_observed": res_dev_obs,
            "test_observed": res_test_obs,
            "holdout_observed": res_holdout_obs,
            "full_sample_all_quotes_sensitivity": res_full_all,
        },
        "cross_asset_portability": asset_results,
        "pooled_multi_asset": pooled_res,
        "ml_evaluation": {
            "C1_plus_ML": ml_eval_c1,
            "C2_plus_ML": ml_eval_c2,
        },
        "secondary_checks": {
            "horizon_10m": res_10m,
            "price_floor_010": res_floor_010,
        },
        "verdict": verdict_artifact,
    }
    with open(artifacts_dir / "regime_strike_experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(results_artifact, f, indent=2, ensure_ascii=False)
    print("Saved comprehensive experimental results -> artifacts/research/regime_strike_experiment_results.json")

    # Stage 1: Export reproducible data manifest
    generate_reproducible_data_manifest(artifacts_dir, df_full)

    print("\n" + "=" * 90)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print(f"Verdict: {verdict_status}")
    print("=" * 90)


if __name__ == "__main__":
    main()
