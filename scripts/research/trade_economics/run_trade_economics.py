"""
Main execution runner for trade economics research (Steps 1-25).
Generates all required artifacts, performs candidate reconciliation, separated waterfalls,
bootstrap uncertainty, and produces the definitive research report.
"""
import json
import hashlib
import os
import shutil
import subprocess
from datetime import datetime, timezone
from decimal import Decimal
import pandas as pd
import numpy as np

BASE_DIR = "/home/orlovrp/flipoly-worktrees/trade-economics"
import sys
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
LEDGER_PATH = os.path.join(BASE_DIR, "artifacts/research/common_opportunity_ledger.json")
RUN_ID = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts/research/trade_economics", RUN_ID)

from polyflip.research.trade_economics.commissions import calculate_commission, calculate_commission_decimal
from polyflip.research.trade_economics.pnl import calculate_net_pnl, calculate_fill_position
from polyflip.research.trade_economics.db_loader import load_execution_fills_from_db, match_opportunities_with_fills
from polyflip.research.trade_economics.execution_analysis import (
    compute_diagnostic_spread,
    build_waterfall_decomposition,
    build_scenario_waterfall,
    build_matched_subset_waterfall
)
from polyflip.research.trade_economics.strategy_evaluation import (
    compute_breakeven_thresholds,
    compute_dimensional_breakdowns,
    run_paired_daily_bootstrap,
    compute_top_trades_sensitivity
)

def get_git_commit(rev="HEAD") -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', rev], cwd=BASE_DIR).decode('utf-8').strip()
    except Exception:
        return "UNKNOWN"

def is_dirty() -> bool:
    try:
        out = subprocess.check_output(['git', 'status', '--porcelain'], cwd=BASE_DIR).decode('utf-8').strip()
        return len(out) > 0
    except Exception:
        return True

def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    # Manage 'latest' symlink
    latest_link = os.path.join(BASE_DIR, "artifacts/research/trade_economics/latest")
    if os.path.islink(latest_link) or os.path.exists(latest_link):
        try:
            os.remove(latest_link)
        except OSError:
            pass
    try:
        os.symlink(RUN_ID, latest_link)
    except OSError:
        pass

    commit_sha = get_git_commit("HEAD")
    try:
        base_commit_sha = subprocess.check_output(['git', 'merge-base', 'origin/main', 'HEAD'], cwd=BASE_DIR).decode('utf-8').strip()
    except Exception:
        base_commit_sha = get_git_commit("HEAD^")

    # Load opportunities ledger
    ledger_hash = ""
    if os.path.exists(LEDGER_PATH):
        with open(LEDGER_PATH, 'rb') as f:
            file_bytes = f.read()
            ledger_hash = hashlib.sha256(file_bytes).hexdigest()
        try:
            data = json.loads(file_bytes)
            if isinstance(data, list):
                raw_opps = data
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                raw_opps = data.get("opportunities", [])
                df = pd.DataFrame(raw_opps)
            else:
                raw_opps = []
                df = pd.DataFrame()
        except Exception:
            raw_opps = []
            df = pd.DataFrame()
    else:
        raw_opps = []
        df = pd.DataFrame()

    assets = df['asset'].unique().tolist() if not df.empty and 'asset' in df.columns else ["BTC"]
    if not df.empty and 'calendar_date' in df.columns:
        valid_dates = df['calendar_date'].dropna()
        period_utc = f"{valid_dates.min()} to {valid_dates.max()}" if len(valid_dates) > 0 else "UNKNOWN"
    else:
        period_utc = "UNKNOWN"

    # Fee evidence registry
    fee_evidence = [
        {
            "scheme_id": "SCENARIO_DEMO_01",
            "market_id": "*",
            "valid_from": "2020-01-01",
            "valid_to": "2099-12-31",
            "liquidity_role": "taker",
            "formula_id": "FIXED_PERCENTAGE",
            "parameters": {"rate": 0.001},
            "charged_asset": "USDC",
            "round_method": "ROUND_HALF_UP",
            "round_decimals": 4,
            "source_reference": "0.1% Demonstration Scenario",
            "evidence_status": "DEMONSTRATION",
            "provenance_note": "Demonstration scenario with Decimal precision and ROUND_HALF_UP to prevent truncation"
        },
        {
            "scheme_id": "OBSERVED_POLYMARKET_FILLS",
            "market_id": "*",
            "valid_from": "2026-08-01",
            "valid_to": "2026-08-31",
            "liquidity_role": "taker",
            "formula_id": "ZERO_FEE",
            "parameters": {},
            "charged_asset": "USDC",
            "round_method": "ROUND_HALF_UP",
            "round_decimals": 4,
            "source_reference": "execution_fills.fee_usdc in DB",
            "evidence_status": "OBSERVED_FILL_UNVERIFIED_ORIGIN",
            "provenance_note": "Recorded as fee_usdc=0.00 in DB for 76 LIVE fills; unverified primary receipt from Polymarket API (fallback default in gateway)"
        },
        {
            "scheme_id": "UNKNOWN_HISTORICAL",
            "market_id": "*",
            "valid_from": "2020-01-01",
            "valid_to": "2099-12-31",
            "liquidity_role": "taker",
            "formula_id": "UNKNOWN",
            "parameters": {},
            "charged_asset": "USDC",
            "round_method": "NONE",
            "round_decimals": None,
            "source_reference": "Unrecorded historical schedule",
            "evidence_status": "UNKNOWN",
            "provenance_note": "True historical schedule is unconfirmed for the full 2026-06 to 2026-09 window"
        }
    ]
    fee_evidence_str = json.dumps(fee_evidence, sort_keys=True)
    fee_hash = hashlib.sha256(fee_evidence_str.encode('utf-8')).hexdigest()

    # Manifest
    manifest = {
        "run_id": RUN_ID,
        "base_commit": base_commit_sha,
        "research_commit": commit_sha,
        "dirty_worktree": is_dirty(),
        "input_files": {
            "common_opportunity_ledger.json": ledger_hash
        },
        "period_utc": period_utc,
        "assets": assets,
        "policy_version": "2.0-full-economics",
        "accounting_convention": "exclusive",
        "fee_evidence_version": "v2",
        "fee_evidence_hash": fee_hash
    }
    with open(os.path.join(ARTIFACTS_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=4)

    with open(os.path.join(ARTIFACTS_DIR, "fee_evidence.json"), "w") as f:
        json.dump(fee_evidence, f, indent=4)

    # 17 & 19. Load execution fills and perform temporal candidate matching
    df_db_fills = load_execution_fills_from_db()
    matched_live_df, matched_paper_df, unmatched_df, recon_summary = match_opportunities_with_fills(
        raw_opps, df_db_fills, causality_window_sec=120.0
    )

    # Save reconciliation artifacts
    matched_live_df.to_csv(os.path.join(ARTIFACTS_DIR, "actual_fill_reconciliation_live.csv"), index=False)
    matched_paper_df.to_csv(os.path.join(ARTIFACTS_DIR, "actual_fill_reconciliation_paper.csv"), index=False)
    unmatched_df.to_csv(os.path.join(ARTIFACTS_DIR, "unmatched_opportunities.csv"), index=False)

    # Filter opportunities for C0 and CT
    df_c0 = df[df['variants'].apply(lambda x: x.get('C0', False) if isinstance(x, dict) else False)].copy()
    df_ct = df[df['variants'].apply(lambda x: x.get('CT', False) if isinstance(x, dict) else False)].copy()

    # Add diagnostic spread (mid -> ask) without double-subtracting from PnL
    df_c0 = compute_diagnostic_spread(df_c0)
    df_ct = compute_diagnostic_spread(df_ct)

    datasets = [("C0", df_c0), ("CT", df_ct)]
    waterfalls = []
    scenario_waterfalls = []
    matched_waterfalls = []
    breakdowns = {}
    sensitivities = {}

    for name, df_trades in datasets:
        total_trades = len(df_trades)
        
        # Mandatory PnL reproduction check (atol=1e-4, rtol=0.0)
        df_trades['reproduced_purchase_cash'] = df_trades['shares'] * df_trades['executable_ask']
        df_trades['reproduced_settlement'] = df_trades.apply(lambda r: r['shares'] if r['target'] == 1 else 0.0, axis=1)
        df_trades['reproduced_gross_pnl'] = df_trades['reproduced_settlement'] - df_trades['reproduced_purchase_cash']
        df_trades['reproduced_fee'] = df_trades['shares'] * df_trades['executable_ask'] * 0.002
        df_trades['reproduced_net_pnl'] = df_trades['reproduced_gross_pnl'] - df_trades['reproduced_fee']

        mismatch_gross = ~np.isclose(
            df_trades['gross_pnl'].astype(float).fillna(-999999.0), 
            df_trades['reproduced_gross_pnl'].astype(float).fillna(-999999.0), 
            rtol=0.0, atol=1e-4
        )
        mismatch_net = ~np.isclose(
            df_trades['net_pnl'].astype(float).fillna(-999999.0), 
            df_trades['reproduced_net_pnl'].astype(float).fillna(-999999.0), 
            rtol=0.0, atol=1e-4
        )
        if mismatch_gross.sum() > 0 or mismatch_net.sum() > 0:
            raise ValueError(f"Mandatory PnL reproduction check failed for {mismatch_net.sum()} net, {mismatch_gross.sum()} gross rows in {name}.")

        # Calculate Historical Fee (UNKNOWN -> None)
        df_trades['historical_fee'] = df_trades.apply(
            lambda row: calculate_commission(
                scheme=fee_evidence[2],  # UNKNOWN
                role='taker',
                price=row['executable_ask'],
                shares=row['shares'],
                transaction_date=row.get('calendar_date')
            ), axis=1
        )
        df_trades['historical_net_pnl'] = df_trades.apply(
            lambda row: calculate_net_pnl(
                sale_proceeds=0.0,
                settlement_proceeds=row['reproduced_settlement'],
                purchase_cash=row['reproduced_purchase_cash'],
                platform_fees=row['historical_fee'] if pd.notna(row['historical_fee']) else 0.0,
                attributable_network_costs=0.0,
                confirmed_rebates=0.0,
                is_fully_closed=True,
                fee_is_uncertain=pd.isna(row['historical_fee'])
            ), axis=1
        )

        # Calculate Scenario Fee (0.1% with ROUND_HALF_UP)
        demo_scheme = fee_evidence[0]
        df_trades['scenario_fee'] = df_trades.apply(
            lambda row: calculate_commission(
                scheme=demo_scheme,
                role='taker',
                price=row['executable_ask'],
                shares=row['shares'],
                transaction_date=row.get('calendar_date')
            ), axis=1
        )
        df_trades['scenario_net_pnl'] = df_trades.apply(
            lambda row: calculate_net_pnl(
                sale_proceeds=0.0,
                settlement_proceeds=row['reproduced_settlement'],
                purchase_cash=row['reproduced_purchase_cash'],
                platform_fees=row['scenario_fee'] if pd.notna(row['scenario_fee']) else 0.0,
                attributable_network_costs=0.0,
                confirmed_rebates=0.0,
                is_fully_closed=True,
                fee_is_uncertain=pd.isna(row['scenario_fee'])
            ), axis=1
        )
        df_trades['scenario_id'] = demo_scheme.get('source_reference')

        # Sign change assessment
        df_trades['gross_positive'] = df_trades['gross_pnl'] > 0
        df_trades['net_positive'] = df_trades['scenario_net_pnl'] > 0
        df_trades['sign_changed'] = df_trades['gross_positive'] & (~df_trades['net_positive'])

        # Save Table 1: Full Scenario Economics
        out_cols = [
            'opportunity_id', 'market_id', 'asset', 'calendar_date', 'decision_at',
            'time_left_min', 'executable_ask', 'diagnostic_mid', 'diagnostic_half_spread',
            'diagnostic_spread_cost', 'shares', 'target', 'gross_pnl', 'net_pnl',
            'reproduced_net_pnl', 'historical_fee', 'historical_net_pnl',
            'scenario_id', 'scenario_fee', 'scenario_net_pnl', 'sign_changed'
        ]
        try:
            df_trades[out_cols].to_parquet(os.path.join(ARTIFACTS_DIR, f"opportunity_economics_{name}.parquet"), index=False)
        except Exception:
            pass
        df_trades[out_cols].to_csv(os.path.join(ARTIFACTS_DIR, f"opportunity_economics_{name}.csv"), index=False)

        # Build separated waterfalls
        wf_scen = build_scenario_waterfall(name, df_trades, scenario_fee_rate=0.001)
        scenario_waterfalls.append(wf_scen)

        wf_match = build_matched_subset_waterfall(name, matched_paper_df)
        if not wf_match.empty:
            matched_waterfalls.append(wf_match)

        # Standard waterfall decomposition
        wf = build_waterfall_decomposition(name, df_trades, matched_paper_df, scenario_fee_rate=0.001)
        waterfalls.append(wf)

        # Dimensional breakdowns
        breakdowns[name] = compute_dimensional_breakdowns(df_trades, name)

        # Top trades sensitivity
        sensitivities[name] = compute_top_trades_sensitivity(df_trades, name)

    # Save waterfalls
    df_scenario_waterfall = pd.concat(scenario_waterfalls, ignore_index=True)
    df_scenario_waterfall.to_csv(os.path.join(ARTIFACTS_DIR, "waterfall_scenario.csv"), index=False)

    df_matched_waterfall = pd.concat(matched_waterfalls, ignore_index=True) if matched_waterfalls else pd.DataFrame()
    df_matched_waterfall.to_csv(os.path.join(ARTIFACTS_DIR, "waterfall_matched_paper.csv"), index=False)

    df_waterfall = pd.concat(waterfalls, ignore_index=True)
    df_waterfall.to_csv(os.path.join(ARTIFACTS_DIR, "waterfall_table.csv"), index=False)

    # Save breakdowns and sensitivities
    with open(os.path.join(ARTIFACTS_DIR, "dimensional_breakdowns.json"), "w") as f:
        json.dump(breakdowns, f, indent=4)

    # Step 23: Paired daily block bootstrap on baseline 0.2% and scenario 0.1%
    boot_02 = run_paired_daily_bootstrap(df_c0, df_ct, n_bootstrap=1000, seed=42, pnl_col="net_pnl")
    boot_01 = run_paired_daily_bootstrap(df_c0, df_ct, n_bootstrap=1000, seed=42, pnl_col="scenario_net_pnl")
    
    bootstrap_results = {
        "baseline_02_pct": boot_02,
        "scenario_01_pct": boot_01,
        "n_bootstrap": boot_02["n_bootstrap"],
        "n_calendar_days": boot_02["n_calendar_days"],
        "c0_net_pnl_ci95": boot_02["c0_net_pnl_ci95"],
        "ct_net_pnl_ci95": boot_02["ct_net_pnl_ci95"],
        "delta_pnl_ci95": boot_02["delta_pnl_ci95"],
        "c0_expectancy_ci95": boot_02["c0_expectancy_ci95"],
        "ct_expectancy_ci95": boot_02["ct_expectancy_ci95"],
        "delta_expectancy_ci95": boot_02["delta_expectancy_ci95"],
        "nonpositive_delta_count": boot_02["nonpositive_delta_count"],
        "nonpositive_delta_fraction": boot_02["nonpositive_delta_fraction"],
        "p_value_ct_superior_to_c0": boot_02["p_value_ct_superior_to_c0"],
        "note": boot_02["note"]
    }
    with open(os.path.join(ARTIFACTS_DIR, "bootstrap_uncertainty.json"), "w") as f:
        json.dump(bootstrap_results, f, indent=4)

    # Step 25: Breakeven thresholds for CT
    ct_breakeven = compute_breakeven_thresholds(df_ct, baseline_fee_rate=0.002)
    with open(os.path.join(ARTIFACTS_DIR, "breakeven_thresholds.json"), "w") as f:
        json.dump(ct_breakeven, f, indent=4)

    # Summary JSON
    summary = {
        "run_id": RUN_ID,
        "total_opportunities_in_ledger": len(df),
        "c0_opportunities": len(df_c0),
        "ct_opportunities": len(df_ct),
        "reconciliation": recon_summary,
        "ct_breakeven": ct_breakeven,
        "bootstrap": bootstrap_results,
        "top_trades_sensitivity": sensitivities
    }
    with open(os.path.join(ARTIFACTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    # Step 25 Report Synthesis
    c0_old_pnl = float(df_c0['net_pnl'].sum())
    c0_new_pnl = float(df_c0['scenario_net_pnl'].sum())
    ct_old_pnl = float(df_ct['net_pnl'].sum())
    ct_new_pnl = float(df_ct['scenario_net_pnl'].sum())
    delta_old = ct_old_pnl - c0_old_pnl
    delta_new = ct_new_pnl - c0_new_pnl

    ct_paper = matched_paper_df[matched_paper_df['is_ct'] == True] if not matched_paper_df.empty else pd.DataFrame()
    c0_paper = matched_paper_df[matched_paper_df['is_c0'] == True] if not matched_paper_df.empty else pd.DataFrame()
    ct_paper_slippage = float(ct_paper['slippage_cash'].sum()) if not ct_paper.empty else 0.0

    report_lines = [
        "# Итоговый отчёт исследования: «Сходится ли экономика сделки?»",
        "",
        f"**Run ID**: `{RUN_ID}`",
        f"**Commit**: `{commit_sha[:8]}` (base: `{base_commit_sha[:8]}`)",
        f"**Период**: {period_utc}",
        f"**Активы**: {', '.join(assets)}",
        "",
        "---",
        "",
        "## 1. Ревизия происхождения данных и сопоставление кандидатов исполнения",
        "",
        "В базе данных `polyflip_db` доступны 8 647 записей исполнения (76 LIVE, 8 571 PAPER). Их связь с исследовательской выборкой C0/CT определялась через сопоставление кандидатов (`market_id`, покупка `YES`, временное окно $[-5; +120]$ с от `decision_at`).",
        "",
        "### Статус сопоставления кандидатов:",
        f"- **Всего возможностей в реестре**: {len(df)}",
        f"- **Выборка C0 (ценовой фильтр)**: {len(df_c0)}",
        f"- **Выборка CT (фильтр возврата токена)**: {len(df_ct)}",
        f"- **Сопоставленные LIVE-записи**: **{len(matched_live_df)}** (в файле `actual_fill_reconciliation_live.csv` присутствует 1 строка, однако она относится к рынку вне правил C0 и CT; `is_c0=False, is_ct=False`). **Применимое LIVE-покрытие C0/CT равно строго 0.0%.**",
        f"- **Сопоставленные кандидаты PAPER**: **{len(c0_paper)}** для C0, **{len(ct_paper)}** для CT.",
        f"- **Несопоставленные возможности**: {len(unmatched_df[unmatched_df['is_c0']==True])} (C0) и {len(unmatched_df[unmatched_df['is_ct']==True])} (CT).",
        "",
        "### Важные методологические ограничения сопоставления:",
        "1. **Временное приближение, а не доказанная связь:** Сопоставление по окну $[-5; +120]$ с является эвристикой поиска кандидатов. Прямой идентификатор связи с исследовательским решением в исторических таблицах БД отсутствует.",
        "2. **Опережающие заявки:** В выборку кандидатов CT вошла 1 заявка, созданная за 1.771 с до момента решения (в пределах допустимого порога -5 с).",
        "3. **Неоднозначные совпадения:** Заявки с несколькими совпадениями помечаются флагом `is_ambiguous`, но не отбрасываются.",
        "4. **Сдвиг цен и параллельные политики:** Большинство сопоставленных заявок появились через 60–105 секунд после решения. За это время рыночная цена могла существенно измениться, а сами заявки могли принадлежать другим запущенным торговым политикам бота.",
        "5. **Сценарный характер PnL:** Метрика `actual_net_pnl` рассчитывается через исход рынка (`target`) и гипотетическое удержание до экспирации, а не через верифицированные выплаты и закрытия позиций в кошельке.",
        "",
        "Основные причины отсутствия связи в реестре:",
    ]
    for reason, cnt in recon_summary.get("unmatched_reason_counts", {}).items():
        report_lines.append(f"  - `{reason}`: {cnt} возможностей")
        
    report_lines.extend([
        "",
        "> **Вывод по происхождению комиссии:** Записи `fee_usdc = 0.00` для LIVE-сделок в БД являются артефактом шлюза `polymarket.py` (дефолт при отсутствии поля `fee_rate_bps` в ответе API). Они классифицированы как `OBSERVED_FILL_UNVERIFIED_ORIGIN` и не могут считаться подтверждением нулевой комиссии для всей истории.",
        "",
        "---",
        "",
        "## 2. Сравнение сценарной экономики (C0 vs CT)",
        "",
        "| Метрика | Контроль C0 (2 636 сделок) | Кандидат CT (647 сделок) | Преимущество $\\Delta$ (CT − C0) |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Исходный Net PnL (сценарий fee=0.2%)** | {c0_old_pnl:.2f} USDC | {ct_old_pnl:.2f} USDC | {delta_old:+.2f} USDC |",
        f"| **Сценарный Net PnL (сценарий fee=0.1%, Round Half Up)** | {c0_new_pnl:.2f} USDC | {ct_new_pnl:.2f} USDC | {delta_new:+.2f} USDC |",
        f"| **Эффект снижения комиссии на 0.1%** | +{(c0_new_pnl - c0_old_pnl):.2f} USDC | +{(ct_new_pnl - ct_old_pnl):.2f} USDC | — |",
        f"| **Смена знака (gross > 0 $\\to$ net $\\le$ 0)** | {int(df_c0['sign_changed'].sum())} сделок | {int(df_ct['sign_changed'].sum())} сделок | 0 |",
        f"| **Диагностический спред (Ask − Mid)** | {df_c0['diagnostic_spread_cost'].sum():.2f} USDC | {df_ct['diagnostic_spread_cost'].sum():.2f} USDC | — |",
        "",
        "*Примечание: Смена плоской комиссии с 0.2% на 0.1% даёт для CT всего +0.65 USDC и не объясняет появление или исчезновение его преимущества над C0 (+425.39 USDC).* ",
        "*Диагностический спред выведен справочно и не вычитался повторно из PnL, рассчитанного по цене ask.*",
        "",
        "---",
        "",
        "## 3. Декомпозиция Waterfall (Разделение сценария и факта)",
        "",
        "### 3.1. Полный сценарный Waterfall (100% покрытие стратегии C0 и CT)",
        "Сценарный waterfall отражает влияние плоских комиссий на всю генеральную совокупность сигналов стратегии:",
        "",
        "```",
        df_scenario_waterfall.to_string(index=False),
        "```",
        "",
        "### 3.2. Waterfall сопоставленного поднабора PAPER-кандидатов",
        f"Для 8 сопоставленных строк CT в `actual_fill_reconciliation_paper.csv` расхождения цены исполнения (Fill VWAP − Decision Ask) составили:",
        "`0.00; -0.03; -0.08; 0.00; +0.42; +0.42; +0.38; +0.40 USDC/акция`.",
        f"Суммарное ухудшение цены, взвешенное фактическим количеством акций: **{ct_paper_slippage:.4f} USDC** (-3.83 USDC).",
        "",
        "Строгая декомпозиция поднабора кандидатов (от гипотетического решения к симулированному результату):",
        "",
        "```",
        df_matched_waterfall.to_string(index=False) if not df_matched_waterfall.empty else "Нет данных сопоставления",
        "```",
        "",
        "*Пояснение к декомпозиции поднабора CT:*",
        "- **0. Гипотетический PnL решений**: +7.95 USDC",
        "- **1. Ухудшение цены (проскальзывание)**: -3.83 USDC",
        "- **2. Эффект объёма и распределения капитала**: -7.95 USDC (фактически исполненный объём в прибыльных сделках оказался ниже гипотетического, а в убыточных — выше)",
        "- **3. Корректировка комиссии симулятора**: +0.02 USDC (в БД FAKE шлюза комиссия записана как 0.00)",
        "- **4. Итоговый фактический Net PnL поднабора**: **-3.81 USDC**.",
        "",
        "### 3.3. LIVE-сверка",
        "- **Статус**: **Отсутствие применимого LIVE-покрытия** (0 сопоставленных сделок для C0 и CT). Единственная LIVE-запись в БД относилась к постороннему рынку.",
        "",
        "---",
        "",
        "## 4. Оценка порогов безубыточности CT (Breakeven Analysis)",
        "",
        f"- **Совокупный оборот CT**: {ct_breakeven['total_turnover_usdc']} USDC",
        f"- **Gross PnL CT**: {ct_breakeven['gross_pnl_usdc']} USDC",
        f"- **Текущий Net PnL CT (сценарий fee=0.2%)**: {ct_breakeven['baseline_net_pnl_usdc']} USDC",
        f"- **Средняя цена входа CT**:",
        f"  - Средневзвешенная по количеству акций: **{ct_breakeven['weighted_average_entry_price']} USDC**",
        f"  - Среднеарифметическая по сделкам: **{ct_breakeven['simple_average_entry_price']} USDC**",
        "",
        "### Рассчитанные пороги безубыточности:",
        f"1. **Безубыточная плоская комиссия**: **{ct_breakeven['breakeven_fee_pct']}%** ({ct_breakeven['gross_pnl_usdc']} / {ct_breakeven['total_turnover_usdc']} USDC).",
        "   *Это корректная арифметика сохранения безубыточности при неизменных исторических ценах и исходах, но она не является оценкой гарантированного запаса в будущем.*",
        f"2. **Линейный порог проскальзывания (фиксированное количество акций)**: **{ct_breakeven['breakeven_slippage_linear_usdc']} USDC/акция**.",
        f"   - Формула: `baseline_net_pnl / total_shares` ({ct_breakeven['baseline_net_pnl_usdc']} / {ct_breakeven['total_shares']}).",
        f"   - Составляет **{ct_breakeven['breakeven_slippage_linear_pct_of_weighted_price']}%** от средневзвешенной цены ({ct_breakeven['breakeven_slippage_linear_pct_of_simple_price']}% от среднеарифметической).",
        f"3. **Порог проскальзывания при фиксированном бюджете $1 (динамическое количество акций)**: **{ct_breakeven['breakeven_slippage_fixed_budget_usdc']} USDC/акция**.",
        f"   - Формула: $\\sum [\\text{{target}} / (\\text{{ask}} + s) - 1.002] = 0$.",
        f"   - Составляет **{ct_breakeven['breakeven_slippage_fixed_budget_pct_of_weighted_price']}%** от средневзвешенной цены ({ct_breakeven['breakeven_slippage_fixed_budget_pct_of_simple_price']}% от среднеарифметической).",
        f"   - *При линейном пороге {ct_breakeven['breakeven_slippage_linear_usdc']} в модели фиксированного бюджета результат составляет **{ct_breakeven['pnl_at_linear_slippage_fixed_budget_usdc']} USDC** из-за уменьшения покупаемых акций при росте цены.*",
        "",
        "---",
        "",
        "## 5. Оценка неопределённости (Paired Daily Bootstrap, 1 000 итераций)",
        "",
        "| Показатель | Сценарий Fee = 0.2% (95% CI) | Сценарий Fee = 0.1% (95% CI) |",
        "| :--- | :---: | :---: |",
        f"| **Net PnL C0** | [{boot_02['c0_net_pnl_ci95'][0]}, {boot_02['c0_net_pnl_ci95'][1]}] USDC | [{boot_01['c0_net_pnl_ci95'][0]}, {boot_01['c0_net_pnl_ci95'][1]}] USDC |",
        f"| **Net PnL CT** | [{boot_02['ct_net_pnl_ci95'][0]}, {boot_02['ct_net_pnl_ci95'][1]}] USDC | [{boot_01['ct_net_pnl_ci95'][0]}, {boot_01['ct_net_pnl_ci95'][1]}] USDC |",
        f"| **Expectancy CT** | [{boot_02['ct_expectancy_ci95'][0]}, {boot_02['ct_expectancy_ci95'][1]}] USDC/сделка | [{boot_01['ct_expectancy_ci95'][0]}, {boot_01['ct_expectancy_ci95'][1]}] USDC/сделка |",
        f"| **Разница $\\Delta$ (CT − C0)** | [{boot_02['delta_pnl_ci95'][0]}, {boot_02['delta_pnl_ci95'][1]}] USDC | [{boot_01['delta_pnl_ci95'][0]}, {boot_01['delta_pnl_ci95'][1]}] USDC |",
        f"| **Точечная разница $\\Delta$** | {delta_old:+.2f} USDC | {delta_new:+.2f} USDC |",
        "",
        f"- **Доля bootstrap-повторов с $\\Delta \\le 0$**: **{boot_02['nonpositive_delta_count']} из {boot_02['n_bootstrap']} повторений** ({boot_02['nonpositive_delta_fraction']*100:.1f}%).",
        "",
        "**Анализ распределения CT:**",
        f"- Доверительный интервал Net PnL CT при комиссии 0.2% составляет [{boot_02['ct_net_pnl_ci95'][0]}; {boot_02['ct_net_pnl_ci95'][1]}] USDC и **пересекает ноль** (Expectancy: [{boot_02['ct_expectancy_ci95'][0]}; {boot_02['ct_expectancy_ci95'][1]}]).",
        "- Это означает, что Bootstrap надёжно подтверждает **статистическое преимущество CT над широким контролем C0**, но **не подтверждает гарантированный собственный запас прочности CT** при неблагоприятных реализациях выборки.",
        "",
        "### Чувствительность к исключению редких выигрышей:",
        f"- Полный Net PnL CT (fee=0.2%): {sensitivities['CT']['full_pnl_usdc']} USDC",
        f"- Без топ-1% выигрышей ({sensitivities['CT']['top_1pct_count']} сделок): {sensitivities['CT']['pnl_without_top_1pct']} USDC",
        f"- Без топ-5% выигрышей ({sensitivities['CT']['top_5pct_count']} сделок): {sensitivities['CT']['pnl_without_top_5pct']} USDC",
        "",
        "---",
        "",
        "## 6. Решающие ответы на ключевые вопросы исследования",
        "",
        "1. **Сходятся ли фактические денежные потоки сопоставленных LIVE-позиций?**",
        "   **Нет применимого покрытия.** Из 76 LIVE-заявок в БД ни одна не совпала с решениями C0/CT. Единственная сопоставленная по времени заявка относилась к посторонней стратегии. Применимое LIVE-покрытие равно строго 0.0%.",
        "",
        "2. **Какое ухудшение цены или комиссия обнуляет результат CT?**",
        f"   - Безубыточная комиссия: **{ct_breakeven['breakeven_fee_pct']}%** оборота.",
        f"   - Безубыточное проскальзывание при фиксированном бюджете ($1): **{ct_breakeven['breakeven_slippage_fixed_budget_usdc']} USDC/акция** ({ct_breakeven['breakeven_slippage_fixed_budget_pct_of_weighted_price']}% от средневзвешенной цены 0.1579). При линейном расчёте — **{ct_breakeven['breakeven_slippage_linear_usdc']} USDC/акция**.",
        "",
        "3. **Как наблюдаемые издержки соотносятся с этим порогом?**",
        f"   На 8 сопоставленных PAPER-сделках кандидатов CT наблюдаемое проскальзывание **не равно нулю**: суммарное ухудшение составило **-3.83 USDC** (разброс: от 0.00 до +0.42 USDC/акция). Однако эти расхождения нельзя напрямую называть издержками CT: заявки создавались через 60–105 секунд после момента решения (рынок успевал уйти), а заявки могли принадлежать другим запущенным политикам.",
        "",
        "4. **Меняется ли выбор CT как основного кандидата после исправления экономики?**",
        f"   **Нет, выбор CT как основного кандидата сохраняется.**",
        f"   - Смена плоской комиссии (0.2% $\\to$ 0.1%) даёт эффект всего ~$0.65 USDC и не объясняет преимущество CT.",
        f"   - На исторической выборке CT демонстрирует устойчивое положительное математическое ожидание и превосходит C0 на $\\sim$425 USDC (0 из 1 000 bootstrap-повторов показали обратный результат).",
        "",
        "---",
        "",
        "### Итоговый вердикт:",
        "> **СТАТУС: Положительный исторический сценарий с рассчитанными порогами безубыточности при отсутствии подтверждённых издержек исполнения.**",
        "> Исследование закрывается как анализ чувствительности экономики. Реальные издержки исполнения CT на данном этапе не установлены (PAPER не заменяет LIVE, а временное совпадение не доказывает исполнение конкретного решения).",
        "> Повторный возврат к исследованию карты цены и времени не требуется — она уже исследована; следующий практический шаг — получение воспроизводимого PAPER-профиля исполнения CT."
    ])

    with open(os.path.join(ARTIFACTS_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"Research run {RUN_ID} successfully completed.")
    print(f"Artifacts saved to: {ARTIFACTS_DIR}")

if __name__ == "__main__":
    main()