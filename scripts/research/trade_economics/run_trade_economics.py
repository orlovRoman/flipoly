import json
import hashlib
import os
import subprocess
from datetime import datetime
import pandas as pd
import numpy as np

BASE_DIR = "/home/orlovrp/flipoly-worktrees/trade-economics"
LEDGER_PATH = os.path.join(BASE_DIR, "artifacts/research/common_opportunity_ledger.json")
import shutil
RUN_ID = datetime.now().strftime("run_%Y%m%d_%H%M%S")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts/research/trade_economics", RUN_ID)

from polyflip.research.trade_economics.commissions import calculate_commission
from polyflip.research.trade_economics.pnl import apply_fee_to_budget, calculate_net_pnl

def get_git_commit(rev="HEAD"):
    try:
        return subprocess.check_output(['git', 'rev-parse', rev], cwd=BASE_DIR).decode('utf-8').strip()
    except Exception:
        return "UNKNOWN"
        
def is_dirty():
    try:
        return bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=BASE_DIR).strip())
    except Exception:
        return True

def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    latest_link = os.path.join(BASE_DIR, "artifacts/research/trade_economics", "latest")
    if os.path.exists(latest_link) or os.path.islink(latest_link):
        if os.path.isdir(latest_link) and not os.path.islink(latest_link):
            shutil.rmtree(latest_link)
        else:
            os.remove(latest_link)
    try:
        os.symlink(RUN_ID, latest_link)
    except OSError:
        pass
    
    commit_sha = get_git_commit("HEAD")
    try:
        # Assuming origin/main is the base for research
        base_commit_sha = subprocess.check_output(['git', 'merge-base', 'origin/main', 'HEAD'], cwd=BASE_DIR).decode('utf-8').strip()
    except Exception:
        base_commit_sha = get_git_commit("HEAD^")
    
    if os.path.exists(LEDGER_PATH):
        with open(LEDGER_PATH, 'rb') as f:
            file_bytes = f.read()
            ledger_hash = hashlib.sha256(file_bytes).hexdigest()
            
        try:
            data = json.loads(file_bytes)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame(data.get("opportunities", []))
        except Exception:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()
        ledger_hash = ""
        
    assets = df['asset'].unique().tolist() if not df.empty and 'asset' in df.columns else ["all"]
    
    if not df.empty and 'calendar_date' in df.columns:
        valid_dates = df['calendar_date'].dropna()
        if len(valid_dates) > 0:
            period_utc = f"{valid_dates.min()} to {valid_dates.max()}"
        else:
            period_utc = "UNKNOWN"
    else:
        period_utc = "UNKNOWN"
        
    # 6-7. Fee evidence
    # Demonstration fee and Unknown fee
    fee_evidence = [
        {
            "market_id": "*",
            "valid_from": "2020-01-01",
            "valid_to": "2099-12-31",
            "liquidity_role": "taker",
            "formula_id": "FIXED_PERCENTAGE",
            "parameters": {"rate": 0.001},
            "charged_asset": "USD",
            "round_method": "ROUND_DOWN",
            "round_decimals": 4,
            "source_reference": "DEMONSTRATION",
            "evidence_status": "DEMONSTRATION"
        },
        {
            "market_id": "*",
            "valid_from": "2020-01-01",
            "valid_to": "2099-12-31",
            "liquidity_role": "taker",
            "formula_id": "UNKNOWN",
            "parameters": {},
            "charged_asset": "USD",
            "round_method": "NONE",
            "source_reference": "UNKNOWN",
            "evidence_status": "UNKNOWN"
        }
    ]
    with open(os.path.join(ARTIFACTS_DIR, "fee_evidence.json"), "w") as f:
        json.dump(fee_evidence, f, indent=4)
        
    fee_evidence_str = json.dumps(fee_evidence, sort_keys=True)
    fee_hash = hashlib.sha256(fee_evidence_str.encode('utf-8')).hexdigest()
        
    # 4. Create manifest
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
        "policy_version": "1.0",
        "accounting_convention": "exclusive",
        "fee_evidence_version": "v1",
        "fee_evidence_hash": fee_hash
    }
    
    with open(os.path.join(ARTIFACTS_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=4)
        
    # 5. Split actuals and historical
    actual_fills = "BLOCKED_DATA"
    
    reports_parts = []
    
    if not df.empty and 'variants' in df.columns:
        # Filter for C0 and CT specifically using flags
        df_c0 = df[df['variants'].apply(lambda x: x.get('C0', False) if isinstance(x, dict) else False)].copy()
        df_ct = df[df['variants'].apply(lambda x: x.get('CT', False) if isinstance(x, dict) else False)].copy()
        
        datasets = [("C0", df_c0), ("CT", df_ct)]
        
        for name, df_trades in datasets:
            total_trades = len(df_trades)
            # Check mandatory columns
            required_cols = ['gross_pnl', 'net_pnl', 'shares', 'executable_ask', 'target']
            missing_cols = [c for c in required_cols if c not in df_trades.columns]
            if missing_cols:
                raise ValueError(f"Missing mandatory columns for reproduction: {missing_cols} in {name}.")
            
            # 13. Reproduce old calculation (baseline is 0.002 taker fee)
            def get_settlement(r):
                if pd.isna(r.get('target')) or str(r.get('final_outcome')).upper() == 'UNKNOWN':
                    return np.nan
                return float(r['shares']) if float(r['target']) == 1.0 else 0.0

            df_trades['reproduced_purchase_cash'] = df_trades['shares'] * df_trades['executable_ask']
            df_trades['reproduced_settlement'] = df_trades.apply(get_settlement, axis=1)
            df_trades['reproduced_gross_pnl'] = df_trades['reproduced_settlement'] - df_trades['reproduced_purchase_cash']
            df_trades['reproduced_fee'] = df_trades['shares'] * df_trades['executable_ask'] * 0.002
            df_trades['reproduced_net_pnl'] = df_trades['reproduced_gross_pnl'] - df_trades['reproduced_fee']
            
            # Check reproduction with strict tolerances
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
            
            # Apply historical schema (UNKNOWN, index 1)
            historical_scheme = fee_evidence[1]
            df_trades['historical_fee'] = df_trades.apply(
                lambda row: calculate_commission(
                    scheme=historical_scheme,
                    role='taker',
                    price=row['executable_ask'],
                    shares=row['shares'],
                    transaction_date=row.get('calendar_date')
                ), axis=1
            )
            
            # Calculate historical PnL
            df_trades['purchase_cash'] = df_trades['reproduced_purchase_cash']
            df_trades['settlement_proceeds'] = df_trades['reproduced_settlement']
            df_trades['historical_net_pnl'] = df_trades.apply(
                lambda row: calculate_net_pnl(
                    sale_proceeds=0.0,
                    settlement_proceeds=row['settlement_proceeds'],
                    purchase_cash=row['purchase_cash'],
                    platform_fees=row['historical_fee'],
                    attributable_network_costs=0.0,
                    confirmed_rebates=0.0,
                    is_fully_closed=True,
                    fee_is_uncertain=pd.isna(row['historical_fee'])
                ), axis=1
            )
            
            # Apply demonstration schema (index 0)
            scenario_scheme = fee_evidence[0]
            df_trades['scenario_fee'] = df_trades.apply(
                lambda row: calculate_commission(
                    scheme=scenario_scheme,
                    role='taker',
                    price=row['executable_ask'],
                    shares=row['shares'],
                    transaction_date=row.get('calendar_date')
                ), axis=1
            )
            
            df_trades['scenario_net_pnl'] = df_trades.apply(
                lambda row: calculate_net_pnl(
                    sale_proceeds=0.0,
                    settlement_proceeds=row['settlement_proceeds'],
                    purchase_cash=row['purchase_cash'],
                    platform_fees=row['scenario_fee'],
                    attributable_network_costs=0.0,
                    confirmed_rebates=0.0,
                    is_fully_closed=True,
                    fee_is_uncertain=pd.isna(row['scenario_fee'])
                ), axis=1
            )
            df_trades['scenario_id'] = scenario_scheme.get('source_reference', 'DEMONSTRATION')
            
            # Coverage is % of trades where historical_fee is not None
            valid_fees = df_trades['historical_fee'].notna().sum()
            coverage_pct = (valid_fees / total_trades * 100) if total_trades > 0 else 0.0
            
            # 21. Sign change assessment (Demonstration)
            df_trades['gross_positive'] = df_trades['reproduced_gross_pnl'] > 0
            df_trades['net_positive'] = df_trades['scenario_net_pnl'] > 0
            df_trades['sign_changed'] = df_trades['gross_positive'] & (~df_trades['net_positive'])
            
            try:
                df_trades.to_parquet(os.path.join(ARTIFACTS_DIR, f"opportunity_economics_{name}.parquet"), index=False)
            except Exception:
                df_trades.to_csv(os.path.join(ARTIFACTS_DIR, f"opportunity_economics_{name}.csv"), index=False)
                
            accounting_diffs = df_trades[['opportunity_id', 'net_pnl', 'reproduced_net_pnl', 'scenario_net_pnl', 'sign_changed']]
            accounting_diffs.to_csv(os.path.join(ARTIFACTS_DIR, f"accounting_differences_{name}.csv"), index=False)
            
            sign_flips = int(df_trades['sign_changed'].sum())
            old_net_sum = float(df_trades['reproduced_net_pnl'].sum())
            new_net_sum = float(df_trades['scenario_net_pnl'].fillna(0).sum())
            
            if "datasets_summary" not in locals():
                datasets_summary = {}
            datasets_summary[name] = {
                "total_trades": total_trades,
                "coverage_pct": coverage_pct,
                "status": "UNKNOWN",
                "scenario_net_pnl_sum": new_net_sum,
                "reproduced_net_pnl_sum": old_net_sum,
                "sign_flips": sign_flips
            }
            
            reports_parts.append(f"### {name} Выборка\n"
                                 f"Всего сделок: {total_trades}\n"
                                 f"Смена знака PnL: {sign_flips}\n"
                                 f"Сумма старого Net PnL (fee=0.2%): {old_net_sum:.2f}\n"
                                 f"Сумма нового Net PnL (демонстрация fee=0.1%): {new_net_sum:.2f}\n"
                                 f"Покрытие подтверждённой комиссией: {coverage_pct:.1f}%\n")

    summary = {
        "total_opportunities": len(df),
        "actual_fills": actual_fills,
        "datasets": datasets_summary if "datasets_summary" in locals() else {}
    }
    with open(os.path.join(ARTIFACTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
        
    report_content = f"""# Отчёт об экономике сделки
    
Run ID: {RUN_ID}
Всего возможностей: {len(df)}

""" + "\n".join(reports_parts) + """
Вывод: Данный расчёт является демонстрационным сценарием (0.1%).
Исторические комиссии неизвестны (обозначены как UNKNOWN).
"""
    with open(os.path.join(ARTIFACTS_DIR, "report.md"), "w") as f:
        f.write(report_content)
        
    print(f"Research run {RUN_ID} completed successfully.")

if __name__ == "__main__":
    main()
