import json
import hashlib
import os
import subprocess
from datetime import datetime
import pandas as pd
import numpy as np

BASE_DIR = "/home/orlovrp/flipoly-worktrees/trade-economics"
LEDGER_PATH = os.path.join(BASE_DIR, "artifacts/research/common_opportunity_ledger.json")
RUN_ID = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts/research/trade_economics", RUN_ID)

from polyflip.research.trade_economics.commissions import calculate_commission
from polyflip.research.trade_economics.pnl import apply_fee_to_budget, calculate_net_pnl

def get_git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=BASE_DIR).decode('utf-8').strip()
    except Exception:
        return "UNKNOWN"

def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    commit_sha = get_git_commit()
    
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
        
    # 4. Create manifest
    manifest = {
        "run_id": RUN_ID,
        "base_commit": commit_sha,
        "research_commit": commit_sha,
        "dirty_worktree": True,
        "input_files": {
            "common_opportunity_ledger.json": ledger_hash
        },
        "period_utc": datetime.utcnow().strftime('%Y-%m'),
        "assets": assets,
        "policy_version": "1.0",
        "accounting_convention": "exclusive",
        "fee_evidence_version": "v1"
    }
    
    with open(os.path.join(ARTIFACTS_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=4)
        
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
            "rounding_rule": "ROUND_DOWN",
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
            "rounding_rule": "NONE",
            "source_reference": "UNKNOWN",
            "evidence_status": "UNKNOWN"
        }
    ]
    with open(os.path.join(ARTIFACTS_DIR, "fee_evidence.json"), "w") as f:
        json.dump(fee_evidence, f, indent=4)
        
    # 5. Split actuals and historical
    actual_fills = "BLOCKED_DATA"
    
    reports_parts = []
    
    if not df.empty and 'variants' in df.columns:
        # Filter for C0 and CT specifically using flags
        df_c0 = df[df['variants'].apply(lambda x: x.get('C0', False) if isinstance(x, dict) else False)].copy()
        df_ct = df[df['variants'].apply(lambda x: x.get('CT', False) if isinstance(x, dict) else False)].copy()
        
        datasets = [("C0", df_c0), ("CT", df_ct)]
        
        for name, df_trades in datasets:
            # 13. Reproduce old calculation (baseline is 0.002 taker fee)
            df_trades['reproduced_fee'] = df_trades['shares'] * df_trades['executable_ask'] * 0.002
            df_trades['reproduced_net_pnl'] = df_trades['gross_pnl'] - df_trades['reproduced_fee']
            
            # Check reproduction
            if 'net_pnl' in df_trades.columns:
                mismatch = ~np.isclose(df_trades['net_pnl'], df_trades['reproduced_net_pnl'], atol=1e-4) & df_trades['net_pnl'].notna()
                if mismatch.sum() > 0:
                    print(f"Warning: {mismatch.sum()} rows failed PnL reproduction check in {name}.")
            
            # 14. Apply demonstration schema (index 0)
            scheme = fee_evidence[0]
            
            df_trades['new_fee'] = df_trades.apply(
                lambda row: calculate_commission(
                    scheme=scheme,
                    role='taker',
                    price=row['executable_ask'],
                    shares=row['shares']
                ), axis=1
            )
            
            # 12. Calculate new PnL
            df_trades['purchase_cash'] = df_trades['shares'] * df_trades['executable_ask']
            df_trades['settlement_proceeds'] = df_trades.apply(lambda r: r['shares'] if r['target'] == 1 else 0.0, axis=1)
            df_trades['new_net_pnl'] = df_trades.apply(
                lambda row: calculate_net_pnl(
                    sale_proceeds=0.0,
                    settlement_proceeds=row['settlement_proceeds'],
                    purchase_cash=row['purchase_cash'],
                    platform_fees=row['new_fee'],
                    attributable_network_costs=0.0,
                    confirmed_rebates=0.0,
                    is_fully_closed=True,
                    fee_is_uncertain=pd.isna(row['new_fee'])
                ), axis=1
            )
            
            # Also calculate with UNKNOWN schema (index 1) to show coverage
            df_trades['unknown_fee'] = df_trades.apply(
                lambda row: calculate_commission(
                    scheme=fee_evidence[1],
                    role='taker',
                    price=row['executable_ask'],
                    shares=row['shares']
                ), axis=1
            )
            # Coverage is % of trades where fee is not None
            # Here, unknown_fee is always None
            
            # 21. Sign change assessment (Demonstration)
            df_trades['gross_positive'] = df_trades['gross_pnl'] > 0
            df_trades['net_positive'] = df_trades['new_net_pnl'] > 0
            df_trades['sign_changed'] = df_trades['gross_positive'] & (~df_trades['net_positive'])
            
            try:
                df_trades.to_parquet(os.path.join(ARTIFACTS_DIR, f"opportunity_economics_{name}.parquet"), index=False)
            except Exception:
                df_trades.to_csv(os.path.join(ARTIFACTS_DIR, f"opportunity_economics_{name}.csv"), index=False)
                
            accounting_diffs = df_trades[['opportunity_id', 'net_pnl', 'reproduced_net_pnl', 'new_net_pnl', 'sign_changed']]
            accounting_diffs.to_csv(os.path.join(ARTIFACTS_DIR, f"accounting_differences_{name}.csv"), index=False)
            
            total_trades = len(df_trades)
            sign_flips = int(df_trades['sign_changed'].sum())
            old_net_sum = float(df_trades['reproduced_net_pnl'].sum())
            new_net_sum = float(df_trades['new_net_pnl'].sum())
            
            reports_parts.append(f"""### {name} Выборка
Всего сделок: {total_trades}
Смена знака PnL: {sign_flips}
Сумма старого Net PnL (fee=0.2%): {old_net_sum:.2f}
Сумма нового Net PnL (демонстрация fee=0.1%): {new_net_sum:.2f}
Покрытие подтверждённой комиссией: 0.0% (исторические комиссии неизвестны)
""")

    summary = {
        "total_opportunities": len(df),
        "actual_fills": actual_fills
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
