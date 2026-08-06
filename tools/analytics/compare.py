import pandas as pd
import io

def main():
    lines = []
    with open('local_compare_data.csv', 'r', encoding='utf-16') as f:
        for line in f:
            if line.startswith('time="'):
                continue
            lines.append(line.strip())
            
    df = pd.read_csv(io.StringIO('\n'.join(lines)))
    
    if len(df) == 0:
        print("No data.")
        return
        
    print(f"Total evaluated trades: {len(df)}")
    
    df['lgbm_buy'] = df['direction_value'].apply(lambda x: 'YES' if x == 'UP' else ('NO' if x == 'DOWN' else 'NONE'))
    
    df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce').fillna(0.0)
    df['won'] = df['pnl'] > 0
    
    df['actual_winner'] = df.apply(lambda row: row['outcome_bought'] if row['won'] else ('NO' if row['outcome_bought'] == 'YES' else 'YES'), axis=1)
    
    # lr_correct means we won (since we bought outcome_bought based on what LR/combined decided)
    df['lr_correct'] = df['won']
    df['lgbm_correct'] = df['lgbm_buy'] == df['actual_winner']
    
    df_lgbm_opinion = df[df['lgbm_buy'] != 'NONE']
    print(f"\nTrades where LightGBM had an opinion (UP/DOWN): {len(df_lgbm_opinion)}")
    print(f"Our Execution (mostly LogReg) WinRate on these: {df_lgbm_opinion['lr_correct'].mean():.1%}")
    print(f"LightGBM WinRate on these: {df_lgbm_opinion['lgbm_correct'].mean():.1%}")
    
    df_conflict = df_lgbm_opinion[df_lgbm_opinion['lgbm_buy'] != df_lgbm_opinion['outcome_bought']]
    print(f"\nTrades where they CONFLICTED: {len(df_conflict)}")
    if len(df_conflict) > 0:
        print(f"LogReg was right: {df_conflict['lr_correct'].sum()} times ({df_conflict['lr_correct'].mean():.1%})")
        print(f"LightGBM was right: {df_conflict['lgbm_correct'].sum()} times ({df_conflict['lgbm_correct'].mean():.1%})")
    
    df_agree = df_lgbm_opinion[df_lgbm_opinion['lgbm_buy'] == df_lgbm_opinion['outcome_bought']]
    print(f"\nTrades where they AGREED: {len(df_agree)}")
    if len(df_agree) > 0:
        print(f"Both right: {df_agree['lr_correct'].sum()} times ({df_agree['lr_correct'].mean():.1%})")

if __name__ == '__main__':
    main()
