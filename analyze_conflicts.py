import pandas as pd
import io

def main():
    with open('local_conflict_data.csv', 'r') as f:
        # Ignore warning lines
        lines = [line.strip() for line in f if not line.startswith('time=')]
        
    df = pd.read_csv(io.StringIO('\n'.join(lines)))
    if len(df) == 0:
        return
        
    df = df[df['p_flip'].notna()]
    df = df[df['lgbm_vote'].notna()]
    df = df[df['lgbm_vote'] != 'NONE']
    df = df[df['candidate_side'].notna()]
    
    def get_lr_vote(row):
        is_yes = row['candidate_side'] == 'YES'
        fresh_yes_price = row['fresh_price'] if is_yes else (1.0 - row['fresh_price'])
        
        p_flip = row['p_flip']
        if p_flip > 0.62:
            lr_votes_yes = fresh_yes_price < 0.5
        elif p_flip < 0.40:
            lr_votes_yes = fresh_yes_price >= 0.5
        else:
            return 'NONE'
            
        return 'UP' if lr_votes_yes else 'DOWN'

    df['lr_vote'] = df.apply(get_lr_vote, axis=1)
    
    df = df[df['lr_vote'] != 'NONE']
    
    print(f"Trades where both models made a firm directional prediction: {len(df)}")
    if len(df) == 0:
        return
        
    df['lgbm_is_right'] = ((df['lgbm_vote'] == 'UP') & (df['true_winner'] == 'YES')) | ((df['lgbm_vote'] == 'DOWN') & (df['true_winner'] == 'NO'))
    df['lr_is_right'] = ((df['lr_vote'] == 'UP') & (df['true_winner'] == 'YES')) | ((df['lr_vote'] == 'DOWN') & (df['true_winner'] == 'NO'))
    
    df_conflict = df[df['lgbm_vote'] != df['lr_vote']]
    print(f"\nCONFLICTS (LGBM vs LR): {len(df_conflict)}")
    if len(df_conflict) > 0:
        print(f"LogReg was right: {df_conflict['lr_is_right'].sum()} times ({df_conflict['lr_is_right'].mean():.1%})")
        print(f"LightGBM was right: {df_conflict['lgbm_is_right'].sum()} times ({df_conflict['lgbm_is_right'].mean():.1%})")

    df_agree = df[df['lgbm_vote'] == df['lr_vote']]
    print(f"\nAGREEMENTS: {len(df_agree)}")
    if len(df_agree) > 0:
        print(f"Both right: {df_agree['lr_is_right'].sum()} times ({df_agree['lr_is_right'].mean():.1%})")
        
if __name__ == '__main__':
    main()
