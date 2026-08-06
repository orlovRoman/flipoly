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
    
    # original logic
    df['lgbm_is_right'] = ((df['lgbm_vote'] == 'UP') & (df['true_winner'] == 'YES')) | ((df['lgbm_vote'] == 'DOWN') & (df['true_winner'] == 'NO'))
    df['lr_is_right'] = ((df['lr_vote'] == 'UP') & (df['true_winner'] == 'YES')) | ((df['lr_vote'] == 'DOWN') & (df['true_winner'] == 'NO'))
    
    # inverted logic
    df['inverted_lgbm_vote'] = df['lgbm_vote'].apply(lambda x: 'DOWN' if x == 'UP' else ('UP' if x == 'DOWN' else 'NONE'))
    df['inverted_lgbm_is_right'] = ((df['inverted_lgbm_vote'] == 'UP') & (df['true_winner'] == 'YES')) | ((df['inverted_lgbm_vote'] == 'DOWN') & (df['true_winner'] == 'NO'))
    
    print(f"Total firm predictions (both models): {len(df)}")
    print(f"LogReg WinRate: {df['lr_is_right'].sum()} / {len(df)} ({df['lr_is_right'].mean():.1%})")
    print(f"Original LightGBM WinRate: {df['lgbm_is_right'].sum()} / {len(df)} ({df['lgbm_is_right'].mean():.1%})")
    print(f"Inverted LightGBM WinRate: {df['inverted_lgbm_is_right'].sum()} / {len(df)} ({df['inverted_lgbm_is_right'].mean():.1%})")
    
    # New Conflicts
    df_new_conflict = df[df['inverted_lgbm_vote'] != df['lr_vote']]
    print(f"\nNEW CONFLICTS (Inverted LGBM vs LR): {len(df_new_conflict)}")
    if len(df_new_conflict) > 0:
        print(f"LogReg was right: {df_new_conflict['lr_is_right'].sum()} times ({df_new_conflict['lr_is_right'].mean():.1%})")
        print(f"Inverted LightGBM was right: {df_new_conflict['inverted_lgbm_is_right'].sum()} times ({df_new_conflict['inverted_lgbm_is_right'].mean():.1%})")
        
    df_new_agree = df[df['inverted_lgbm_vote'] == df['lr_vote']]
    print(f"\nNEW AGREEMENTS: {len(df_new_agree)}")
    if len(df_new_agree) > 0:
        print(f"Both right: {df_new_agree['lr_is_right'].sum()} times ({df_new_agree['lr_is_right'].mean():.1%})")

if __name__ == '__main__':
    main()
