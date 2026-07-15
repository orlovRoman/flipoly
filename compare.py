import pandas as pd

df = pd.read_csv('data.csv')

def get_strategy_group(row):
    features_str = str(row['strategy_type']).lower()
    if 'аутсайдер' in features_str or 'outsider' in features_str:
        return 'Аутсайдер'
    elif 'фаворит' in features_str or 'favorite' in features_str:
        return 'Фаворит'
    else:
        return 'Другое'

df['strategy_group'] = df.apply(get_strategy_group, axis=1)
df['asset_clean'] = df['asset'].apply(lambda x: str(x).split('_')[0].split('USDT')[0].upper())

outsiders = df[df['strategy_group'] == 'Аутсайдер']
btc = outsiders[outsiders['asset_clean'] == 'BTC']
doge = outsiders[outsiders['asset_clean'] == 'DOGE']

def analyze(name, subset):
    wins = subset[subset['pnl'] > 0]
    losses = subset[subset['pnl'] <= 0]
    print(f"--- {name} ---")
    print(f"Total trades: {len(subset)}")
    print(f"Win Rate: {len(wins)/len(subset)*100:.1f}%")
    print(f"Total PnL: ${subset['pnl'].sum():.2f}")
    print(f"Avg bet size: ${subset['amount_usdc'].mean():.2f}")
    print(f"Avg bet size (Wins): ${wins['amount_usdc'].mean():.2f}")
    print(f"Avg bet size (Losses): ${losses['amount_usdc'].mean():.2f}")
    print(f"Avg execution price (Wins): {wins['executed_price'].mean():.3f}")
    print(f"Avg execution price (Losses): {losses['executed_price'].mean():.3f}")
    print(f"Avg PnL per Win: ${wins['pnl'].mean():.2f}")
    print(f"Avg PnL per Loss: ${losses['pnl'].mean():.2f}")
    print("")

analyze("BTC Outsider", btc)
analyze("DOGE Outsider", doge)
