import sqlite3
import pandas as pd
import sys

def generate_analytics():
    conn = sqlite3.connect('file:///home/orlovrp/polymarket-bot/vault/database.sqlite?mode=ro', uri=True, timeout=10)
    
    # Query all completed trades
    query = """
    SELECT asset, status, pnl, strategy_type, amount_usdc, executed_price, outcome_bought
    FROM trade_history 
    WHERE status IN ('SUCCESS', 'FAILED') AND pnl IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    
    if df.empty:
        return "В базе данных пока нет завершенных сделок с зафиксированным PnL."
    
    # Clean asset name
    df['asset_clean'] = df['asset'].apply(lambda x: x.split('_')[0].split('USDT')[0].upper())
    
    # Categorize strategy_type
    def get_strategy_group(row):
        st = str(row['strategy_type']).lower()
        if 'аутсайдер' in st or 'outsider' in st:
            return 'Аутсайдер'
        elif 'фаворит' in st or 'favorite' in st:
            return 'Фаворит'
        else:
            return 'Другое'
            
    df['strategy_group'] = df.apply(get_strategy_group, axis=1)
    
    # Aggregate by asset and strategy
    agg = df.groupby(['asset_clean', 'strategy_group']).agg(
        trades_count=('pnl', 'count'),
        total_pnl=('pnl', 'sum'),
        wins=('pnl', lambda x: (x > 0).sum()),
        losses=('pnl', lambda x: (x <= 0).sum()),
        volume=('amount_usdc', 'sum')
    ).reset_index()
    
    agg['win_rate'] = (agg['wins'] / agg['trades_count'] * 100).round(1)
    agg['total_pnl'] = agg['total_pnl'].round(2)
    agg['volume'] = agg['volume'].round(2)
    
    # Generate Markdown
    md = "# Аналитика PnL: Активы и Стратегии\n\n"
    
    total_pnl = df['pnl'].sum()
    total_trades = len(df)
    total_wins = (df['pnl'] > 0).sum()
    total_winrate = (total_wins / total_trades * 100) if total_trades > 0 else 0
    
    md += f"**Всего сделок:** {total_trades}\n"
    md += f"**Общий PnL:** {total_pnl:.2f} USDC\n"
    md += f"**Общий Win Rate:** {total_winrate:.1f}%\n\n"
    
    md += "## В разрезе Стратегий\n\n"
    md += "| Стратегия | Сделок | Win Rate | Объем (USDC) | PnL (USDC) |\n"
    md += "|-----------|--------|----------|--------------|------------|\n"
    
    strat_agg = df.groupby('strategy_group').agg(
        trades_count=('pnl', 'count'),
        total_pnl=('pnl', 'sum'),
        wins=('pnl', lambda x: (x > 0).sum()),
        volume=('amount_usdc', 'sum')
    ).reset_index()
    
    for _, row in strat_agg.iterrows():
        wr = (row['wins'] / row['trades_count'] * 100) if row['trades_count'] > 0 else 0
        md += f"| {row['strategy_group']} | {row['trades_count']} | {wr:.1f}% | ${row['volume']:.2f} | **${row['total_pnl']:.2f}** |\n"
        
    md += "\n## Детализация по Активам и Стратегиям\n\n"
    md += "| Актив | Стратегия | Сделок | Win Rate | PnL (USDC) |\n"
    md += "|-------|-----------|--------|----------|------------|\n"
    
    agg = agg.sort_values(by=['asset_clean', 'total_pnl'], ascending=[True, False])
    for _, row in agg.iterrows():
        md += f"| {row['asset_clean']} | {row['strategy_group']} | {row['trades_count']} | {row['win_rate']}% | **${row['total_pnl']:.2f}** |\n"
        
    return md

if __name__ == "__main__":
    print(generate_analytics())
