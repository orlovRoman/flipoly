import asyncio
import pandas as pd
from sqlalchemy import create_engine, text

# Подключение к БД
engine = create_engine('postgresql://postgres:postgres@localhost:5432/polyflip')

def analyze():
    # 1. Последние метрики моделей
    with engine.connect() as conn:
        models = pd.read_sql("""
            SELECT asset, version, created_at, metrics 
            FROM model_registry 
            WHERE is_active = true
        """, conn)
    
    print("=== MODEL METRICS ===")
    for _, row in models.iterrows():
        print(f"Asset: {row['asset']}, Version: {row['version']}, Created: {row['created_at']}")
        print(row['metrics'])
        print("-" * 20)

    # 2. Статистика сделок за последние 24 часа по Outsider (trade_mode = 'ml_trend' или 'combined' ? Надо проверить trade_reason)
    with engine.connect() as conn:
        trades = pd.read_sql("""
            SELECT asset, trade_mode, p_win, pnl, bet_size, buy_price, created_at
            FROM trade_history
            WHERE created_at >= NOW() - INTERVAL '24 hours'
        """, conn)
    
    print("\n=== TRADES (24h) ===")
    out = trades[trades['trade_mode'].str.contains('ml_trend|outsider', case=False, na=False)]
    if len(out) == 0:
        # fallback, maybe mode is different
        out = trades
        
    stats = out.groupby('asset').agg(
        count=('pnl', 'count'),
        win_rate=('pnl', lambda x: (x > 0).mean() * 100),
        total_pnl=('pnl', 'sum'),
        avg_p_win=('p_win', 'mean'),
        avg_buy_price=('buy_price', 'mean'),
        avg_bet_size=('bet_size', 'mean'),
        max_win=('pnl', 'max'),
        max_loss=('pnl', 'min')
    ).reset_index()
    
    print(stats.to_string())

analyze()
