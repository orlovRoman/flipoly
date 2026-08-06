import psycopg2
import pandas as pd
from datetime import datetime, timedelta

def main():
    conn = psycopg2.connect("postgresql://postgres:postgres@localhost:5432/polyflip")
    
    query = """
    SELECT 
        created_at, 
        asset, 
        strategy_type, 
        status,
        executed_price, 
        pnl,
        predicted_flip_prob,
        active_features,
        outcome_bought
    FROM trade_history
    WHERE status IN ('WON', 'LOST')
    """
    df = pd.read_sql(query, conn)
    
    # Calculate some basic metrics
    df['win'] = df['status'] == 'WON'
    df['pnl'] = df['pnl'].astype(float)
    
    # By Model Type
    print("=== By Model (active_features / strategy_type) ===")
    model_stats = df.groupby('active_features').agg({
        'status': 'count',
        'win': 'mean',
        'pnl': 'sum'
    }).rename(columns={'status': 'Trades', 'win': 'WinRate'})
    print(model_stats)
    
    print("\n=== By Asset ===")
    asset_stats = df.groupby('asset').agg({
        'status': 'count',
        'win': 'mean',
        'pnl': 'sum'
    }).rename(columns={'status': 'Trades', 'win': 'WinRate'})
    print(asset_stats)
    
    print("\n=== By Model and Asset ===")
    cross_stats = df.groupby(['active_features', 'asset']).agg({
        'status': 'count',
        'win': 'mean',
        'pnl': 'sum'
    }).rename(columns={'status': 'Trades', 'win': 'WinRate'})
    print(cross_stats)

if __name__ == "__main__":
    main()
