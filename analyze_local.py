import pandas as pd
import sys

def main():
    try:
        # Read the file line by line to skip the Docker compose warnings
        # The file is UTF-16 encoded because of PowerShell redirection `>`.
        lines = []
        with open('local_data.csv', 'r', encoding='utf-16') as f:
            for line in f:
                # Docker warnings start with time="
                if line.startswith('time="'):
                    continue
                lines.append(line.strip())
        
        # Now parse the CSV lines
        import csv
        import io
        
        csv_data = '\n'.join(lines)
        df = pd.read_csv(io.StringIO(csv_data))
        
        if len(df) == 0:
            print("No settled trades found in the DB (pnl IS NOT NULL AND status = 'SUCCESS').")
            return
            
        print("Total Settled Trades:", len(df))
        
        df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce').fillna(0.0)
        df['win'] = df['pnl'] > 0
        df['loss'] = df['pnl'] < 0
        
        # Replace empty active_features with 'ml_strategy' or appropriate
        df['active_features'] = df['active_features'].fillna('ml_strategy')
        
        def format_stats(group):
            trades = len(group)
            wins = group['win'].sum()
            losses = group['loss'].sum()
            winrate = wins / trades if trades > 0 else 0
            pnl = group['pnl'].sum()
            return pd.Series({
                'Trades': trades,
                'Wins': wins,
                'Losses': losses,
                'WinRate': f"{winrate*100:.1f}%",
                'Total PnL': f"${pnl:.2f}"
            })
            
        print("\n=== PERFORMANCE BY MODEL TYPE (active_features) ===")
        model_stats = df.groupby('active_features').apply(format_stats)
        print(model_stats.to_string())
        
        print("\n=== PERFORMANCE BY ASSET ===")
        asset_stats = df.groupby('asset').apply(format_stats)
        print(asset_stats.to_string())
        
        print("\n=== PERFORMANCE BY MODEL AND ASSET ===")
        cross_stats = df.groupby(['active_features', 'asset']).apply(format_stats)
        print(cross_stats.to_string())
        
    except Exception as e:
        print("Error processing data:", e)

if __name__ == '__main__':
    main()
