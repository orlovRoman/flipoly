import psycopg2
import pandas as pd
from datetime import datetime, timedelta

def main():
    conn = psycopg2.connect("postgresql://polyflip:secret@db:5432/polyflip")
    
    query = """
    WITH market_outcomes AS (
        SELECT 
            th.market_id,
            MAX(CASE 
                WHEN th.pnl > 0 THEN th.outcome_bought
                WHEN th.pnl < 0 THEN CASE WHEN th.outcome_bought = 'YES' THEN 'NO' ELSE 'YES' END
            END) as true_winner
        FROM trade_history th
        WHERE th.pnl IS NOT NULL AND th.status = 'SUCCESS'
        GROUP BY th.market_id
    )
    SELECT 
        dfl.market_id,
        dfl.trading_mode,
        dfl.final_action,
        dfl.p_flip,
        dfl.fresh_price,
        dfl.candidate_side,
        dfl.direction_value as lgbm_vote,
        mo.true_winner
    FROM decision_funnel_log dfl
    JOIN market_outcomes mo ON dfl.market_id = mo.market_id
    WHERE mo.true_winner IS NOT NULL
      AND dfl.trading_mode = 'COMBINED'
    """
    df = pd.read_sql(query, conn)
    
    # We want to know what LogReg voted.
    # LogReg outputs p_flip. It predicts OUTSIDER if p_flip > 0.5, TREND if p_flip < 0.5.
    # We also know fresh_price is the price of candidate_side.
    # Since we don't have all data, let's just output this raw data to a CSV and analyze it locally!
    
    # Actually, we can just print the analysis right here in the docker container!
    
    def get_lr_vote(row):
        # We need to know if it voted YES or NO.
        # LogReg predicts the OUTSIDER (p_flip) and TREND (1 - p_flip).
        # We don't have fresh_yes_price, but we have fresh_price and candidate_side.
        # It's easier: p_flip is the probability of FLIP (Outsider winning).
        return 'YES' # Placeholder, we will do it locally.
    
    print(df.to_csv(index=False))

if __name__ == "__main__":
    main()
