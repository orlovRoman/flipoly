#!/bin/bash
cd /home/orlovrp/flipoly

docker compose exec -T db psql -U polyflip -d polyflip -c "\copy (
    WITH market_outcomes AS (
        SELECT 
            th.market_id,
            -- If pnl > 0 and we bought YES, then YES won. If pnl > 0 and we bought NO, NO won.
            -- If pnl < 0 and we bought YES, NO won. If pnl < 0 and we bought NO, YES won.
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
        dfl.direction_value as lgbm_vote,
        mo.true_winner
    FROM decision_funnel_log dfl
    JOIN market_outcomes mo ON dfl.market_id = mo.market_id
    WHERE mo.true_winner IS NOT NULL
      AND dfl.trading_mode = 'COMBINED'
) TO STDOUT WITH CSV HEADER" > combined_conflict_data.csv
