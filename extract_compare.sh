#!/bin/bash
cd /home/orlovrp/flipoly
docker compose exec -T db psql -U polyflip -d polyflip -c "\copy (
    SELECT 
        th.active_features, 
        th.asset, 
        th.status, 
        th.pnl,
        th.outcome_bought, 
        dfl.p_flip, 
        dfl.fresh_price, 
        dfl.candidate_side, 
        dfl.direction_value
    FROM trade_history th
    JOIN decision_funnel_log dfl ON th.decision_run_id = dfl.decision_run_id
    WHERE th.pnl IS NOT NULL AND th.status = 'SUCCESS'
) TO STDOUT WITH CSV HEADER" > compare_data.csv
