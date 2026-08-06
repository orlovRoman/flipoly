#!/bin/bash
cd /home/orlovrp/flipoly
docker compose exec -T db psql -U polyflip -d polyflip -c "\copy (SELECT active_features, asset, status, executed_price, pnl, predicted_flip_prob, outcome_bought FROM trade_history WHERE pnl IS NOT NULL AND status = 'SUCCESS') TO STDOUT WITH CSV HEADER" > data.csv
