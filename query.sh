#!/bin/bash
cd /home/orlovrp/flipoly
docker compose exec -T db psql -U polyflip -d polyflip -c "SELECT active_features, COUNT(*) FROM trade_history WHERE pnl IS NOT NULL AND status='SUCCESS' AND decision_run_id IS NOT NULL GROUP BY active_features;"
