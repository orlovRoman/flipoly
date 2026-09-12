-- coverage verification: actual days/rows before full export (read-only)
SET statement_timeout TO '300s';
SELECT 'snap_min' AS q, MIN(recorded_at) AS v FROM market_snapshots;
SELECT 'snap_max' AS q, MAX(recorded_at) AS v FROM market_snapshots;
SELECT 'snap_days' AS q, COUNT(DISTINCT date_trunc('day', recorded_at)) AS v FROM market_snapshots;
SELECT 'depth_min' AS q, MIN(received_at) AS v FROM orderbook_depth_snapshots;
SELECT 'depth_max' AS q, MAX(received_at) AS v FROM orderbook_depth_snapshots;
SELECT 'ct_res_total' AS q, reltuples::bigint AS v FROM pg_class WHERE relname = 'ct_decision_reservations';
SELECT 'ct_res_minmax' AS q, MIN(decision_at) AS a, MAX(decision_at) AS b FROM ct_decision_reservations;
SELECT date_trunc('day', recorded_at)::date AS d, COUNT(*) AS snap_rows
  FROM market_snapshots GROUP BY 1 ORDER BY 1;
SELECT date_trunc('day', received_at)::date AS d, COUNT(*) AS depth_rows
  FROM orderbook_depth_snapshots GROUP BY 1 ORDER BY 1;
