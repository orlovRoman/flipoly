-- strike existence check (guarded; aborts safely on timeout, no writes)
SET statement_timeout TO '120s';
SELECT 'snap_strike_rows' AS q, COUNT(*) AS v FROM market_snapshots WHERE strike_value IS NOT NULL;
SELECT 'snap_strike_sources' AS q, COALESCE(strike_source, 'NULL') AS s, COUNT(*) AS v FROM market_snapshots
  WHERE strike_value IS NOT NULL GROUP BY 2 ORDER BY 3 DESC LIMIT 20;
SELECT 'live_strike_rows' AS q, COUNT(*) AS v FROM live_markets WHERE strike_value IS NOT NULL;
SELECT 'live_settle_src' AS q, COALESCE(settlement_price_source, 'NULL') AS s, COUNT(*) AS v FROM live_markets
  GROUP BY 2 ORDER BY 3 DESC LIMIT 20;
