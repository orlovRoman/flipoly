-- canonical_models coverage audit, step 10 (read-only, cheap indexed slices only)
SET statement_timeout TO '60s';
SET TRANSACTION READ ONLY;
SELECT 'snapshots_total' AS q, reltuples::bigint AS v FROM pg_class WHERE relname = 'market_snapshots';
SELECT 'depth_total' AS q, reltuples::bigint AS v FROM pg_class WHERE relname = 'orderbook_depth_snapshots';
SELECT 'underlying_total' AS q, reltuples::bigint AS v FROM pg_class WHERE relname = 'underlying_observations';
SELECT 'day_markets_2026_09_09' AS q, COUNT(DISTINCT market_id) AS v FROM market_snapshots
  WHERE recorded_at >= make_timestamptz(2026,9,9,0,0,0) AND recorded_at < make_timestamptz(2026,9,10,0,0,0);
SELECT 'day_rows_2026_09_09' AS q, COUNT(*) AS v FROM market_snapshots
  WHERE recorded_at >= make_timestamptz(2026,9,9,0,0,0) AND recorded_at < make_timestamptz(2026,9,10,0,0,0);
SELECT 'day_strike_rows_2026_09_09' AS q, COUNT(*) AS v FROM market_snapshots
  WHERE recorded_at >= make_timestamptz(2026,9,9,0,0,0) AND recorded_at < make_timestamptz(2026,9,10,0,0,0)
    AND strike_value IS NOT NULL;
SELECT 'day_asset' AS q, asset, COUNT(DISTINCT market_id) AS markets, COUNT(*) AS rows FROM market_snapshots
  WHERE recorded_at >= make_timestamptz(2026,9,9,0,0,0) AND recorded_at < make_timestamptz(2026,9,10,0,0,0)
  GROUP BY asset ORDER BY asset;
SELECT 'depth_day_rows_2026_09_09' AS q, COUNT(*) AS v FROM orderbook_depth_snapshots
  WHERE received_at >= make_timestamptz(2026,9,9,0,0,0) AND received_at < make_timestamptz(2026,9,10,0,0,0);
SELECT 'underlying_sources' AS q, instrument, source, COUNT(*) AS v FROM underlying_observations
  WHERE event_at >= make_timestamptz(2026,9,9,0,0,0) AND event_at < make_timestamptz(2026,9,10,0,0,0)
  GROUP BY instrument, source ORDER BY instrument, source;
