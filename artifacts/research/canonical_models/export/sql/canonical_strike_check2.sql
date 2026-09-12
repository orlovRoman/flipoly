-- strike-adjacent columns + sample markets with settlement source (small table only)
SET statement_timeout TO '30s';
SELECT 'live_market_start' AS q, COUNT(*) AS v FROM live_markets WHERE market_start_at IS NOT NULL;
SELECT 'live_market_end' AS q, COUNT(*) AS v FROM live_markets WHERE market_end_at IS NOT NULL;
SELECT 'live_oracle' AS q, COUNT(*) AS v FROM live_markets WHERE oracle_price IS NOT NULL;
SELECT 'live_binance_px' AS q, COUNT(*) AS v FROM live_markets WHERE binance_price IS NOT NULL;
SELECT market_id, asset, market_start_at, market_end_at, settlement_price_source
  FROM live_markets WHERE settlement_price_source IS NOT NULL
  ORDER BY market_end_at DESC LIMIT 3;
