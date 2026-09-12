-- strike probe candidates: 5 per asset spread across history, closed by time (small table only)
SET statement_timeout TO '30s';
WITH x AS (
  SELECT market_id, asset, end_time_est,
         row_number() OVER (PARTITION BY asset ORDER BY end_time_est) AS rn,
         COUNT(*) OVER (PARTITION BY asset) AS c
  FROM live_markets
  WHERE end_time_est < NOW() - INTERVAL '1 day'
    AND end_time_est >= make_timestamptz(2026,7,1,0,0,0)
)
SELECT market_id, asset, end_time_est FROM x
WHERE rn IN (1, c/4, c/2, (3*c)/4, c)
ORDER BY asset, end_time_est;
