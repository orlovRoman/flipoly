#!/bin/bash
# Этап 3 flipslice: dev-market [end-700, end-250] narrow slices (flip-feature eval).
set -e
OUT=/tmp/audit_exp/flipslice
MAN=/tmp/audit_exp/manifest_parts
mkdir -p $OUT $MAN
export PGAPPNAME=lgbm_audit_export
{ echo "BEGIN TRANSACTION READ ONLY;";
  echo "\copy (SELECT s.id,s.market_id,s.recorded_at,s.mid_price,s.best_bid,s.best_ask,s.spread,s.time_left_seconds,s.volume_5min,s.price_velocity,s.final_outcome,s.market_end_at FROM (SELECT m.market_id, COALESCE(x.e1, l.market_end_at, l.end_time_est) AS end3 FROM (SELECT DISTINCT market_id FROM market_snapshots) m LEFT JOIN (SELECT market_id, MAX(market_end_at) AS e1 FROM market_snapshots GROUP BY 1) x USING (market_id) LEFT JOIN live_markets l USING (market_id)) e JOIN market_snapshots s ON s.market_id = e.market_id AND s.recorded_at >= e.end3 - INTERVAL '700 seconds' AND s.recorded_at <= e.end3 - INTERVAL '250 seconds' WHERE e.end3 >= TIMESTAMPTZ '2026-07-06 00:00:00+00' AND e.end3 < TIMESTAMPTZ '2026-09-12 00:00:00+00' ORDER BY s.market_id, s.recorded_at) TO STDOUT CSV HEADER";
  echo "COMMIT;"; } | docker exec -i -e PGAPPNAME=lgbm_audit_export polyflip_db psql -U polyflip -d polyflip -v ON_ERROR_STOP=1 -q 2>$OUT/err.log | gzip > $OUT/flipslice.csv.gz
[ ${PIPESTATUS[0]} -eq 0 ] || { echo PSQL-FAILED; exit 1; }
ROWS=$(gzip -dc $OUT/flipslice.csv.gz | wc -l)
SHA=$(sha256sum $OUT/flipslice.csv.gz | awk '{print $1}')
echo "flipslice|all|$ROWS|$SHA" >> $MAN/flipslice.rows
echo "flipslice rows=$ROWS sha=${SHA:0:12}"
echo FLIPSLICE-ALLDONE
