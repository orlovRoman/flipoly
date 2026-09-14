#!/bin/bash
# Этап 3 snapsdec export: snapshots of funnel-active markets per day, 08-03..09-13.
set -e
OUT=/tmp/audit_exp/snapsdec
MAN=/tmp/audit_exp/manifest_parts
mkdir -p $OUT $MAN
export PGAPPNAME=lgbm_audit_export
COLS="s.id,s.market_id,s.asset,s.recorded_at,s.received_timestamp,s.market_timestamp,s.mid_price,s.best_bid,s.best_ask,s.poly_up_best_bid,s.poly_up_best_ask,s.poly_up_mid,s.poly_down_best_bid,s.poly_down_best_ask,s.poly_down_mid,s.spread,s.time_left_seconds,s.market_end_at,s.final_outcome,s.binance_spot_mid,s.binance_perp_mid,s.binance_price,s.oracle_price,s.strike_value,s.strike_source,s.volume_5min,s.price_velocity"
DAY=2026-08-03
while [ "$DAY" \< "2026-09-14" ]; do
  NX=$(date -u -d "$DAY +1 day" +%F)
  F=$OUT/snapsdec_$DAY.csv.gz
  { echo "BEGIN TRANSACTION READ ONLY;";
    echo "\copy (SELECT $COLS FROM market_snapshots s JOIN (SELECT DISTINCT market_id FROM decision_funnel_log WHERE created_at >= TIMESTAMPTZ '$DAY 00:00:00+00' AND created_at < TIMESTAMPTZ '$NX 00:00:00+00') dm USING (market_id) WHERE s.recorded_at >= TIMESTAMPTZ '$DAY 00:00:00+00' AND s.recorded_at < TIMESTAMPTZ '$NX 00:00:00+00' ORDER BY s.id) TO STDOUT CSV HEADER";
    echo "COMMIT;"; } | docker exec -i -e PGAPPNAME=lgbm_audit_export polyflip_db psql -U polyflip -d polyflip -v ON_ERROR_STOP=1 -q 2>$OUT/err_$DAY.log | gzip > $F
  [ ${PIPESTATUS[0]} -eq 0 ] || { echo "PSQL-FAILED $DAY"; exit 1; }
  ROWS=$(gzip -dc $F | wc -l)
  SHA=$(sha256sum $F | awk '{print $1}')
  echo "snapsdec|$DAY|$ROWS|$SHA" >> $MAN/snapsdec.rows
  echo "$DAY rows=$ROWS"
  DAY=$NX
done
echo SNAPSDEC-ALLDONE
