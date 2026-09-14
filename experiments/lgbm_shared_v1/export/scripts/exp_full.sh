#!/bin/bash
# Этап 3 full small tables: candles, live_markets, trades, execution_*, settings.
set -e
OUT=/tmp/audit_exp/full
MAN=/tmp/audit_exp/manifest_parts
mkdir -p $OUT $MAN
export PGAPPNAME=lgbm_audit_export
one() {
  local NAME=$1 QUERY=$2
  local F=$OUT/$NAME.csv.gz
  { echo "BEGIN TRANSACTION READ ONLY;";
    echo "\copy ($QUERY) TO STDOUT CSV HEADER";
    echo "COMMIT;"; } | docker exec -i -e PGAPPNAME=lgbm_audit_export polyflip_db psql -U polyflip -d polyflip -v ON_ERROR_STOP=1 -q 2>$OUT/err_$NAME.log | gzip > $F
  [ ${PIPESTATUS[0]} -eq 0 ] || { echo "PSQL-FAILED $NAME"; exit 1; }
  local ROWS=$(gzip -dc $F | wc -l)
  local SHA=$(sha256sum $F | awk '{print $1}')
  echo "full|$NAME|$ROWS|$SHA" >> $MAN/full.rows
  echo "$NAME rows=$ROWS"
}
one candles "SELECT id,symbol,interval,open_time,close_time,open,high,low,close,volume,taker_buy_volume,source,is_closed FROM crypto_candles ORDER BY symbol,interval,open_time"
one livemarkets "SELECT market_id,asset,question,end_time_est,market_end_at,market_start_at,yes_token_id,no_token_id,current_yes_price,current_no_price,trading_status,resolution_status,final_outcome,resolved_at,underlying_price,strike_value,strike_source,condition_id,slug,status FROM live_markets ORDER BY market_id"
one trades "SELECT * FROM trade_history ORDER BY id"
one exrequests "SELECT * FROM execution_requests ORDER BY id"
one exattempts "SELECT * FROM execution_attempts ORDER BY id"
one exfills "SELECT * FROM execution_fills ORDER BY id"
one exevents "SELECT * FROM execution_events ORDER BY id"
one settings "SELECT * FROM runtime_settings ORDER BY 1"
echo FULL-ALLDONE
