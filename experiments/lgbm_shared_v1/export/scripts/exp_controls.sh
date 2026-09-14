#!/bin/bash
set -e
OUT=/tmp/audit_exp/controls
mkdir -p $OUT
export PGAPPNAME=lgbm_audit_export
{ echo "BEGIN TRANSACTION READ ONLY;";
  echo "\copy (SELECT TO_CHAR(created_at, 'YYYY-MM-DD') AS day, COUNT(*) AS n FROM decision_funnel_log WHERE created_at >= TIMESTAMPTZ '2026-07-22 00:00:00+00' AND created_at < TIMESTAMPTZ '2026-09-14 00:00:00+00' GROUP BY 1 ORDER BY 1) TO STDOUT CSV HEADER";
  echo "COMMIT;"; } | docker exec -i -e PGAPPNAME=lgbm_audit_export polyflip_db psql -U polyflip -d polyflip -v ON_ERROR_STOP=1 -q 2>/dev/null > $OUT/ctl_funnel_day.csv
[ ${PIPESTATUS[0]} -eq 0 ] || exit 1
{ echo "BEGIN TRANSACTION READ ONLY;";
  echo "\copy (SELECT TO_CHAR(s.recorded_at, 'YYYY-MM-DD') AS day, COUNT(*) AS n FROM market_snapshots s JOIN (SELECT DISTINCT market_id, TO_CHAR(created_at, 'YYYY-MM-DD') AS fday FROM decision_funnel_log WHERE created_at >= TIMESTAMPTZ '2026-08-03 00:00:00+00' AND created_at < TIMESTAMPTZ '2026-09-14 00:00:00+00') dm ON dm.market_id = s.market_id AND dm.fday = TO_CHAR(s.recorded_at, 'YYYY-MM-DD') WHERE s.recorded_at >= TIMESTAMPTZ '2026-08-03 00:00:00+00' AND s.recorded_at < TIMESTAMPTZ '2026-09-14 00:00:00+00' GROUP BY 1 ORDER BY 1) TO STDOUT CSV HEADER";
  echo "COMMIT;"; } | docker exec -i -e PGAPPNAME=lgbm_audit_export polyflip_db psql -U polyflip -d polyflip -v ON_ERROR_STOP=1 -q 2>/dev/null > $OUT/ctl_snapsdec_day.csv
[ ${PIPESTATUS[0]} -eq 0 ] || exit 1
echo "COPY-SHOWN $(head -2 $OUT/ctl_funnel_day.csv | tail -1)"
wc -l $OUT/ctl_funnel_day.csv $OUT/ctl_snapsdec_day.csv
echo CONTROLS-ALLDONE
