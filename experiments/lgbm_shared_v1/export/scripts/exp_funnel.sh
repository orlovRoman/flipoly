#!/bin/bash
# Этап 3 funnel export: daily gzip chunks 07-22..09-13, read-only sessions.
set -e
OUT=/tmp/audit_exp/funnel
MAN=/tmp/audit_exp/manifest_parts
mkdir -p $OUT $MAN
echo '=== cleanup old heavy files (already on D:) ==='
rm -rf /tmp/audit_blobs /tmp/audit_inv/registry_meta.csv
df -h /tmp | tail -1
export PGAPPNAME=lgbm_audit_export
echo '=== session check ==='
echo "SHOW transaction_read_only;" | docker exec -i -e PGAPPNAME=lgbm_audit_export polyflip_db psql -U polyflip -d polyflip -v ON_ERROR_STOP=1 -tAc "BEGIN TRANSACTION READ ONLY;" -c "SHOW transaction_read_only;" -c "COMMIT;" 2>&1 | head -5
COLS="id,created_at,timestamp,market_id,asset,condition_id,trading_mode,execution_mode,decision_run_id,final_action,skip_reason,proposed_action,proposed_price,proposed_amount_usdc,direction_model_key,direction_model_version,direction_regime,direction_status,direction_probability,direction_value,direction_p_up,direction_p_down,direction_p_up_raw,direction_p_down_raw,direction_raw_opinion,direction_threshold_up,direction_threshold_down,direction_discount_mult,direction_error_detail,primary_model_key,primary_model_version,confirm_model_key,confirm_model_version,confirm_direction,confirm_passed,entry_requested_key,entry_model_key,entry_model_version,entry_model_phase,entry_model_source,entry_status,entry_model_ece,fallback_reason,p_flip,p_flip_raw,p_market_yes,p_logreg_yes,p_lgbm_yes,p_logreg_win,p_candidate_win,candidate_side,candidate_ask,gross_edge,cost_buffer,net_edge,edge,min_edge_used,fresh_price,threshold_lower,threshold_upper,strike_source,strike_proxy,underlying_price,distance_to_strike_pct,max_acceptable_price,mrf_mode,mrf_phase,mrf_asset_phase,mrf_applied,mrf_strength,mrf_confidence,mrf_multiplier,mrf_policy_version,mrf_gate_would_block,mrf_gate_reason,mrf_final_action,mrf_final_bet,weighted_policy_mode,weighted_selected_side,weighted_p_market_yes,weighted_p_logreg_yes,weighted_p_lgbm_yes,weighted_p_final_yes,weighted_market_weight,weighted_logreg_weight,weighted_lgbm_weight,weighted_yes_net_ev,weighted_no_net_ev,weighted_fee_rate,weighted_fee_source,weighted_selection_reason,weighted_policy_id,weighted_edge_lower_bound,weighted_missing_components,weighted_expected_execution_price"
DAY=2026-07-22
while [ "$DAY" \< "2026-09-14" ]; do
  NX=$(date -u -d "$DAY +1 day" +%F)
  F=$OUT/funnel_$DAY.csv.gz
  { echo "BEGIN TRANSACTION READ ONLY;";
    echo "\copy (SELECT $COLS FROM decision_funnel_log WHERE created_at >= TIMESTAMPTZ '$DAY 00:00:00+00' AND created_at < TIMESTAMPTZ '$NX 00:00:00+00' ORDER BY id) TO STDOUT CSV HEADER";
    echo "COMMIT;"; } | docker exec -i -e PGAPPNAME=lgbm_audit_export polyflip_db psql -U polyflip -d polyflip -v ON_ERROR_STOP=1 -q 2>$OUT/err_$DAY.log | gzip > $F
  [ ${PIPESTATUS[0]} -eq 0 ] || { echo "PSQL-FAILED $DAY"; exit 1; }
  ROWS=$(gzip -dc $F | wc -l)
  SHA=$(sha256sum $F | awk '{print $1}')
  echo "funnel|$DAY|$ROWS|$SHA" >> $MAN/funnel.rows
  echo "$DAY rows=$ROWS"
  DAY=$NX
done
echo FUNNEL-ALLDONE
