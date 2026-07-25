SELECT 
  COUNT(*) AS total_lgbm_decisions,
  SUM(CASE WHEN (lgbm_metadata::json->>'bet_size_multiplier')::float = 0.0 OR (lgbm_metadata::json->>'vote_action') = 'SKIP' THEN 1 ELSE 0 END) AS veto_count,
  SUM(CASE WHEN (lgbm_metadata::json->>'bet_size_multiplier')::float = 0.5 AND (lgbm_metadata::json->>'vote_action') != 'SKIP' THEN 1 ELSE 0 END) AS none_count,
  SUM(CASE WHEN (lgbm_metadata::json->>'bet_size_multiplier')::float = 1.0 AND (lgbm_metadata::json->>'vote_action') != 'SKIP' THEN 1 ELSE 0 END) AS agree_count
FROM trade_history 
WHERE lgbm_metadata IS NOT NULL;
