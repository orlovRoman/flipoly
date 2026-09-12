-- Read-only export for replay_dashboard_policy.py.
-- Run this inside the database container and redirect COPY output to a CSV.
-- The server only exports; the expensive replay runs on the analyst's machine.
COPY (
WITH decisions AS (
    SELECT DISTINCT ON (decision_run_id)
        decision_run_id::text AS opportunity_id,
        market_id::text AS market_id,
        upper(asset) AS asset,
        created_at AS decision_at,
        COALESCE(weighted_p_final_yes, p_logreg_yes, p_lgbm_yes, p_market_yes) AS p_model_yes,
        CASE
            WHEN weighted_p_final_yes IS NOT NULL THEN 'weighted_p_final_yes'
            WHEN p_logreg_yes IS NOT NULL THEN 'p_logreg_yes'
            WHEN p_lgbm_yes IS NOT NULL THEN 'p_lgbm_yes'
            ELSE 'p_market_yes'
        END AS p_source
    FROM decision_funnel_log
    WHERE created_at >= now() - interval '10 days'
      AND decision_run_id IS NOT NULL
    ORDER BY decision_run_id, id DESC
)
SELECT
    d.opportunity_id,
    d.market_id,
    d.asset,
    d.decision_at,
    COALESCE(q.time_left_seconds::double precision,
             EXTRACT(EPOCH FROM (q.market_end_at - d.decision_at))::double precision) AS time_left_sec,
    q.poly_up_best_ask AS yes_ask,
    q.poly_down_best_ask AS no_ask,
    q.poly_up_mid AS yes_mid,
    q.poly_down_mid AS no_mid,
    d.p_model_yes,
    d.p_source,
    l.final_outcome,
    l.label_available_at,
    q.recorded_at AS quote_at,
    EXTRACT(EPOCH FROM (d.decision_at - q.recorded_at))::double precision AS quote_age_sec
FROM decisions d
LEFT JOIN LATERAL (
    SELECT
        s.recorded_at,
        s.time_left_seconds,
        s.market_end_at,
        s.poly_up_best_ask,
        s.poly_down_best_ask,
        s.poly_up_mid,
        s.poly_down_mid
    FROM market_snapshots s
    WHERE s.market_id = d.market_id
      AND s.recorded_at <= d.decision_at
    ORDER BY s.recorded_at DESC
    LIMIT 1
) q ON TRUE
LEFT JOIN LATERAL (
    SELECT s.final_outcome, s.recorded_at AS label_available_at
    FROM market_snapshots s
    WHERE s.market_id = d.market_id
      AND s.final_outcome IN ('YES','NO')
      AND s.recorded_at >= d.decision_at
    ORDER BY s.recorded_at ASC
    LIMIT 1
) l ON TRUE
) TO STDOUT WITH CSV HEADER;
