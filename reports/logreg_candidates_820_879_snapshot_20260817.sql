-- STEP 0: candidate snapshot SQL, read-only
-- Source: polyflip/scripts/baseline_candidate_replay_20260817.sql

-- STEP 0: LogReg Candidate Replay Baseline Read-Only Queries
-- Target: polyflip_db (PostgreSQL) via polyflip_api
-- Date: 2026-08-17
-- Target branch: logreg-improvement-20260815

-- 1. Summary of Candidate Models (ID 820..879)
SELECT
    COUNT(*)::int AS total_candidates,
    MIN(id)::int AS min_id,
    MAX(id)::int AS max_id,
    COUNT(*) FILTER (WHERE is_active = true)::int AS active_count,
    COUNT(*) FILTER (WHERE is_active = false)::int AS inactive_count,
    COUNT(*) FILTER (WHERE activated_at IS NOT NULL)::int AS activated_at_not_null_count,
    COUNT(*) FILTER (WHERE activation_source IS NOT NULL)::int AS activation_source_not_null_count,
    COUNT(*) FILTER (WHERE activated_by IS NOT NULL)::int AS activated_by_not_null_count,
    COUNT(*) FILTER (WHERE activation_reason IS NOT NULL)::int AS activation_reason_not_null_count,
    COUNT(*) FILTER (WHERE quality_override IS TRUE)::int AS quality_override_true_count
FROM model_registry
WHERE id BETWEEN 820 AND 879;

-- 2. Detail of Candidate Models (ID 820..879) - Safe metadata only
SELECT
    id,
    asset,
    version,
    interval,
    model_type,
    is_active,
    quality_gate_passed,
    dataset_fingerprint,
    accuracy,
    baseline,
    ece,
    brier_score,
    backtest_pnl,
    backtest_trades,
    backtest_wr,
    decision_threshold,
    decision_threshold_down,
    train_samples,
    validation_samples,
    positive_rate,
    precision_at_threshold,
    recall_at_threshold,
    f1_at_threshold,
    training_window_start,
    training_window_end,
    trained_at,
    activation_source,
    activated_at,
    activated_by,
    activation_reason,
    quality_override,
    created_at
FROM model_registry
WHERE id BETWEEN 820 AND 879
ORDER BY id;

-- 3. Summary of OOF Artifacts for Candidates (ID 820..879)
SELECT
    COUNT(*)::int AS total_linked_oof_count,
    COUNT(DISTINCT model_registry_id)::int AS distinct_candidate_models_with_oof,
    MIN(model_registry_id)::int AS min_candidate_id,
    MAX(model_registry_id)::int AS max_candidate_id
FROM model_registry_oof_artifacts
WHERE model_registry_id BETWEEN 820 AND 879;

-- 4. Detail of OOF Artifacts for Candidates (ID 820..879) - Safe metadata only
SELECT
    id,
    model_registry_id,
    schema_version,
    row_count,
    created_at
FROM model_registry_oof_artifacts
WHERE model_registry_id BETWEEN 820 AND 879
ORDER BY model_registry_id;

-- 5. Baseline Active Models Count and Breakdown by Asset/Type/Interval
SELECT
    asset,
    model_type,
    interval,
    COUNT(*)::int AS active_count
FROM model_registry
WHERE is_active = true
GROUP BY asset, model_type, interval
ORDER BY asset, model_type, interval;

-- 6. Detail of Currently Active Models - Safe metadata only
SELECT
    id,
    asset,
    version,
    interval,
    model_type,
    is_active,
    quality_gate_passed,
    accuracy,
    baseline,
    ece,
    backtest_pnl,
    backtest_trades,
    backtest_wr,
    decision_threshold,
    decision_threshold_down,
    trained_at,
    activated_at,
    activated_by,
    activation_source,
    activation_reason
FROM model_registry
WHERE is_active = true
ORDER BY id;

-- 7. Deployment Revisions - Safe metadata only
SELECT
    id,
    revision_key,
    parent_id,
    manifest_hash,
    status,
    created_by,
    created_at,
    activated_at,
    rolled_back_at
FROM deployment_revisions
ORDER BY id;

-- 8. Deployment Events - Safe metadata only
SELECT
    id,
    revision_id,
    event_type,
    actor,
    reason,
    previous_hash,
    event_hash,
    created_at
FROM deployment_events
ORDER BY id;
