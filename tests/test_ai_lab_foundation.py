from datetime import datetime, timedelta, timezone

import pytest

from polyflip.ai_lab.manifests import (
    ManifestError,
    build_deployment_manifest,
    build_experiment_manifest,
    canonical_json,
)
from polyflip.db.models import Base


def test_ai_lab_tables_are_registered_in_shared_metadata():
    expected = {
        "ai_optimization_runs",
        "ai_run_steps",
        "experiment_configs",
        "ai_model_artifacts",
        "experiment_results",
        "deployment_revisions",
        "deployment_events",
        "ai_shadow_assignments",
        "ai_permissions",
        "ai_approval_requests",
    }
    assert expected.issubset(Base.metadata.tables)


def test_canonical_json_is_stable_for_mapping_order_and_utc_times():
    first = {
        "z": 2,
        "created_at": datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc),
        "a": {"b": 2, "a": 1},
    }
    second = {
        "a": {"a": 1, "b": 2},
        "created_at": "2026-08-13T12:00:00Z",
        "z": 2,
    }
    assert canonical_json(first) == canonical_json(second)
    assert canonical_json(first) == canonical_json(
        {
            "z": 2,
            "created_at": datetime(2026, 8, 13, 14, 0, tzinfo=timezone(timedelta(hours=2))),
            "a": {"b": 2, "a": 1},
        }
    )


def test_experiment_manifest_contains_hash_and_required_reproducibility_inputs():
    manifest = build_experiment_manifest(
        {
            "code_sha": "abc123",
            "dataset_fingerprint": "data-1",
            "feature_pipeline_version": "v3",
            "train_window": ["2026-08-01", "2026-08-10"],
            "oot_window": ["2026-08-11", "2026-08-12"],
            "seed": 7,
            "model_params": {"num_leaves": 15},
            "strategy_params": {"branch": "COMBINED"},
            "backtest_params": {"fee_buffer": 0.02},
        }
    )
    assert manifest["manifest_kind"] == "experiment"
    assert len(manifest["manifest_hash"]) == 64


def test_manifest_rejects_naive_datetime_and_non_finite_float():
    with pytest.raises(ManifestError):
        build_experiment_manifest(
            {
                "code_sha": "abc",
                "dataset_fingerprint": "data",
                "feature_pipeline_version": "v1",
                "train_window": datetime(2026, 8, 13),
                "oot_window": ["2026-08-14", "2026-08-15"],
                "seed": 1,
                "model_params": {"x": float("nan")},
                "strategy_params": {},
                "backtest_params": {},
            }
        )


def test_deployment_manifest_requires_policies():
    with pytest.raises(ManifestError):
        build_deployment_manifest({"models": {}, "strategy": {}})
