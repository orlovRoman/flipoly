from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import inspect

from polyflip.ai_lab.manifests import (
    ManifestError,
    build_deployment_manifest,
    build_experiment_manifest,
    canonical_json,
)
from polyflip.db.models import AIModelArtifact, AIShadowAssignment, Base


def _experiment_payload(**overrides):
    payload = {
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
    payload.update(overrides)
    return payload


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


def test_ai_model_artifact_avoids_reserved_metadata_attribute():
    mapper = inspect(AIModelArtifact)
    assert "artifact_metadata" in mapper.attrs
    # Keep the physical column name stable for the initial migration/backward compatibility.
    assert "metadata" in AIModelArtifact.__table__.c


def test_shadow_assignment_is_linked_to_optimizer_run():
    mapper = inspect(AIShadowAssignment)
    assert "run_id" in mapper.attrs
    assert AIShadowAssignment.__table__.c.run_id.nullable is True


def test_canonical_json_is_stable_for_mapping_order_and_utc_times():
    first = {
        "z": 2,
        "created_at": datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc),
        "a": {"b": 2, "a": 1},
    }
    second = {
        "a": {"a": 1, "b": 2},
        "created_at": datetime(
            2026, 8, 13, 14, 0, tzinfo=timezone(timedelta(hours=2))
        ),
        "z": 2,
    }
    assert canonical_json(first) == canonical_json(second)


def test_canonical_json_preserves_date_type_information():
    assert canonical_json({"t": date(2026, 8, 13)}) != canonical_json(
        {"t": "2026-08-13"}
    )


def test_experiment_manifest_contains_hash_and_schema_version():
    manifest = build_experiment_manifest(_experiment_payload())
    assert manifest["manifest_kind"] == "experiment"
    assert manifest["schema_version"] == "2"
    assert len(manifest["manifest_hash"]) == 64


def test_manifest_rejects_naive_datetime():
    with pytest.raises(ManifestError, match="naive"):
        build_experiment_manifest(
            _experiment_payload(train_window=datetime(2026, 8, 13))
        )


def test_manifest_rejects_non_finite_float():
    with pytest.raises(ManifestError, match="non-finite"):
        build_experiment_manifest(
            _experiment_payload(model_params={"x": float("nan")})
        )


def test_deployment_manifest_requires_policies():
    with pytest.raises(ManifestError):
        build_deployment_manifest({"models": {}, "strategy": {}})
