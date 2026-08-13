"""Autonomous experiment laboratory primitives.

The package is intentionally independent from live execution.  It provides
immutable manifests and audit-oriented persistence models; orchestration and
Codex integration are added in later phases.
"""

from polyflip.ai_lab.manifests import (
    ManifestError,
    build_deployment_manifest,
    build_experiment_manifest,
    canonical_json,
    compute_manifest_hash,
)

__all__ = [
    "ManifestError",
    "build_deployment_manifest",
    "build_experiment_manifest",
    "canonical_json",
    "compute_manifest_hash",
]
