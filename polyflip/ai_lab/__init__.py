"""Autonomous experiment laboratory primitives.

The package is intentionally independent from live execution. It provides
immutable manifests and audit-oriented persistence models. Safe orchestration and
the explicit offline adapter executor live in the `orchestrator` and `executor`
submodules; no live adapter is exposed here.
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
