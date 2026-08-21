import asyncio
import hashlib
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

from polyflip.ai_lab import logreg_adapters
from polyflip.ai_lab.executor import StepContext


class _ArtifactSession:
    def __init__(self):
        self.added = None

    def add(self, value):
        self.added = value
        value.id = 901

    async def flush(self):
        return None


def _context():
    return StepContext(
        run_id=4,
        step_id=8,
        action="TRAIN_MODEL",
        config_id=12,
        config_hash="config-hash",
        objective="logreg artifact contract",
        scope={},
        input_payload={
            "strategy_branch": "OUTSIDER_ONLY",
            "train_window": ["2026-08-01", "2026-08-02"],
            "oot_window": ["2026-08-03", "2026-08-04"],
        },
        model_family="LOGREG",
        feature_set="FS_D1",
        asset="BTC",
    )


def test_logreg_train_artifact_contains_exact_linked_bundle_contract():
    row = SimpleNamespace(
        id=77,
        asset="BTC",
        version=3,
        model_blob=b"serialized-logreg",
        features="f1,f2",
        dataset_fingerprint="logreg-fingerprint",
        training_window_start=datetime(2026, 8, 1, tzinfo=timezone.utc),
        training_window_end=datetime(2026, 8, 2, tzinfo=timezone.utc),
    )
    artifact = asyncio.run(
        logreg_adapters._create_bundle_artifact(_ArtifactSession(), _context(), [row])
    )

    assert artifact.config_id == 12
    assert artifact.run_id == 4
    assert artifact.step_id == 8
    assert artifact.artifact_hash == hashlib.sha256(artifact.artifact_bytes).hexdigest()
    assert artifact.sha256 == artifact.artifact_hash
    assert artifact.artifact_metadata["artifact_id"] == 901
    assert artifact.artifact_metadata["strategy_branch"] == "OUTSIDER_ONLY"
    assert artifact.artifact_metadata["target_semantics"] == "FLIP_VS_FINAL_OUTCOME"
    assert artifact.artifact_metadata["loadability"]["exact_bundle_bytes"] is True


def test_logreg_contract_windows_ignore_none_and_parse_iso():
    rows = [
        SimpleNamespace(
            training_window_start=None,
            training_window_end=None,
        ),
        SimpleNamespace(
            training_window_start="2026-08-01T00:00:00Z",
            training_window_end="2026-08-02T00:00:00+00:00",
        ),
    ]
    train_window, oot_window = logreg_adapters._contract_windows(
        replace(_context(), input_payload={}),
        rows,
        artifact=None,
    )
    assert train_window == (
        datetime(2026, 8, 1, tzinfo=timezone.utc),
        datetime(2026, 8, 2, tzinfo=timezone.utc),
    )
    assert oot_window is None
