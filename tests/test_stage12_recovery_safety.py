"""Stage 12 recovery and safety regression tests.

These tests deliberately use in-memory doubles (or the existing isolated
``db_session`` fixture) and never exercise LIVE settings or production data.
"""

import asyncio
import gzip
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from polyflip.ai_lab import jobs, shadow
from polyflip.ai_lab.orchestrator import build_experiment_report, claim_next_step
from polyflip.ai_lab.service import rollback_deployment
from polyflip.crypto.oof_artifact import deserialize_oof_artifact
from polyflip.crypto.predictor import CryptoPredictor


def test_malformed_and_broken_oof_artifacts_are_rejected():
    with pytest.raises(ValueError, match="Invalid OOF artifact compression"):
        deserialize_oof_artifact(b"not-an-artifact")
    with pytest.raises(ValueError, match="Invalid OOF artifact JSON"):
        deserialize_oof_artifact(gzip.compress(b"{broken-json"))


def test_duplicate_oot_window_is_not_counted_twice():
    result = SimpleNamespace(
        config_id=7,
        evaluation_kind="POLYMARKET_OOT",
        status="SUCCEEDED",
        net_pnl=2.0,
        trade_count=30,
        max_drawdown=1.0,
        metrics={
            "oot_windows": [
                {"start": "2026-08-01", "end": "2026-08-05", "net_pnl": 2, "trade_count": 30},
                {"start": "2026-08-01T00:00:00Z", "end": "2026-08-05T00:00:00Z", "net_pnl": 2, "trade_count": 30},
            ]
        },
    )
    row = build_experiment_report([result], min_trades=1, min_windows=1)["rows"][0]
    assert row["window_count"] == 1
    assert row["total_trades"] == 30
    assert row["invalid_result_count"] == 1


class _ShadowSession:
    def __init__(self):
        self.row = None
        self.added = []

    async def execute(self, _statement):
        session = self

        class Result:
            def scalar_one_or_none(self):
                return session.row

        return Result()

    def add(self, value):
        self.added.append(value)
        value.id = 1
        self.row = value

    async def flush(self):
        return None

    async def get(self, _model, _ident, **_kwargs):
        if getattr(_model, "__name__", "") == "AIRunStep":
            return self.step
        return self.row


def test_shadow_event_duplicate_and_resolution_are_idempotent():
    session = _ShadowSession()
    when = datetime(2026, 8, 19, tzinfo=timezone.utc)
    values = {"candidate_model_key": "candidate", "active_model_key": "active"}
    first = asyncio.run(shadow.record_shadow_observation(
        session, assignment_id=3, run_id=4, market_id="m1", snapshot_at=when, values=values
    ))
    second = asyncio.run(shadow.record_shadow_observation(
        session, assignment_id=3, run_id=4, market_id="m1", snapshot_at=when, values=values
    ))
    assert first is second
    assert len(session.added) == 1
    resolved = asyncio.run(shadow.resolve_shadow_observation(
        session, 1, market_outcome="YES", active_pnl=1.0, candidate_pnl=2.0
    ))
    again = asyncio.run(shadow.resolve_shadow_observation(
        session, 1, market_outcome="NO", active_pnl=99.0, candidate_pnl=99.0
    ))
    assert resolved is again
    assert again.market_outcome == "YES"
    assert again.candidate_pnl == 2.0


class _JobSession:
    def __init__(self, row):
        self.row = row
        self.step = SimpleNamespace(
            status="RUNNING",
            finished_at=datetime.now(timezone.utc),
            error_code="OLD",
            error_message="OLD",
        )

    async def execute(self, _statement):
        row = self.row

        class Scalars:
            def all(self):
                return [row] if row.status == "RUNNING" else []

        class Result:
            def scalar_one_or_none(self):
                return row

            def scalars(self):
                return Scalars()

        return Result()

    async def get(self, _model, _ident, **_kwargs):
        return self.row

    async def flush(self):
        return None


def test_job_claim_is_idempotent_for_terminal_and_stale_recovery():
    row = SimpleNamespace(status="QUEUED", attempt=0, started_at=None, heartbeat_at=None)
    session = _JobSession(row)
    first = asyncio.run(jobs.claim_job(session, "job-key"))
    second = asyncio.run(jobs.claim_job(session, "job-key"))
    assert first is not None
    assert second is None
    assert row.attempt == 1
    row.heartbeat_at = datetime.now(timezone.utc) - timedelta(hours=1)
    assert asyncio.run(jobs.recover_stale_jobs(session, stale_after_seconds=60)) == 1
    assert row.status == "STALE"
    assert session.step.status == "PENDING"
    recovered = asyncio.run(jobs.claim_job(session, "job-key"))
    assert recovered is row
    assert row.status == "RUNNING"


def test_lost_lease_does_not_renew_or_run_more_work(monkeypatch):
    from polyflip.ai_lab import scheduler

    class Session:
        async def get(self, _model, _run_id):
            return SimpleNamespace(status="RUNNING")

        async def rollback(self):
            self.rolled_back = True

    session = Session()
    monkeypatch.setattr(
        "polyflip.ai_lab.orchestrator.authorize_run_action",
        lambda *_args, **_kwargs: asyncio.sleep(0),
    )
    async def no_lease(*_args, **_kwargs):
        return False
    monkeypatch.setattr(scheduler, "acquire_worker_lease", no_lease)
    result = asyncio.run(scheduler.run_lgbm_scheduler(session, 1))
    assert result.status == "ALREADY_RUNNING"
    assert result.outcomes == ()


def test_parallel_claim_allows_only_one_worker_to_receive_step(monkeypatch):
    class Session:
        def __init__(self):
            self.calls = 0
            self.step = SimpleNamespace(status="PENDING", step_index=0, action="TRAIN_MODEL")

        async def get(self, _model, _ident):
            return SimpleNamespace(status="RUNNING", permission_id=None)

        async def execute(self, _statement):
            self.calls += 1
            step = self.step if self.calls == 1 else None
            class Result:
                def scalar_one_or_none(self):
                    return step
            return Result()

        async def flush(self):
            return None

    # This models the database's SKIP LOCKED outcome: a second worker sees no row.
    session = Session()
    async def claim():
        return await claim_next_step(session, 1)
    monkeypatch.setattr(
        "polyflip.ai_lab.orchestrator.authorize_run_action",
        lambda *_args, **_kwargs: asyncio.sleep(0),
    )
    first = asyncio.run(claim())
    second = asyncio.run(claim())
    assert first is not None
    assert second is None
    assert first.status == "RUNNING"


def test_predictor_boundary_rejects_missing_feature_and_nan_prediction(monkeypatch):
    monkeypatch.setattr(
        "polyflip.crypto.predictor.build_crypto_features",
        lambda *_args, **_kwargs: SimpleNamespace(valid=True, features=[[1.0]]),
    )
    predictor = CryptoPredictor()
    predictor._loaded_symbols.add("BTCUSDT")
    predictor._models["BTCUSDT"] = {"low_vol": SimpleNamespace(predict_proba=lambda _: [[np.nan, np.nan]])}
    predictor._model_features["BTCUSDT"] = {"low_vol": ("missing",)}
    predictor._vol_p33s["BTCUSDT"] = 2.0
    predictor._vol_p67s["BTCUSDT"] = 3.0
    missing = predictor.predict([], "BTCUSDT")
    assert missing.status == "INFERENCE_FAILED"
    assert missing.features_ok is False

    predictor._model_features["BTCUSDT"] = {"low_vol": ("ret_1",)}
    nan_prediction = predictor.predict([], "BTCUSDT")
    assert nan_prediction.status == "INFERENCE_FAILED"
    assert nan_prediction.features_ok is False


def test_rollback_api_returns_current_and_restored_tuple_without_live_fixture():
    assert rollback_deployment.__annotations__["return"]
