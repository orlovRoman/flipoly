"""AI Lab adapters for saved LogReg candidates.

The adapter keeps LogReg training/evaluation on the same audited offline
boundary as LightGBM. Training always creates inactive ModelRegistry rows.
Polymarket-OOT replays the saved OOF artifact and never retrains a candidate.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.ai_lab.executor import (
    ACTION_TO_EVALUATION_KIND,
    AdapterResult,
    StepContext,
)
from polyflip.ai_lab.artifact_contracts import (
    TrainingRows,
    artifact_digest,
    artifact_metadata,
    bundle_bytes,
    dataset_fingerprint,
    datetime_window,
    mapping,
    resolve_window,
)
from polyflip.constants import resolve_binance_symbol
from polyflip.crypto.logreg_polymarket_backtest import (
    compute_logreg_polymarket_backtest,
)
from polyflip.crypto.oof_artifact import deserialize_oof_artifact
from polyflip.crypto.polymarket_backtest import aggregate_stored_polymarket_backtests
from polyflip.db.models import (
    AIModelArtifact,
    ExperimentResult,
    ModelRegistry,
    ModelRegistryOOFArtifact,
)
from polyflip.models.trainer import ModelTrainer


LOGREG_MODEL_FAMILIES = frozenset(
    {"LOGREG", "LOGISTIC", "LOGISTIC_REGRESSION", "LOGISTICREGRESSION"}
)
REGIMES = ("contested", "leaning", "decided")


def _mapping(value: Any) -> dict[str, Any]:
    return mapping(value)


def is_logreg_context(context: StepContext) -> bool:
    return str(context.model_family or "").strip().upper() in LOGREG_MODEL_FAMILIES


def _base_asset(context: StepContext) -> str:
    payload = _mapping(context.input_payload)
    scope = _mapping(context.scope)
    value = context.asset or payload.get("asset") or scope.get("asset")
    symbol = resolve_binance_symbol(str(value or ""))
    if not symbol:
        raise ValueError("AI Lab LogReg config must provide BTC/ETH/... asset")
    return symbol.removesuffix("USDT")


def _code_sha() -> str | None:
    value = os.getenv("POLYFLIP_BUILD_SHA")
    return value.strip()[:64] if value and value.strip() else None


def _dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return (
                parsed.replace(tzinfo=timezone.utc)
                if parsed.tzinfo is None
                else parsed
            )
        except ValueError:
            return None
    return None


def _window_bound(rows: Sequence[ModelRegistry], field: str, *, latest: bool) -> datetime | None:
    values = [
        parsed
        for row in rows
        if (parsed := _dt(getattr(row, field, None))) is not None
    ]
    if not values:
        return None
    return max(values) if latest else min(values)


def _fingerprint(rows: Sequence[ModelRegistry]) -> str | None:
    return dataset_fingerprint(rows)


def _row_metrics(row: ModelRegistry) -> dict[str, Any]:
    params = _mapping(row.training_params)
    return {
        "model_registry_id": int(row.id),
        "asset": str(row.asset),
        "version": int(row.version),
        "auc": float(row.accuracy) if row.accuracy is not None else None,
        "ece": float(row.ece) if row.ece is not None else None,
        "brier": (
            float(row.brier_score)
            if row.brier_score is not None
            else params.get("brier_score")
        ),
        "log_loss": params.get("log_loss"),
        "feature_set": params.get("feature_set_version"),
        "is_active": bool(row.is_active),
        "prediction_semantics": params.get("prediction_semantics"),
    }


def _contract_windows(
    context: StepContext,
    rows: Sequence[ModelRegistry],
    artifact: AIModelArtifact | None = None,
) -> tuple[tuple[Any, Any] | None, tuple[Any, Any] | None]:
    metadata = (
        _mapping(getattr(artifact, "artifact_metadata", None))
        if artifact is not None
        else {}
    )
    sources = (
        _mapping(context.input_payload),
        context.backtest_params,
        context.strategy_params,
        context.model_params,
        context.scope,
        metadata,
    )
    train_window = resolve_window(sources, "train")
    oot_window = resolve_window(sources, "oot")
    if train_window is None:
        train_window = (
            _window_bound(rows, "training_window_start", latest=False),
            _window_bound(rows, "training_window_end", latest=True),
        )
    return datetime_window(train_window), datetime_window(oot_window)


async def _training_rows(
    session: AsyncSession,
    context: StepContext,
) -> list[ModelRegistry]:
    result = (
        await session.execute(
            select(ExperimentResult)
            .where(
                ExperimentResult.run_id == context.run_id,
                ExperimentResult.config_id == context.config_id,
                ExperimentResult.evaluation_kind == "TRAIN",
                ExperimentResult.status == "SUCCEEDED",
            )
            .order_by(ExperimentResult.created_at.desc(), ExperimentResult.id.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if result is None or result.artifact_id is None:
        return []
    artifact = await session.get(AIModelArtifact, int(result.artifact_id))
    if artifact is None or artifact.loadability_status != "VALID":
        return []
    if (
        artifact.config_id != context.config_id
        or artifact.run_id != context.run_id
        or artifact.step_id is None
        or (
            getattr(result, "step_id", None) is not None
            and artifact.step_id != result.step_id
        )
    ):
        return []
    metadata = _mapping(artifact.artifact_metadata)
    ids = metadata.get("model_registry_ids")
    if not isinstance(ids, list) or not ids:
        return []
    rows = list(
        (
            await session.execute(
                select(ModelRegistry)
                .where(
                    ModelRegistry.model_type == "logreg",
                    ModelRegistry.id.in_([int(item) for item in ids]),
                )
                .order_by(ModelRegistry.id.asc())
            )
        ).scalars().all()
    )
    by_id = {int(row.id): row for row in rows}
    rows = TrainingRows([by_id[int(item)] for item in ids if int(item) in by_id])
    rows.artifact = artifact
    return rows


async def _create_bundle_artifact(
    session: AsyncSession,
    context: StepContext,
    rows: Sequence[ModelRegistry],
) -> AIModelArtifact:
    for row in rows:
        if row.model_blob is None:
            raise ValueError(f"ModelRegistry {row.id} has no model_blob")
    payload = bundle_bytes(
        rows,
        run_id=context.run_id,
        config_id=context.config_id,
        step_id=context.step_id,
        artifact_kind="LOGREG_REGISTRY_BUNDLE",
        target_semantics="FLIP_VS_FINAL_OUTCOME",
    )
    digest = artifact_digest(payload)
    input_payload = _mapping(context.input_payload)
    strategy = _mapping(context.strategy_params)
    strategy_branch = str(
        input_payload.get("strategy_branch")
        or strategy.get("strategy_branch")
        or strategy.get("branch")
        or "COMBINED"
    ).strip().upper()
    sources = (
        input_payload,
        context.backtest_params,
        context.strategy_params,
        context.model_params,
        context.scope,
    )
    train_window = resolve_window(sources, "train")
    oot_window = resolve_window(sources, "oot")
    if train_window is None:
        train_window = (
            min((row.training_window_start for row in rows if row.training_window_start), default=None),
            max((row.training_window_end for row in rows if row.training_window_end), default=None),
        )
    metadata = artifact_metadata(
        context=context,
        rows=rows,
        artifact_kind="LOGREG_REGISTRY_BUNDLE",
        feature_pipeline_version="LOGREG_FEATURES_V1",
        target_semantics="FLIP_VS_FINAL_OUTCOME",
        feature_semantics={
            "feature_set": context.feature_set,
            "features": [str(row.features or "") for row in rows],
        },
        train_window=train_window,
        oot_window=oot_window,
        strategy_branch=strategy_branch,
    )
    metadata.update(
        {
            "config_hash": context.config_hash,
            "asset": _base_asset(context),
            "model_versions": {str(row.asset): int(row.version) for row in rows},
        }
    )
    artifact = AIModelArtifact(
        config_id=context.config_id,
        run_id=context.run_id,
        step_id=context.step_id,
        model_registry_id=None,
        artifact_uri=f"db://ai-lab/{context.run_id}/{context.config_id}/{digest}",
        artifact_bytes=payload,
        artifact_hash=digest,
        sha256=digest,
        schema_version="1",
        feature_pipeline_version="LOGREG_FEATURES_V1",
        artifact_metadata=metadata,
        loadability_status="VALID",
    )
    session.add(artifact)
    await session.flush()
    metadata["artifact_id"] = int(artifact.id)
    metadata["provenance"]["artifact_id"] = int(artifact.id)
    artifact.artifact_metadata = metadata
    await session.flush()
    return artifact


async def train_logreg(context: StepContext, session: AsyncSession) -> AdapterResult:
    if not is_logreg_context(context):
        raise ValueError("AI Lab TRAIN_MODEL adapter supports LogReg configs only")
    asset = _base_asset(context)
    before = {
        int(value)
        for value in (
            await session.execute(
                select(ModelRegistry.id).where(
                    ModelRegistry.model_type == "logreg",
                    ModelRegistry.asset.in_([asset, *[f"{asset}_{phase}" for phase in REGIMES]]),
                )
            )
        ).scalars().all()
    }
    trained = await ModelTrainer(session).train(
        asset,
        save_settings=False,
        feature_set=context.feature_set or "AUTO",
        activate_after_train=False,
    )
    if not trained:
        return AdapterResult(
            evaluation_kind="TRAIN",
            status="INSUFFICIENT_DATA",
            summary=f"No LogReg candidate was produced for {asset}.",
            error_code="TRAINING_NO_CANDIDATE",
        )
    rows = list(
        (
            await session.execute(
                select(ModelRegistry)
                .where(
                    ModelRegistry.model_type == "logreg",
                    ModelRegistry.id.not_in(before),
                    ModelRegistry.asset.in_(
                        [asset, *[f"{asset}_{phase}" for phase in REGIMES]]
                    ),
                )
                .order_by(ModelRegistry.trained_at.desc(), ModelRegistry.id.desc())
            )
        ).scalars().all()
    )
    if not rows:
        return AdapterResult(
            evaluation_kind="TRAIN",
            status="INSUFFICIENT_DATA",
            summary=f"Trainer completed without new LogReg rows for {asset}.",
            error_code="TRAINING_NO_REGISTRY_ROWS",
        )
    if any(bool(row.is_active) for row in rows):
        # Do not rollback the executor-owned transaction.  Neutralize only the
        # newly created rows, then let the executor persist an auditable failure.
        for row in rows:
            if bool(row.is_active):
                row.is_active = False
                row.activation_source = None
                row.activated_at = None
                row.activated_by = None
        await session.flush()
        raise RuntimeError(
            "safety violation: AI Lab LogReg training produced an active model"
        )
    artifact = await _create_bundle_artifact(session, context, rows)
    train_window, oot_window = _contract_windows(context, rows, artifact)
    metrics = [_row_metrics(row) for row in rows]
    aucs = [item["auc"] for item in metrics if item["auc"] is not None]
    eces = [item["ece"] for item in metrics if item["ece"] is not None]
    return AdapterResult(
        evaluation_kind="TRAIN",
        status="SUCCEEDED",
        artifact_id=int(artifact.id),
        metrics={
            "model_count": len(rows),
            "model_registry_ids": [int(row.id) for row in rows],
            "median_auc": sorted(aucs)[len(aucs) // 2] if aucs else None,
            "median_ece": sorted(eces)[len(eces) // 2] if eces else None,
            "feature_set": context.feature_set,
        },
        slices={str(row.asset): _row_metrics(row) for row in rows},
        code_sha=_code_sha(),
        dataset_fingerprint=_fingerprint(rows),
        train_window_start=train_window[0] if train_window else None,
        train_window_end=train_window[1] if train_window else None,
        oot_window_start=oot_window[0] if oot_window else None,
        oot_window_end=oot_window[1] if oot_window else None,
        summary=(
            f"Saved {len(rows)} inactive LogReg candidate(s) for {asset}; "
            "no active model or RuntimeSettings was changed."
        ),
    )


async def run_logreg_oot(context: StepContext, session: AsyncSession) -> AdapterResult:
    if not is_logreg_context(context):
        raise ValueError("AI Lab OOT adapter supports LogReg configs only")
    rows = await _training_rows(session, context)
    if not rows:
        return AdapterResult(
            evaluation_kind="OOT",
            status="INSUFFICIENT_DATA",
            summary="No successful LogReg TRAIN artifact is available.",
            error_code="TRAIN_ARTIFACT_MISSING",
        )
    artifact = getattr(rows, "artifact", None)
    train_window, oot_window = _contract_windows(context, rows, artifact)
    metrics = [_row_metrics(row) for row in rows]
    aucs = [item["auc"] for item in metrics if item["auc"] is not None]
    eces = [item["ece"] for item in metrics if item["ece"] is not None]
    return AdapterResult(
        evaluation_kind="OOT",
        status="SUCCEEDED",
        artifact_id=int(artifact.id) if artifact is not None else None,
        metrics={
            "model_count": len(rows),
            "auc": sum(aucs) / len(aucs) if aucs else None,
            "ece": sum(eces) / len(eces) if eces else None,
        },
        slices={item["asset"]: item for item in metrics},
        code_sha=_code_sha(),
        dataset_fingerprint=_fingerprint(rows),
        train_window_start=train_window[0] if train_window else None,
        train_window_end=train_window[1] if train_window else None,
        oot_window_start=oot_window[0] if oot_window else None,
        oot_window_end=oot_window[1] if oot_window else None,
        summary="LogReg OOT diagnostics were read from saved registry candidates.",
    )


async def run_logreg_polymarket_oot(
    context: StepContext,
    session: AsyncSession,
) -> AdapterResult:
    if not is_logreg_context(context):
        raise ValueError("AI Lab Polymarket-OOT adapter supports LogReg configs only")
    rows = await _training_rows(session, context)
    if not rows:
        return AdapterResult(
            evaluation_kind="POLYMARKET_OOT",
            status="INSUFFICIENT_DATA",
            summary="No successful LogReg TRAIN artifact is available.",
            error_code="TRAIN_ARTIFACT_MISSING",
        )
    artifact = getattr(rows, "artifact", None)
    input_payload = _mapping(context.input_payload)
    strategy = _mapping(context.strategy_params)
    branch = str(
        input_payload.get("strategy_branch")
        or strategy.get("strategy_branch")
        or strategy.get("branch")
        or "COMBINED"
    ).strip().upper()
    if branch not in {"OUTSIDER_ONLY", "FAVORITE_ONLY", "COMBINED"}:
        raise ValueError("strategy_branch must be OUTSIDER_ONLY, FAVORITE_ONLY or COMBINED")
    results: list[dict[str, Any]] = []
    provenance: list[ModelRegistry] = []
    for row in rows:
        oof_artifact = (
            await session.execute(
                select(ModelRegistryOOFArtifact).where(
                    ModelRegistryOOFArtifact.model_registry_id == row.id
                )
            )
        ).scalar_one_or_none()
        if oof_artifact is None:
            continue
        params = _mapping(row.training_params)
        if params.get("prediction_semantics") != "FLIP_VS_FINAL_OUTCOME":
            continue
        try:
            oof_payload = deserialize_oof_artifact(bytes(oof_artifact.artifact_blob))
            result = compute_logreg_polymarket_backtest(
                oof_payload["frame"],
                oof_payload["oof_scores"],
                oof_payload["quotes"],
                strategy_branch=branch,
            )
        except (TypeError, ValueError) as exc:
            return AdapterResult(
                evaluation_kind="POLYMARKET_OOT",
                status="FAILED",
                summary=f"Saved LogReg artifact {row.id} is invalid: {exc}",
                error_code="POLYMARKET_OOT_INVALID_ARTIFACT",
                error_message=str(exc),
            )
        results.append(result)
        provenance.append(row)
    if not results:
        return AdapterResult(
            evaluation_kind="POLYMARKET_OOT",
            status="INSUFFICIENT_DATA",
            summary=f"No saved LogReg OOF artifact is available for {branch}.",
            error_code="POLYMARKET_OOT_MISSING",
        )
    summary = aggregate_stored_polymarket_backtests(
        results, strategy_branch=branch
    )
    train_window, oot_window = _contract_windows(context, rows, artifact)
    stored_windows = [
        window
        for result in results
        for window in (result.get("oot_windows") or [])
        if isinstance(window, Mapping)
    ]
    metrics = {
        key: summary.get(key)
        for key in (
            "strategy_branch",
            "n_markets",
            "n_quotes",
            "n_oof",
            "n_eligible",
            "n_trades",
            "net_profit",
            "roi_pct",
            "win_rate",
            "max_drawdown_usdc",
            "max_drawdown_pct",
            "coverage_pct",
        )
    }
    metrics["oot_windows"] = stored_windows
    return AdapterResult(
        evaluation_kind="POLYMARKET_OOT",
        status="SUCCEEDED" if summary["n_markets"] else "INSUFFICIENT_DATA",
        metrics=metrics,
        slices={"slices": summary.get("slices", []), "oot_windows": stored_windows},
        trade_count=int(summary.get("n_trades") or 0),
        net_pnl=float(summary.get("net_profit") or 0.0),
        max_drawdown=float(summary.get("max_drawdown_usdc") or 0.0),
        artifact_id=int(artifact.id) if artifact is not None else None,
        code_sha=_code_sha(),
        dataset_fingerprint=_fingerprint(provenance),
        train_window_start=train_window[0] if train_window else None,
        train_window_end=train_window[1] if train_window else None,
        oot_window_start=oot_window[0] if oot_window else None,
        oot_window_end=oot_window[1] if oot_window else None,
        summary=(
            f"LogReg Polymarket-OOT {branch}: "
            f"{int(summary.get('n_trades') or 0)} trades, "
            f"net PnL {float(summary.get('net_profit') or 0.0):.6f} USDC."
        ),
    )
