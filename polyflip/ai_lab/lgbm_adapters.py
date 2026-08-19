"""Safe LightGBM adapters for the autonomous AI Lab.

The adapters intentionally operate only on offline, canonical Polymarket data.
Training uses the existing CryptoModelTrainer with activation and RuntimeSettings
writes disabled. OOT actions replay immutable ModelRegistry metadata and never
retrain a model or call an execution gateway.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.ai_lab.executor import (
    ACTION_TO_EVALUATION_KIND,
    AdapterRegistry,
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
    resolve_windows,
)
from polyflip.crypto.experiment_configs import normalize_experiment_config
from polyflip.crypto.polymarket_backtest import (
    aggregate_stored_polymarket_backtests,
)
from polyflip.crypto.trainer import CryptoModelTrainer
from polyflip.constants import resolve_binance_symbol
from polyflip.ai_lab.logreg_adapters import LOGREG_MODEL_FAMILIES
from polyflip.db.models import (
    AIModelArtifact,
    ExperimentResult,
    ModelRegistry,
)

LIGHTGBM_MODEL_FAMILIES = frozenset({"LGBM", "LIGHTGBM", "CRYPTO_LGBM"})
REGIMES = ("low_vol", "mid_vol", "high_vol")
CANONICAL_TARGET_SOURCE = "POLYMARKET_FINAL_OUTCOME"
CANONICAL_BACKTEST_MODE = "POLYMARKET_OOF"
DEFAULT_FEATURE_PIPELINE_VERSION = "CRYPTO_FEATURES_V2"


def _json_mapping(value: Any) -> dict[str, Any]:
    return mapping(value)


def _model_family_ok(context: StepContext) -> bool:
    return str(context.model_family or "").strip().upper() in LIGHTGBM_MODEL_FAMILIES


def _canonical_symbol(context: StepContext) -> str:
    payload = _json_mapping(context.input_payload)
    scope = _json_mapping(context.scope)
    value = context.asset or payload.get("asset") or scope.get("asset")
    symbol = resolve_binance_symbol(str(value or ""))
    if not symbol:
        raise ValueError(
            "AI Lab LightGBM config must provide asset as BTC/ETH/... or SYMBOLUSDT"
        )
    return symbol


def _normalized_config(context: StepContext) -> dict[str, Any]:
    strategy = _json_mapping(context.strategy_params)
    calibration = _json_mapping(context.calibration_params)
    if not calibration:
        calibration = _json_mapping(strategy.get("calibration"))
    payload = _json_mapping(context.input_payload)
    if not calibration:
        calibration = _json_mapping(payload.get("calibration"))
    return normalize_experiment_config(
        {
            "feature_set": context.feature_set or "A",
            "model": _json_mapping(context.model_params),
            "calibration": calibration,
            "backtest": _json_mapping(context.backtest_params),
        }
    )


def _code_sha() -> str | None:
    value = os.getenv("POLYFLIP_BUILD_SHA")
    return value.strip()[:64] if value and value.strip() else None


def _dt(value: Any) -> datetime | None:
    return value if isinstance(value, datetime) else None


def _contract_windows(
    context: StepContext,
    rows: Sequence[ModelRegistry],
    artifact: AIModelArtifact | None = None,
) -> tuple[tuple[Any, Any] | None, tuple[Any, Any] | None]:
    metadata = (
        _json_mapping(getattr(artifact, "artifact_metadata", None))
        if artifact is not None
        else {}
    )
    sources = (
        _json_mapping(context.input_payload),
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
            min((_dt(row.training_window_start) for row in rows), default=None),
            max((_dt(row.training_window_end) for row in rows), default=None),
        )
    return datetime_window(train_window), datetime_window(oot_window)


def _fingerprint(rows: Sequence[ModelRegistry]) -> str | None:
    return dataset_fingerprint(rows)


def _row_metrics(row: ModelRegistry) -> dict[str, Any]:
    params = _json_mapping(row.training_params)
    return {
        "model_registry_id": int(row.id),
        "asset": str(row.asset),
        "version": int(row.version),
        "regime": str(row.asset).rsplit("_", 1)[-1],
        "auc": float(row.accuracy) if row.accuracy is not None else None,
        "ece": float(row.ece) if row.ece is not None else None,
        "brier": (
            float(row.brier_score) if row.brier_score is not None
            else params.get("brier_score")
        ),
        "log_loss": params.get("log_loss"),
        "train_samples": row.train_samples,
        "oot_samples": params.get("oot_samples", row.validation_samples),
        "dataset_fingerprint": row.dataset_fingerprint,
        "feature_set": params.get("feature_set"),
        "is_active": bool(row.is_active),
    }


def _deduplicate_rows(
    rows: Sequence[ModelRegistry],
    *,
    assets: Sequence[str] | None = None,
) -> list[ModelRegistry]:
    """Keep the newest row per asset while preserving requested asset order."""
    latest: dict[str, ModelRegistry] = {}
    for row in rows:
        latest.setdefault(str(row.asset), row)
    if assets is None:
        return list(latest.values())
    return [latest[asset] for asset in assets if asset in latest]


def _selected_regimes(context: StepContext) -> tuple[str, ...]:
    regime = str(context.regime or "").strip().lower()
    if not regime:
        return REGIMES
    if regime not in REGIMES:
        raise ValueError(f"unsupported LightGBM regime: {regime}")
    return (regime,)


async def _registry_rows(
    session: AsyncSession,
    context: StepContext,
    *,
    ids: Sequence[int] | None = None,
) -> list[ModelRegistry]:
    symbol = _canonical_symbol(context)
    assets = [f"{symbol}_{regime}" for regime in _selected_regimes(context)]
    stmt = (
        select(ModelRegistry)
        .where(
            ModelRegistry.model_type == "lgbm",
            ModelRegistry.asset.in_(assets),
        )
        .order_by(ModelRegistry.trained_at.desc(), ModelRegistry.id.desc())
    )
    if ids:
        stmt = stmt.where(ModelRegistry.id.in_([int(item) for item in ids]))
    rows = list((await session.execute(stmt)).scalars().all())
    if ids:
        by_id = {int(row.id): row for row in rows}
        return [by_id[item] for item in ids if item in by_id]
    return _deduplicate_rows(rows, assets=assets)


async def _training_rows(
    session: AsyncSession,
    context: StepContext,
) -> list[ModelRegistry]:
    """Load rows from this run/config's exact successful TRAIN artifact."""
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
    metadata = _json_mapping(artifact.artifact_metadata)
    ids = metadata.get("model_registry_ids")
    if not isinstance(ids, list) or not ids:
        return []
    rows = TrainingRows(
        await _registry_rows(session, context, ids=[int(item) for item in ids])
    )
    rows.artifact = artifact
    return rows


async def _create_bundle_artifact(
    session: AsyncSession,
    context: StepContext,
    rows: Sequence[ModelRegistry],
    config: Mapping[str, Any],
) -> AIModelArtifact:
    payload = bundle_bytes(
        rows,
        run_id=context.run_id,
        config_id=context.config_id,
        step_id=context.step_id,
        artifact_kind="LIGHTGBM_REGISTRY_BUNDLE",
        target_semantics=CANONICAL_TARGET_SOURCE,
    )
    digest = artifact_digest(payload)
    strategy = _json_mapping(context.strategy_params)
    input_payload = _json_mapping(context.input_payload)
    strategy_branch = str(
        input_payload.get("strategy_branch")
        or strategy.get("strategy_branch")
        or strategy.get("branch")
        or "COMBINED"
    ).strip().upper()
    sources = (
        input_payload,
        config,
        context.backtest_params,
        context.strategy_params,
        context.scope,
    )
    train_window = resolve_window(sources, "train")
    oot_window = resolve_window(sources, "oot")
    if train_window is None:
        train_window = (
            min((_dt(row.training_window_start) for row in rows), default=None),
            max((_dt(row.training_window_end) for row in rows), default=None),
        )
    metadata = artifact_metadata(
        context=context,
        rows=rows,
        artifact_kind="LIGHTGBM_REGISTRY_BUNDLE",
        feature_pipeline_version=(
            config.get("feature_pipeline_version") or DEFAULT_FEATURE_PIPELINE_VERSION
        ),
        target_semantics=CANONICAL_TARGET_SOURCE,
        feature_semantics={
            "feature_set": config["feature_set"],
            "feature_set_version": config["feature_set_version"],
            "features": [str(getattr(row, "features", "") or "") for row in rows],
        },
        train_window=train_window,
        oot_window=oot_window,
        strategy_branch=strategy_branch,
    )
    metadata.update(
        {
            "config_hash": context.config_hash,
            "symbol": _canonical_symbol(context),
            "model_versions": {str(row.asset): int(row.version) for row in rows},
        }
    )
    artifact = AIModelArtifact(
        config_id=context.config_id,
        run_id=context.run_id,
        step_id=context.step_id,
        model_registry_id=None,
        artifact_uri=(
            f"db://ai-lab/{context.run_id}/{context.config_id}/"
            f"{digest}"
        ),
        artifact_bytes=payload,
        artifact_hash=digest,
        sha256=digest,
        schema_version="1",
        feature_pipeline_version=(
            config.get("feature_pipeline_version")
            or DEFAULT_FEATURE_PIPELINE_VERSION
        ),
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


async def train_lgbm(context: StepContext, session: AsyncSession) -> AdapterResult:
    if not _model_family_ok(context):
        raise ValueError("AI Lab TRAIN_MODEL adapter supports LightGBM configs only")
    config = _normalized_config(context)
    symbol = _canonical_symbol(context)
    before = (
        await session.execute(
            select(ModelRegistry.id).where(
                ModelRegistry.model_type == "lgbm",
                ModelRegistry.asset.in_(
                    [f"{symbol}_{regime}" for regime in REGIMES]
                ),
            )
        )
    ).scalars().all()
    before_ids = {int(item) for item in before}

    trained = await CryptoModelTrainer(session).train(
        symbol,
        interval=context.interval or "15m",
        save_settings=False,
        feature_set=config["feature_set"],
        activate_after_train=False,
        experiment_config=config,
        # AIExperimentConfig and legacy LGBMExperimentConfig are separate
        # tables. The AI config remains in artifact metadata instead of being
        # misrepresented as a legacy FK.
        experiment_config_id=None,
    )
    if not trained:
        return AdapterResult(
            evaluation_kind=ACTION_TO_EVALUATION_KIND["TRAIN_MODEL"],
            status="INSUFFICIENT_DATA",
            summary=f"No LightGBM candidate was produced for {symbol}.",
            error_code="TRAINING_NO_CANDIDATE",
        )

    # CryptoModelTrainer currently produces all three volatility regimes in
    # one call. Bundle every newly created row so a regime-scoped config does
    # not leave two freshly trained inactive rows orphaned in ModelRegistry.
    rows_stmt = select(ModelRegistry).where(
        ModelRegistry.model_type == "lgbm",
        ModelRegistry.asset.in_(
            [f"{symbol}_{regime}" for regime in REGIMES]
        ),
    )
    if before_ids:
        rows_stmt = rows_stmt.where(ModelRegistry.id.notin_(before_ids))
    rows = list(
        (
            await session.execute(
                rows_stmt.order_by(ModelRegistry.trained_at.desc(), ModelRegistry.id.desc())
            )
        ).scalars().all()
    )
    if not rows:
        return AdapterResult(
            evaluation_kind="TRAIN",
            status="INSUFFICIENT_DATA",
            summary=f"Trainer completed without new registry rows for {symbol}.",
            error_code="TRAINING_NO_REGISTRY_ROWS",
        )
    if any(bool(row.is_active) for row in rows):
        # The trainer must not leave an active candidate in the surrounding
        # transaction. Roll back before propagating the safety violation so
        # the executor cannot commit the accidental activation later.
        await session.rollback()
        raise RuntimeError(
            "safety violation: AI Lab training produced an active LightGBM model"
        )
    rows = _deduplicate_rows(rows, assets=sorted({str(row.asset) for row in rows}))
    artifact = await _create_bundle_artifact(session, context, rows, config)
    train_window, oot_window = _contract_windows(context, rows, artifact)
    aucs = [float(row.accuracy) for row in rows if row.accuracy is not None]
    eces = [float(row.ece) for row in rows if row.ece is not None]
    return AdapterResult(
        evaluation_kind="TRAIN",
        status="SUCCEEDED",
        artifact_id=int(artifact.id),
        metrics={
            "model_count": len(rows),
            "model_registry_ids": [int(row.id) for row in rows],
            "median_auc": (
                sorted(aucs)[len(aucs) // 2] if aucs else None
            ),
            "median_ece": (
                sorted(eces)[len(eces) // 2] if eces else None
            ),
            "train_samples": sum(int(row.train_samples or 0) for row in rows),
            "feature_set": config["feature_set"],
        },
        slices={
            str(row.asset): _row_metrics(row) for row in rows
        },
        code_sha=_code_sha(),
        dataset_fingerprint=_fingerprint(rows),
        train_window_start=train_window[0] if train_window else None,
        train_window_end=train_window[1] if train_window else None,
        oot_window_start=oot_window[0] if oot_window else None,
        oot_window_end=oot_window[1] if oot_window else None,
        summary=(
            f"Saved {len(rows)} inactive LightGBM candidate(s) for {symbol}; "
            "no live settings or active model was changed."
        ),
    )


async def run_lgbm_oot(context: StepContext, session: AsyncSession) -> AdapterResult:
    if not _model_family_ok(context):
        raise ValueError("AI Lab OOT adapter supports LightGBM configs only")
    rows = await _training_rows(session, context)
    if not rows:
        return AdapterResult(
            evaluation_kind="OOT",
            status="INSUFFICIENT_DATA",
            summary="No successful LightGBM TRAIN artifact is available for this config.",
            error_code="TRAIN_ARTIFACT_MISSING",
        )
    artifact = getattr(rows, "artifact", None)
    train_window, oot_window = _contract_windows(context, rows, artifact)
    metrics = [_row_metrics(row) for row in rows]
    finite_auc = [item["auc"] for item in metrics if item["auc"] is not None]
    finite_ece = [item["ece"] for item in metrics if item["ece"] is not None]
    finite_brier = [item["brier"] for item in metrics if item["brier"] is not None]
    return AdapterResult(
        evaluation_kind="OOT",
        status="SUCCEEDED",
        artifact_id=int(artifact.id) if artifact is not None else None,
        metrics={
            "model_count": len(metrics),
            "auc": sum(finite_auc) / len(finite_auc) if finite_auc else None,
            "ece": sum(finite_ece) / len(finite_ece) if finite_ece else None,
            "brier": sum(finite_brier) / len(finite_brier) if finite_brier else None,
            "oot_samples": sum(int(item["oot_samples"] or 0) for item in metrics),
        },
        slices={item["asset"]: item for item in metrics},
        code_sha=_code_sha(),
        dataset_fingerprint=_fingerprint(rows),
        train_window_start=train_window[0] if train_window else None,
        train_window_end=train_window[1] if train_window else None,
        oot_window_start=oot_window[0] if oot_window else None,
        oot_window_end=oot_window[1] if oot_window else None,
        summary=(
            "OOT diagnostics were read from the saved ModelRegistry candidates; "
            "the models were not retrained."
        ),
    )


async def run_lgbm_polymarket_oot(
    context: StepContext,
    session: AsyncSession,
) -> AdapterResult:
    if not _model_family_ok(context):
        raise ValueError("AI Lab Polymarket-OOT adapter supports LightGBM configs only")
    rows = await _training_rows(session, context)
    if not rows:
        return AdapterResult(
            evaluation_kind="POLYMARKET_OOT",
            status="INSUFFICIENT_DATA",
            summary="No successful LightGBM TRAIN artifact is available for this config.",
            error_code="TRAIN_ARTIFACT_MISSING",
        )

    strategy = _json_mapping(context.strategy_params)
    payload = _json_mapping(context.input_payload)
    branch = str(
        payload.get("strategy_branch")
        or strategy.get("strategy_branch")
        or strategy.get("branch")
        or "COMBINED"
    ).strip().upper()
    if branch not in {"OUTSIDER_ONLY", "FAVORITE_ONLY", "COMBINED"}:
        raise ValueError("strategy_branch must be OUTSIDER_ONLY, FAVORITE_ONLY or COMBINED")

    regime_results: list[dict[str, Any]] = []
    provenance_rows: list[ModelRegistry] = []
    for row in rows:
        params = _json_mapping(row.training_params)
        if (
            params.get("target_source") != CANONICAL_TARGET_SOURCE
            or params.get("backtest_pnl_mode") != CANONICAL_BACKTEST_MODE
        ):
            continue
        variants = _json_mapping(params.get("backtest_variants"))
        value = variants.get(branch)
        if isinstance(value, dict):
            regime_results.append(value)
            provenance_rows.append(row)
        elif branch == "OUTSIDER_ONLY" and isinstance(params.get("backtest"), dict):
            # Backward-compatible read of a canonical candidate that predates
            # the variants map; never use legacy Binance PnL.
            regime_results.append(params["backtest"])
            provenance_rows.append(row)

    if not regime_results:
        return AdapterResult(
            evaluation_kind="POLYMARKET_OOT",
            status="INSUFFICIENT_DATA",
            summary=(
                f"No canonical Polymarket-OOT summary is stored for branch {branch}."
            ),
            error_code="POLYMARKET_OOT_MISSING",
        )

    summary = aggregate_stored_polymarket_backtests(
        regime_results,
        strategy_branch=branch,
    )
    if not summary.get("n_markets"):
        return AdapterResult(
            evaluation_kind="POLYMARKET_OOT",
            status="INSUFFICIENT_DATA",
            summary=f"Polymarket-OOT has no markets for branch {branch}.",
            error_code="POLYMARKET_OOT_EMPTY",
        )
    stored_windows = [
        window
        for result in regime_results
        for window in (result.get("oot_windows") or [])
        if isinstance(window, Mapping)
    ]
    stored_windows = [dict(window) for window in stored_windows]
    configured_windows = resolve_windows(
        (
            payload,
            context.backtest_params,
            context.strategy_params,
            context.model_params,
            context.scope,
        ),
        "oot",
    )
    for index, window in enumerate(stored_windows):
        if index < len(configured_windows):
            window.setdefault("start", configured_windows[index][0])
            window.setdefault("end", configured_windows[index][1])
    metrics = {
        key: summary.get(key)
        for key in (
            "strategy_branch",
            "n_markets",
            "n_quotes",
            "n_oof",
            "n_eligible",
            "roi_pct",
            "win_rate",
            "avg_edge",
            "avg_net_edge",
            "avg_entry_price",
            "coverage_pct",
            "sharpe_ratio",
            "profit_factor",
            "max_drawdown_usdc",
            "max_drawdown_pct",
        )
    }
    metrics["n_trades"] = int(summary.get("n_trades") or 0)
    metrics["net_pnl"] = float(summary.get("net_profit") or 0.0)
    metrics["oot_windows"] = stored_windows
    artifact = getattr(rows, "artifact", None)
    train_window, oot_window = _contract_windows(context, rows, artifact)
    return AdapterResult(
        evaluation_kind="POLYMARKET_OOT",
        status="SUCCEEDED",
        metrics=metrics,
        slices={"slices": summary.get("slices", [])},
        trade_count=int(summary.get("n_trades") or 0),
        net_pnl=float(summary.get("net_profit") or 0.0),
        max_drawdown=float(summary.get("max_drawdown_usdc") or 0.0),
        artifact_id=int(artifact.id) if artifact is not None else None,
        code_sha=_code_sha(),
        dataset_fingerprint=_fingerprint(provenance_rows),
        train_window_start=train_window[0] if train_window else None,
        train_window_end=train_window[1] if train_window else None,
        oot_window_start=oot_window[0] if oot_window else None,
        oot_window_end=oot_window[1] if oot_window else None,
        summary=(
            f"Polymarket-OOT {branch}: {int(summary.get('n_trades') or 0)} "
            f"trades, net PnL {float(summary.get('net_profit') or 0.0):.6f} USDC."
        ),
    )


def _is_logreg_context(context: StepContext) -> bool:
    return str(context.model_family or "").strip().upper() in LOGREG_MODEL_FAMILIES


async def _train_dispatch(
    context: StepContext, session: AsyncSession
) -> AdapterResult:
    if _is_logreg_context(context):
        from polyflip.ai_lab.logreg_adapters import train_logreg
        return await train_logreg(context, session)
    return await train_lgbm(context, session)


async def _oot_dispatch(
    context: StepContext, session: AsyncSession
) -> AdapterResult:
    if _is_logreg_context(context):
        from polyflip.ai_lab.logreg_adapters import run_logreg_oot
        return await run_logreg_oot(context, session)
    return await run_lgbm_oot(context, session)


async def _polymarket_oot_dispatch(
    context: StepContext, session: AsyncSession
) -> AdapterResult:
    if _is_logreg_context(context):
        from polyflip.ai_lab.logreg_adapters import run_logreg_polymarket_oot
        return await run_logreg_polymarket_oot(context, session)
    return await run_lgbm_polymarket_oot(context, session)


def build_lgbm_adapter_registry(session: AsyncSession) -> AdapterRegistry:
    """Create an explicit registry; no adapters are installed globally."""
    return (
        AdapterRegistry()
        .register("TRAIN_MODEL", lambda context: _train_dispatch(context, session))
        .register("RUN_OOT_BACKTEST", lambda context: _oot_dispatch(context, session))
        .register(
            "RUN_POLYMARKET_OOT",
            lambda context: _polymarket_oot_dispatch(context, session),
        )
    )
