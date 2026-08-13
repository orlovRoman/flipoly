# polyflip/api/crypto_dashboard.py
"""
Дашборд для крипто-домена (LightGBM Up/Down).
Полностью изолирован от Polymarket-дашборда.
Подключается в main.py одной строкой.

"""

from __future__ import annotations

import asyncio
import os

STATIC_VERSION = os.getenv("POLYFLIP_BUILD_SHA", "dev")
import json
import time
from typing import Literal, Any
from datetime import datetime, timezone
import numpy as np

import structlog
from fastapi import APIRouter, Body, Depends, Request, Query, HTTPException
from pydantic import BaseModel, Field
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, update, delete, func, cast, Numeric, text
from collections import defaultdict
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.api.auth import verify_api_key
import polyflip.constants as C
from polyflip.crypto.backtester import run_backtest
from polyflip.crypto.polymarket_backtest import aggregate_stored_polymarket_backtests, compute_oof_polymarket_backtest
from polyflip.crypto.feature_builder import build_features
from polyflip.crypto.candle_repository import get_recent_candles
from polyflip.crypto.trainer import CryptoModelTrainer
from polyflip.crypto.feature_sets import CONTROL_FEATURES, get_feature_set, normalize_feature_set, parse_feature_names, validate_feature_schema
from polyflip.db.connection import async_session, get_db_session
from polyflip.db.models import (
    ModelRegistry, ModelRegistryOOFArtifact, TradeHistory, RuntimeSettings, LGBMExperimentConfig,
    LGBMTrainingJob,
)
from polyflip.crypto.predictor import CryptoPredictor
from polyflip.crypto.oof_artifact import OOF_ARTIFACT_SCHEMA_VERSION, deserialize_oof_artifact
from polyflip.settings_registry import registry_defaults
from polyflip.crypto.experiment_configs import (
    normalize_experiment_config, experiment_config_hash,
)

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/lightgbm", tags=["LightGBM"])

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(base_dir, "templates"))

CRYPTO_SYMBOLS = ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "XRPUSDT", "SOLUSDT"]

# Кэш и активные процессы обучения
_cache: dict = {}
_CACHE_TTL = 10  # снизим до 10 секунд для лучшей отзывчивости настроек
_active_trainings: dict[str, dict] = {}
_OOF_BACKTEST_CACHE_TTL = 300


@router.get("")
async def crypto_page(request: Request, db: AsyncSession = Depends(get_db_session)):
    """HTML-страница крипто-дашборда."""
    defs = registry_defaults()
    api_key = request.cookies.get("api_key", "")

    row = (await db.execute(
        select(RuntimeSettings).where(RuntimeSettings.key == "ENABLE_ECE_CORRECTION")
    )).scalar_one_or_none()
    enable_ece = row.value.lower() in ("true", "1", "yes") if row else True

    return templates.TemplateResponse(
        request=request,
        name="crypto.html",
        context={
            "symbols": CRYPTO_SYMBOLS,
            "root_path": request.scope.get("root_path", ""),
            "api_key": api_key,
            "defaults": {
                "n_estimators": int(defs.get("CRYPTO_LGBM_N_ESTIMATORS", "300")),
                "learning_rate": float(defs.get("CRYPTO_LGBM_LEARNING_RATE", "0.05")),
                "num_leaves": int(defs.get("CRYPTO_LGBM_NUM_LEAVES", "31")),
                "max_depth": int(defs.get("CRYPTO_LGBM_MAX_DEPTH", "5")),
                "min_child_samples": int(
                    defs.get("CRYPTO_LGBM_MIN_CHILD_SAMPLES", "20")
                ),
                "subsample": float(defs.get("CRYPTO_LGBM_SUBSAMPLE", "0.8")),
                "colsample_bytree": float(
                    defs.get("CRYPTO_LGBM_COLSAMPLE_BYTREE", "0.8")
                ),
                "reg_alpha": float(defs.get("CRYPTO_LGBM_REG_ALPHA", "0.1")),
                "reg_lambda": float(defs.get("CRYPTO_LGBM_REG_LAMBDA", "1.0")),
                "n_jobs": int(defs.get("CRYPTO_LGBM_N_JOBS", "2")),
                "early_stopping_rounds": int(defs.get("CRYPTO_LGBM_EARLY_STOPPING_ROUNDS", "30")),
                "search_trials": int(defs.get("CRYPTO_LGBM_HYPERPARAM_SEARCH_TRIALS", "1")),
                "min_edge": float(defs.get("BACKTEST_MIN_EDGE", "0.04")),
                "enable_ece_correction": enable_ece,
            },
        },
    )
def round_optional(value, digits=4):
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


async def _persisted_training_states(db: AsyncSession) -> dict[str, dict]:
    """Return durable training state so it survives an API restart."""
    rows = (
        await db.execute(
            select(LGBMTrainingJob)
            .order_by(LGBMTrainingJob.created_at.desc(), LGBMTrainingJob.id.desc())
            .limit(100)
        )
    ).scalars().all()
    states: dict[str, dict] = {}
    for row in rows:
        if row.symbol in states:
            continue
        state = {
            "job_id": row.id,
            "status": "training" if row.status in {"QUEUED", "RUNNING"} else row.status.lower(),
            "symbol": row.symbol,
            "feature_set": row.feature_set,
            "activate_after_train": bool(row.activate_after_train),
            "experiment_config_id": row.experiment_config_id,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "started_at": row.started_at.isoformat() if row.started_at else None,
            "finished_at": row.finished_at.isoformat() if row.finished_at else None,
        }
        if row.error:
            state["error"] = row.error
        if getattr(row, "error_traceback", None):
            state["error_traceback"] = row.error_traceback
        states[row.symbol] = state
    return states

@router.get("/api/status", dependencies=[Depends(verify_api_key)])
async def crypto_status(db: AsyncSession = Depends(get_db_session)):
    """
    Возвращает текущее состояние крипто-моделей:
    версию, AUC, ECE, порог, список фич, дату обучения, важность фичей и гиперпараметры.
    """
    now = time.time()
    persisted_trainings = await _persisted_training_states(db)
    if "status" in _cache and now - _cache["status"]["ts"] < _CACHE_TTL:
        res = dict(_cache["status"]["data"])
        res["active_trainings"] = persisted_trainings or _active_trainings
        return res

    allowed_assets = []
    for s in CRYPTO_SYMBOLS:
        allowed_assets.extend([f"{s}_low_vol", f"{s}_mid_vol", f"{s}_high_vol", s])

    stmt = (
        select(
            ModelRegistry.asset,
            ModelRegistry.version,
            ModelRegistry.is_active,
            ModelRegistry.accuracy,
            ModelRegistry.baseline,
            ModelRegistry.ece,
            ModelRegistry.features,
            ModelRegistry.trained_at,
            ModelRegistry.quality_gate_passed,
            ModelRegistry.quality_gate_reasons,
            ModelRegistry.activation_source,
            ModelRegistry.quality_override,
            ModelRegistry.backtest_pnl,
            ModelRegistry.backtest_trades,
            ModelRegistry.backtest_wr,
            ModelRegistry.activated_at,
            ModelRegistry.activation_reason,
            ModelRegistry.training_params,
            ModelRegistry.feature_importance,
            ModelRegistry.precision_at_threshold,
            ModelRegistry.recall_at_threshold,
            ModelRegistry.f1_at_threshold,
            ModelRegistry.brier_score,
            ModelRegistry.decision_threshold,
            ModelRegistry.decision_threshold_down,
        )
        .where(ModelRegistry.asset.in_(allowed_assets))
        .order_by(ModelRegistry.asset, ModelRegistry.version.desc())
    )
    rows = (await db.execute(stmt)).all()

    # Важность признаков из RuntimeSettings
    fi_keys = [f"CRYPTO_FI_{a}" for a in allowed_assets]
    fi_stmt = select(RuntimeSettings).where(RuntimeSettings.key.in_(fi_keys))
    fi_rows = (await db.execute(fi_stmt)).scalars().all()
    feature_importances = {}
    for r in fi_rows:
        try:
            sym = r.key.replace("CRYPTO_FI_", "")
            feature_importances[sym] = json.loads(r.value)
        except Exception:
            pass

    # Текущие гиперпараметры обучения из БД
    settings_keys = [
        "CRYPTO_LGBM_N_ESTIMATORS",
        "CRYPTO_LGBM_LEARNING_RATE",
        "CRYPTO_LGBM_NUM_LEAVES",
        "CRYPTO_LGBM_MAX_DEPTH",
        "CRYPTO_LGBM_MIN_CHILD_SAMPLES",
        "CRYPTO_LGBM_SUBSAMPLE",
        "CRYPTO_LGBM_COLSAMPLE_BYTREE",
        "CRYPTO_LGBM_REG_ALPHA",
        "CRYPTO_LGBM_REG_LAMBDA",
        "CRYPTO_LGBM_N_JOBS",
        "CRYPTO_LGBM_EARLY_STOPPING_ROUNDS",
        "CRYPTO_LGBM_HYPERPARAM_SEARCH_TRIALS",
        "BACKTEST_MIN_EDGE",
        "LGBM_EPSILON_QUANTILE",
        "ENABLE_ECE_CORRECTION",
    ]
    set_stmt = select(RuntimeSettings).where(RuntimeSettings.key.in_(settings_keys))
    set_rows = (await db.execute(set_stmt)).scalars().all()
    db_settings = {r.key: r.value for r in set_rows}

    defs = registry_defaults()

    def _safe_int(key, default):
        val = db_settings.get(key, defs.get(key, default))
        if val == "None" or val is None:
            return int(default)
        try:
            return int(val)
        except (ValueError, TypeError):
            return int(default)

    def _safe_float(key, default):
        val = db_settings.get(key, defs.get(key, default))
        if val == "None" or val is None:
            return float(default)
        try:
            return float(val)
        except (ValueError, TypeError):
            return float(default)

    active_settings = {
        "n_estimators": _safe_int("CRYPTO_LGBM_N_ESTIMATORS", "300"),
        "learning_rate": _safe_float("CRYPTO_LGBM_LEARNING_RATE", "0.05"),
        "num_leaves": _safe_int("CRYPTO_LGBM_NUM_LEAVES", "31"),
        "max_depth": _safe_int("CRYPTO_LGBM_MAX_DEPTH", "5"),
        "min_child_samples": _safe_int("CRYPTO_LGBM_MIN_CHILD_SAMPLES", "20"),
        "subsample": _safe_float("CRYPTO_LGBM_SUBSAMPLE", "0.8"),
        "colsample_bytree": _safe_float("CRYPTO_LGBM_COLSAMPLE_BYTREE", "0.8"),
        "reg_alpha": _safe_float("CRYPTO_LGBM_REG_ALPHA", "0.1"),
        "reg_lambda": _safe_float("CRYPTO_LGBM_REG_LAMBDA", "1.0"),
        "min_edge": _safe_float("BACKTEST_MIN_EDGE", "0.04"),
        "epsilon_quantile": _safe_float("LGBM_EPSILON_QUANTILE", "0.6"),
        "enable_ece_correction": db_settings.get("ENABLE_ECE_CORRECTION", "true").lower() in ("true", "1", "yes"),
        "n_jobs": _safe_int("CRYPTO_LGBM_N_JOBS", "2"),
        "early_stopping_rounds": _safe_int("CRYPTO_LGBM_EARLY_STOPPING_ROUNDS", "30"),
        "search_trials": _safe_int("CRYPTO_LGBM_HYPERPARAM_SEARCH_TRIALS", "1"),
    }

    models_info = {}
    for m in rows:
        key = f"{m.asset}_v{m.version}"
        try:
            models_info[key] = {
                "asset": m.asset,
                "version": m.version,
                "is_active": m.is_active,
                "auc": round_optional(m.accuracy),
                "baseline": round_optional(m.baseline),
                "ece": round_optional(m.ece) if getattr(m, "ece", None) else None,
                "threshold": round_optional(m.decision_threshold),
                "threshold_down": round_optional(m.decision_threshold_down),
                "features": m.features.split(",") if getattr(m, "features", None) else list(CONTROL_FEATURES),
                "feature_set": (m.training_params or {}).get("feature_set", "A"),
                "feature_set_version": (m.training_params or {}).get("feature_set_version", "legacy"),
                "comparison_key": (m.training_params or {}).get("comparison_key"),
                "validation_scheme": (m.training_params or {}).get("validation_scheme"),
                "oot_samples": (m.training_params or {}).get("oot_samples"),
                "oot_markets": (m.training_params or {}).get("oot_markets"),
                "log_loss": (m.training_params or {}).get("log_loss"),
                "feature_audit": (m.training_params or {}).get("feature_audit", {}),
                "feature_audit_summary": (m.training_params or {}).get("feature_audit_summary", {}),
                "target_source": (m.training_params or {}).get("target_source"),
                "is_loadable": (m.training_params or {}).get("target_source") == "POLYMARKET_FINAL_OUTCOME",
                "loadability_reason": (None if (m.training_params or {}).get("target_source") == "POLYMARKET_FINAL_OUTCOME" else "NON_CANONICAL_TARGET"),
                "trained_at": (
                    m.trained_at.isoformat() if getattr(m, "trained_at", None) else None
                ),
                "feature_importance": m.feature_importance or feature_importances.get(m.asset, {}),
                # Аудит Quality Gate и активации
                "quality_gate_passed": m.quality_gate_passed,
                "quality_gate_reasons": m.quality_gate_reasons,
                "activation_source": m.activation_source,
                "quality_override": getattr(m, "quality_override", None),
                "activated_at": (
                    m.activated_at.isoformat() if getattr(m, "activated_at", None) else None
                ),
                "activation_reason": getattr(m, "activation_reason", None),
                # Precision / Recall / F1 / Brier из реестра
                "precision": round_optional(m.precision_at_threshold) if getattr(m, "precision_at_threshold", None) is not None else None,
                "recall": round_optional(m.recall_at_threshold) if getattr(m, "recall_at_threshold", None) is not None else None,
                "f1": round_optional(m.f1_at_threshold) if getattr(m, "f1_at_threshold", None) is not None else None,
                "brier_score": round_optional(m.brier_score) if getattr(m, "brier_score", None) is not None else None,
                "backtest_pnl": round_optional(getattr(m, "backtest_pnl", None), 6),
                "backtest_trades": int(m.backtest_trades) if getattr(m, "backtest_trades", None) is not None else 0,
                "backtest_wr": round_optional(getattr(m, "backtest_wr", None), 6),
                "backtest_pnl_mode": (m.training_params or {}).get("backtest_pnl_mode"),
            }
        except Exception as e:
            logger.error("crypto_status_model_parse_error", key=key, error=str(e))

    result = {
        "models": models_info,
        "symbols": CRYPTO_SYMBOLS,
        "settings": active_settings,
        "active_trainings": persisted_trainings or _active_trainings,
        "feature_importances": {
            asset: feature_importances.get(asset, {})
            for asset in set(m.asset for m in rows if m.is_active)
        },
    }
    _cache["status"] = {"ts": now, "data": result}
    return result


class LightGBMDecisionModeRequest(BaseModel):
    mode: Literal["OFF", "SHADOW", "ACTIVE"]
    reason: str | None = None


@router.get("/api/lightgbm-decision-mode", dependencies=[Depends(verify_api_key)])
async def get_lightgbm_decision_mode(
    db: AsyncSession = Depends(get_db_session),
):
    row = (await db.execute(
        select(RuntimeSettings).where(RuntimeSettings.key == "LIGHTGBM_DECISION_MODE")
    )).scalar_one_or_none()

    mode = row.value.upper() if (row and row.value) else "SHADOW"
    if mode not in {"OFF", "SHADOW", "ACTIVE"}:
        mode = "SHADOW"

    return {
        "mode": mode,
        "affects_trading": mode == "ACTIVE",
        "paper_uses_logreg_only": mode != "ACTIVE",
        "live_uses_logreg_only": mode != "ACTIVE",
        "updated_at": row.updated_at.isoformat() if (row and row.updated_at) else None,
        "updated_by": row.updated_by if row else None,
    }


@router.patch("/api/lightgbm-decision-mode", dependencies=[Depends(verify_api_key)])
async def set_lightgbm_decision_mode(
    payload: LightGBMDecisionModeRequest,
    db: AsyncSession = Depends(get_db_session),
):
    now = datetime.now(timezone.utc)

    row = (await db.execute(
        select(RuntimeSettings)
        .where(RuntimeSettings.key == "LIGHTGBM_DECISION_MODE")
        .with_for_update()
    )).scalar_one_or_none()

    if row:
        old_mode = row.value
        row.value = payload.mode
        row.updated_at = now
        row.updated_by = "crypto_dashboard_ui"
    else:
        old_mode = "SHADOW"
        db.add(RuntimeSettings(
            key="LIGHTGBM_DECISION_MODE",
            value=payload.mode,
            updated_at=now,
            updated_by="crypto_dashboard_ui",
        ))

    await db.commit()

    logger.warning(
        "lightgbm_decision_mode_changed",
        old_mode=old_mode,
        new_mode=payload.mode,
        reason=payload.reason,
    )

    return {
        "status": "success",
        "old_mode": old_mode,
        "mode": payload.mode,
        "effective_immediately": True,
    }


@router.post("/api/settings", dependencies=[Depends(verify_api_key)])
async def save_crypto_settings(
    settings: dict, db: AsyncSession = Depends(get_db_session)
):
    """Сохраняет измененные гиперпараметры в RuntimeSettings."""
    now = datetime.now(timezone.utc)
    keys_map = {
        "n_estimators": "CRYPTO_LGBM_N_ESTIMATORS",
        "learning_rate": "CRYPTO_LGBM_LEARNING_RATE",
        "num_leaves": "CRYPTO_LGBM_NUM_LEAVES",
        "max_depth": "CRYPTO_LGBM_MAX_DEPTH",
        "min_child_samples": "CRYPTO_LGBM_MIN_CHILD_SAMPLES",
        "subsample": "CRYPTO_LGBM_SUBSAMPLE",
        "colsample_bytree": "CRYPTO_LGBM_COLSAMPLE_BYTREE",
        "reg_alpha": "CRYPTO_LGBM_REG_ALPHA",
        "reg_lambda": "CRYPTO_LGBM_REG_LAMBDA",
        "min_edge": "BACKTEST_MIN_EDGE",
        "epsilon_quantile": "LGBM_EPSILON_QUANTILE",
        "enable_ece_correction": "ENABLE_ECE_CORRECTION",
        "n_jobs": "CRYPTO_LGBM_N_JOBS",
        "early_stopping_rounds": "CRYPTO_LGBM_EARLY_STOPPING_ROUNDS",
        "search_trials": "CRYPTO_LGBM_HYPERPARAM_SEARCH_TRIALS",
    }

    for key, db_key in keys_map.items():
        if key in settings:
            val_str = str(settings[key]).lower() if isinstance(settings[key], bool) else str(settings[key])
            row = (
                await db.execute(
                    select(RuntimeSettings).where(RuntimeSettings.key == db_key)
                )
            ).scalar_one_or_none()
            if row:
                row.value = val_str
                row.updated_at = now
                row.updated_by = "crypto_dashboard_ui"
            else:
                db.add(
                    RuntimeSettings(
                        key=db_key,
                        value=val_str,
                        updated_at=now,
                        updated_by="crypto_dashboard_ui",
                    )
                )

    await db.commit()
    _cache.pop("status", None)
    return {"status": "success", "message": "Настройки успешно сохранены!"}


class ExperimentConfigRequest(BaseModel):
    name: str
    description: str | None = None
    asset: str | None = None
    volatility_regime: str | None = None
    feature_set: str = "A"
    model: dict[str, Any] = Field(default_factory=dict)
    calibration: dict[str, Any] = Field(default_factory=dict)
    backtest: dict[str, Any] = Field(default_factory=dict)
    parent_id: int | None = None
    created_by: str = "dashboard"


def _experiment_config_response(row: LGBMExperimentConfig) -> dict[str, Any]:
    return {
        "id": row.id,
        "name": row.name,
        "description": row.description,
        "asset": row.asset,
        "volatility_regime": row.volatility_regime,
        "feature_set": row.feature_set,
        "feature_set_version": row.feature_set_version,
        "model": row.model_params,
        "calibration": row.calibration_params,
        "backtest": row.backtest_params,
        "config_hash": row.config_hash,
        "parent_id": row.parent_id,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "created_by": row.created_by,
        "is_archived": bool(row.is_archived),
    }


async def _list_experiment_configs(
    db: AsyncSession,
    asset: str | None = None,
    include_archived: bool = False,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    stmt = select(LGBMExperimentConfig)
    if asset:
        stmt = stmt.where(LGBMExperimentConfig.asset == asset.strip().upper())
    if not include_archived:
        stmt = stmt.where(LGBMExperimentConfig.is_archived.is_(False))
    stmt = stmt.order_by(LGBMExperimentConfig.created_at.desc(), LGBMExperimentConfig.id.desc())
    rows = (await db.execute(stmt.limit(limit).offset(offset))).scalars().all()
    return {
        "configs": [_experiment_config_response(row) for row in rows],
        "limit": limit,
        "offset": offset,
        "has_more": len(rows) == limit,
    }


@router.get("/api/experiment-configs", dependencies=[Depends(verify_api_key)])
async def list_experiment_configs(
    asset: str | None = None,
    include_archived: bool = False,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db_session),
):
    return await _list_experiment_configs(db, asset, include_archived, limit, offset)


@router.post("/api/experiment-configs", dependencies=[Depends(verify_api_key)])
async def create_experiment_config(
    payload: ExperimentConfigRequest,
    db: AsyncSession = Depends(get_db_session),
):
    if not payload.name.strip():
        raise HTTPException(status_code=422, detail="name must not be empty")
    if payload.asset and payload.asset.strip().upper() not in CRYPTO_SYMBOLS:
        raise HTTPException(status_code=422, detail="asset must be a supported crypto symbol")
    try:
        config = normalize_experiment_config({
            "feature_set": payload.feature_set,
            "model": payload.model,
            "calibration": payload.calibration,
            "backtest": payload.backtest,
        })
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    parent = None
    if payload.parent_id is not None:
        parent = await db.get(LGBMExperimentConfig, payload.parent_id)
        if parent is None:
            raise HTTPException(status_code=404, detail="parent experiment config not found")
    row = LGBMExperimentConfig(
        name=payload.name.strip()[:128],
        description=payload.description,
        asset=payload.asset.strip().upper() if payload.asset else None,
        volatility_regime=payload.volatility_regime.strip().lower() if payload.volatility_regime else None,
        feature_set=config["feature_set"],
        feature_set_version=config["feature_set_version"],
        model_params=config["model"],
        calibration_params=config["calibration"],
        backtest_params=config["backtest"],
        config_hash=experiment_config_hash(config),
        parent_id=payload.parent_id,
        created_at=datetime.now(timezone.utc),
        created_by=payload.created_by.strip()[:128] or "dashboard",
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return {"status": "created", "config": _experiment_config_response(row)}


class CopyExperimentConfigRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    created_by: str = Field(default="dashboard", max_length=128)


@router.post("/api/experiment-configs/{config_id}/copy", dependencies=[Depends(verify_api_key)])
async def copy_experiment_config(
    config_id: int,
    payload: CopyExperimentConfigRequest,
    db: AsyncSession = Depends(get_db_session),
):
    source = await db.get(LGBMExperimentConfig, config_id)
    if source is None or source.is_archived:
        raise HTTPException(status_code=404, detail="experiment config not found")
    config = normalize_experiment_config({
        "feature_set": source.feature_set,
        "model": source.model_params,
        "calibration": source.calibration_params,
        "backtest": source.backtest_params,
    })
    row = LGBMExperimentConfig(
        name=payload.name.strip(), description=source.description, asset=source.asset,
        volatility_regime=source.volatility_regime,
        feature_set=config["feature_set"], feature_set_version=config["feature_set_version"],
        model_params=config["model"], calibration_params=config["calibration"],
        backtest_params=config["backtest"], config_hash=experiment_config_hash(config),
        parent_id=source.id, created_at=datetime.now(timezone.utc),
        created_by=payload.created_by.strip() or "dashboard",
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return {"status": "created", "config": _experiment_config_response(row)}

class SetEceCorrectionRequest(BaseModel):
    enabled: bool

@router.get("/api/enable-ece-correction", dependencies=[Depends(verify_api_key)])
async def get_ece_correction_status(db: AsyncSession = Depends(get_db_session)):
    row = (await db.execute(
        select(RuntimeSettings).where(RuntimeSettings.key == "ENABLE_ECE_CORRECTION")
    )).scalar_one_or_none()
    enabled = row.value.lower() in ("true", "1", "yes") if row else True
    return {"enabled": enabled}

@router.patch("/api/enable-ece-correction", dependencies=[Depends(verify_api_key)])
async def set_ece_correction_status(
    payload: SetEceCorrectionRequest,
    db: AsyncSession = Depends(get_db_session)
):
    now = datetime.now(timezone.utc)
    val_str = "true" if payload.enabled else "false"
    row = (await db.execute(
        select(RuntimeSettings).where(RuntimeSettings.key == "ENABLE_ECE_CORRECTION").with_for_update()
    )).scalar_one_or_none()

    if row:
        row.value = val_str
        row.updated_at = now
        row.updated_by = "crypto_dashboard_ui"
    else:
        db.add(RuntimeSettings(
            key="ENABLE_ECE_CORRECTION",
            value=val_str,
            updated_at=now,
            updated_by="crypto_dashboard_ui",
        ))

    await db.commit()
    return {"status": "success", "enabled": payload.enabled}


def _backtest_options_for_model(params: dict[str, Any] | None) -> dict[str, Any]:
    """Return the immutable backtest options persisted with a model."""
    params = params or {}
    raw = params.get("backtest_config")
    if raw is None:
        experiment = params.get("experiment_config")
        raw = experiment.get("backtest") if isinstance(experiment, dict) else None
    if raw is None:
        return {}
    try:
        return normalize_experiment_config({"backtest": raw})["backtest"]
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"error": "INVALID_BACKTEST_CONFIG", "message": str(exc)}) from exc


async def _stored_lgbm_polymarket_backtest(
    db: AsyncSession,
    *,
    symbol: str,
    feature_set: str,
    strategy_branch: str,
) -> dict:
    """Return the OOF Polymarket PnL persisted by the latest A/B/C run."""
    branch = strategy_branch.strip().upper()
    if branch not in {"OUTSIDER_ONLY", "FAVORITE_ONLY", "COMBINED"}:
        raise HTTPException(
            status_code=422,
            detail="strategy_branch must be OUTSIDER_ONLY, FAVORITE_ONLY or COMBINED",
        )
    assets = [f"{symbol}_{regime}" for regime in ("low_vol", "mid_vol", "high_vol")]
    rows = (
        await db.execute(
            select(ModelRegistry)
            .where(
                ModelRegistry.asset.in_(assets),
                ModelRegistry.model_type == "lgbm",
            )
            .order_by(ModelRegistry.asset, ModelRegistry.version.desc())
        )
    ).scalars().all()
    latest: dict[str, ModelRegistry] = {}
    for row in rows:
        params = row.training_params or {}
        if params.get("target_source") != "POLYMARKET_FINAL_OUTCOME":
            continue
        if params.get("feature_set", "A") != feature_set:
            continue
        latest.setdefault(row.asset, row)

    regime_results: list[dict] = []
    regime_payload: dict[str, dict] = {}
    artifact_ids = [row.id for row in latest.values()]
    artifact_rows = (
        await db.execute(
            select(ModelRegistryOOFArtifact).where(
                ModelRegistryOOFArtifact.model_registry_id.in_(artifact_ids),
                ModelRegistryOOFArtifact.schema_version == OOF_ARTIFACT_SCHEMA_VERSION,
            )
        )
    ).scalars().all() if artifact_ids else []
    artifacts = {artifact.model_registry_id: artifact for artifact in artifact_rows}
    for asset, row in latest.items():
        params = row.training_params or {}
        variants = params.get("backtest_variants") or {}
        variant = None
        artifact = artifacts.get(row.id)
        if artifact is not None:
            config_options = _backtest_options_for_model(params)
            config_hash = experiment_config_hash({"backtest": config_options})[:12] if config_options else "runtime"
            cache_key = f"lgbm_oof_{row.id}_{branch}_{config_hash}"
            cached = _cache.get(cache_key)
            if cached and time.time() - cached["ts"] < _OOF_BACKTEST_CACHE_TTL:
                variant = cached["data"]
            else:
                try:
                    payload = deserialize_oof_artifact(artifact.artifact_blob)
                    computed = compute_oof_polymarket_backtest(
                        payload["frame"], payload["oof_scores"], payload["quotes"],
                        strategy_branch=branch,
                        **config_options,
                    )
                    variant = {key: value for key, value in computed.items() if key != "trades"}
                    _cache[cache_key] = {"data": variant, "ts": time.time()}
                except ValueError as exc:
                    raise HTTPException(
                        status_code=422,
                        detail={"error": "OOF_ARTIFACT_INVALID", "model_id": row.id, "message": str(exc)},
                    ) from exc
        if variant is None:
            variant = variants.get(branch)
        if variant is None and branch == "OUTSIDER_ONLY":
            variant = params.get("backtest")
        if not isinstance(variant, dict) or not variant:
            continue
        regime_results.append(variant)
        regime_payload[asset] = {
            **variant,
            "version": row.version,
            "feature_set": feature_set,
            "model_id": row.id,
            "artifact_available": artifact is not None,
        }

    if not regime_results:
        raise HTTPException(
            status_code=404,
            detail={
                "error": (
                    f"No saved OOF Polymarket backtest for {symbol}, "
                    f"feature_set={feature_set}, branch={branch}. "
                    "Retrain the model with OOF metrics enabled."
                ),
                "symbol": symbol,
                "feature_set": feature_set,
                "strategy_branch": branch,
            },
        )

    summary = aggregate_stored_polymarket_backtests(
        regime_results, strategy_branch=branch
    )
    return {
        "symbol": symbol,
        "interval": "15m",
        "pnl_mode": "POLYMARKET_OOF",
        "feature_set": feature_set,
        "strategy_branch": branch,
        "n_markets": summary["n_markets"],
        "n_quotes": summary["n_quotes"],
        "n_oof": summary["n_oof"],
        "n_eligible": summary["n_eligible"],
        "n_trades": summary["n_trades"],
        "stake_usdc": summary["stake_usdc"],
        "win_rate": summary["win_rate"],
        "net_profit": summary["net_profit"],
        "total_return_net": summary["net_profit"],
        "roi_pct": summary["roi_pct"],
        "sharpe_ratio": summary["sharpe_ratio"],
        "max_drawdown_pct": summary["max_drawdown_pct"],
        "edge_rate": summary["avg_net_edge"],
        "avg_edge": summary["avg_edge"],
        "avg_net_edge": summary["avg_net_edge"],
        "coverage_pct": summary["coverage_pct"],
        "coverage_reasons": summary["coverage_reasons"],
        "is_profitable": summary["net_profit"] > 0,
        "slices": summary["slices"],
        "regimes": regime_payload,
        "pnl_curve": summary["equity_curve"],
        "summary": {
            "markets": summary["n_markets"],
            "quotes": summary["n_quotes"],
            "trades": summary["n_trades"],
            "coverage_pct": summary["coverage_pct"],
            "coverage_reasons": summary["coverage_reasons"],
            "strategy_branch": branch,
        },
    }

def _build_lgbm_experiment_report(group: dict[str, Any], strategy_branch: str) -> dict[str, Any]:
    """Build a comparable, advisory report without activating any model."""
    branch = strategy_branch.strip().upper()
    candidates = list(group.get("variants") or [])
    # D is the control for the D/E/F experiment. Keep A as a
    # compatibility fallback for older A/B/C groups.
    control = next(
        (row for row in candidates if row.get("feature_set") == "D"),
        next((row for row in candidates if row.get("feature_set") == "A"), None),
    )

    def metric(row: dict[str, Any] | None, name: str) -> float | None:
        if not row or row.get(name) is None:
            return None
        try:
            value = float(row[name])
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    def oot_metric(backtest: dict[str, Any], name: str, fallback: str) -> float | None:
        windows = [item for item in (backtest.get("oot_windows") or []) if isinstance(item, dict)]
        values = [metric(item, name) for item in windows]
        values = [value for value in values if value is not None]
        return float(np.median(values)) if values else metric(backtest, fallback)

    def branch_result(row: dict[str, Any]) -> dict[str, Any]:
        variants = row.get("backtest_variants") or {}
        result = variants.get(branch) or {}
        return result if isinstance(result, dict) else {}

    report_rows: list[dict[str, Any]] = []
    for row in candidates:
        backtest = branch_result(row)
        trades_value = oot_metric(backtest, "n_trades", "n_trades")
        trades = int(round(trades_value or 0.0))
        pnl = oot_metric(backtest, "net_profit", "net_profit")
        drawdown = oot_metric(backtest, "max_drawdown_pct", "max_drawdown_pct")
        total_trades = int(round(metric(backtest, "n_trades") or 0.0))
        windows = [
            item for item in (backtest.get("oot_windows") or [])
            if isinstance(item, dict)
        ]
        window_pnls = [
            float(item.get("net_profit") or 0.0)
            for item in windows
            if item.get("net_profit") is not None
        ]
        stable_oot = len(windows) >= 3 and len(window_pnls) >= 3
        control_pnl = oot_metric(branch_result(control), "net_profit", "net_profit") if control else None
        report_rows.append({
            "model_id": row.get("model_id"),
            "feature_set": row.get("feature_set"),
            "version": row.get("version"),
            "auc": metric(row, "auc"),
            "ece": metric(row, "ece"),
            "brier": metric(row, "brier"),
            "pnl": pnl,
            "trades": trades,
            "median_oot_pnl": pnl,
            "median_oot_trades": trades,
            "median_oot_drawdown_pct": drawdown,
            "total_oot_trades": total_trades,
            "oot_window_count": len(windows),
            "oot_window_pnls": window_pnls,
            "stable_oot": stable_oot,
            "roi_pct": metric(backtest, "roi_pct"),
            "coverage_pct": metric(backtest, "coverage_pct"),
            "delta_vs_control": {
                "auc": (metric(row, "auc") - metric(control, "auc")) if metric(row, "auc") is not None and metric(control, "auc") is not None else None,
                "ece": (metric(row, "ece") - metric(control, "ece")) if metric(row, "ece") is not None and metric(control, "ece") is not None else None,
                "brier": (metric(row, "brier") - metric(control, "brier")) if metric(row, "brier") is not None and metric(control, "brier") is not None else None,
                "pnl": (pnl - control_pnl) if pnl is not None and control_pnl is not None else None,
                "trades": trades - int(round(oot_metric(branch_result(control), "n_trades", "n_trades") or 0.0)) if control else None,
            },
            "feature_audit_summary": row.get("feature_audit_summary") or {},
        })

    # Activation recommendations require real coverage: at least 50 total
    # OOT trades and three chronological windows. AUC/ECE fallback is only
    # diagnostic and is never presented as an activation decision.
    pnl_candidates = [
        row for row in report_rows
        if row["median_oot_pnl"] is not None
        and row["total_oot_trades"] >= 50
        and row["stable_oot"]
    ]
    if pnl_candidates:
        winner = max(
            pnl_candidates,
            key=lambda row: (
                row["median_oot_pnl"],
                -(row["median_oot_drawdown_pct"] or 0.0),
                row["stable_oot"],
                row["total_oot_trades"],
                row["auc"] or float("-inf"),
            ),
        )
        recommendation_status = "READY_FOR_SHADOW"
        reason = (
            f"Highest median {branch} OOT net PnL among candidates with "
            ">=50 trades and three OOT windows; validate in SHADOW before activation."
        )
    elif report_rows:
        winner = max(
            report_rows,
            key=lambda row: (
                row["median_oot_pnl"] if row["median_oot_pnl"] is not None else float("-inf"),
                -(row["median_oot_drawdown_pct"] or 0.0),
                row["auc"] if row["auc"] is not None else float("-inf"),
            ),
        )
        recommendation_status = "NO_PNL_SAMPLE"
        reason = (
            "No candidate satisfies >=50 OOT trades and three stable windows; "
            "ranking is diagnostic only and falls back to AUC/ECE."
        )
    else:
        winner = None
        recommendation_status = "NO_CANDIDATES"
        reason = "No comparable candidates were saved for this group."

    stable_by_variant = {
        row["feature_set"]: (row.get("feature_audit_summary") or {}).get("stable_features", [])
        for row in report_rows
    }
    stable_sets = [set(values) for values in stable_by_variant.values() if values]
    common_stable = sorted(set.intersection(*stable_sets)) if stable_sets else []
    return {
        "comparison_key": group.get("comparison_key"),
        "asset": group.get("asset"),
        "regime": group.get("regime"),
        "strategy_branch": branch,
        "rows": report_rows,
        "recommended_model_id": winner["model_id"] if winner else None,
        "recommended_variant": winner["feature_set"] if winner else None,
        "recommendation_status": recommendation_status,
        "recommendation_reason": reason,
        "stable_features_by_variant": stable_by_variant,
        "common_stable_features": common_stable,
        "activation_policy": "MANUAL_SHADOW_REQUIRED",
    }


async def _collect_lgbm_experiment_groups(
    db: AsyncSession,
    comparison_key_filter: str | None = None,
) -> dict:
    """Return comparable LightGBM A/B/C candidates and saved OOT summaries."""
    stmt = (
        select(
            ModelRegistry.id, ModelRegistry.asset, ModelRegistry.version,
            ModelRegistry.accuracy, ModelRegistry.ece, ModelRegistry.brier_score,
            ModelRegistry.is_active, ModelRegistry.trained_at,
            ModelRegistry.training_params, ModelRegistry.backtest_pnl,
            ModelRegistry.backtest_trades,
        )
        .where(ModelRegistry.model_type == "lgbm")
        .order_by(ModelRegistry.asset, ModelRegistry.trained_at.desc(), ModelRegistry.version.desc())
        .limit(500)
    )
    if comparison_key_filter:
        # JSON filtering keeps report requests bounded to the selected experiment.
        stmt = stmt.where(ModelRegistry.training_params["comparison_key"].as_string() == comparison_key_filter)
    rows = (await db.execute(stmt)).all()
    model_ids = [row.id for row in rows]
    artifact_ids = set((await db.execute(
        select(ModelRegistryOOFArtifact.model_registry_id).where(
            ModelRegistryOOFArtifact.model_registry_id.in_(model_ids),
            ModelRegistryOOFArtifact.schema_version == OOF_ARTIFACT_SCHEMA_VERSION,
        ) if model_ids else select(ModelRegistryOOFArtifact.model_registry_id).where(False)
    )).scalars().all())
    groups: dict[str, list[dict]] = {}
    for model in rows:
        params = model.training_params or {}
        key = params.get("comparison_key")
        if not key or params.get("target_source") != "POLYMARKET_FINAL_OUTCOME":
            continue
        variants = params.get("backtest_variants") or {}
        best_branch = None
        best_pnl = None
        for branch, variant in variants.items():
            if not isinstance(variant, dict) or variant.get("net_profit") is None:
                continue
            pnl = float(variant["net_profit"])
            if best_pnl is None or pnl > best_pnl:
                best_branch, best_pnl = branch, pnl
        candidate = {
            "model_id": model.id,
            "asset": model.asset,
            "regime": model.asset.rsplit("_", 1)[-1],
            "version": model.version,
            "feature_set": params.get("feature_set", "A"),
            "feature_set_version": params.get("feature_set_version", "legacy"),
            "auc": model.accuracy,
            "ece": model.ece,
            "brier": model.brier_score if model.brier_score is not None else params.get("brier_score"),
            "log_loss": params.get("log_loss"),
            "oot_markets": params.get("oot_markets"),
            "backtest_variants": variants,
            "backtest_pnl": model.backtest_pnl,
            "backtest_trades": model.backtest_trades,
            "is_active": bool(model.is_active),
            "trained_at": model.trained_at.isoformat() if model.trained_at else None,
            "artifact_available": model.id in artifact_ids,
            "feature_audit_summary": params.get("feature_audit_summary", {}),
            "model_config": params.get("model_config", {}),
            "best_branch": best_branch,
            "best_branch_pnl": best_pnl,
            "best_branch_source": "training_summary",
        }
        groups.setdefault(key, []).append(candidate)
    payload = []
    for key, candidates in groups.items():
        candidates.sort(key=lambda item: (item["feature_set"], item["regime"], item["version"]))
        sample = candidates[0]
        payload.append({
            "comparison_key": key,
            "asset": sample["asset"].rsplit("_", 1)[0],
            "regime": sample["regime"],
            "variants": candidates,
            "variant_count": len(candidates),
            "comparable": len({item["feature_set"] for item in candidates}) >= 2,
        })
    payload.sort(key=lambda item: item["comparison_key"])
    return {"groups": payload, "count": len(payload)}


@router.get("/api/experiments", dependencies=[Depends(verify_api_key)])
async def lgbm_experiment_groups(db: AsyncSession = Depends(get_db_session)) -> dict:
    """Return comparable LightGBM A/B/C candidates and saved OOT summaries."""
    return await _collect_lgbm_experiment_groups(db)


@router.get("/api/experiments/report", dependencies=[Depends(verify_api_key)])
async def lgbm_experiment_report(
    comparison_key: str,
    strategy_branch: Literal["OUTSIDER_ONLY", "FAVORITE_ONLY", "COMBINED"] = "COMBINED",
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """Return an advisory report for one comparable A/B/C experiment group."""
    payload = await _collect_lgbm_experiment_groups(db, comparison_key_filter=comparison_key)
    group = next((item for item in payload["groups"] if item["comparison_key"] == comparison_key), None)
    if group is None:
        raise HTTPException(status_code=404, detail="Comparable experiment group not found")
    return _build_lgbm_experiment_report(group, strategy_branch)

async def _saved_lgbm_model_polymarket_backtest(
    db: AsyncSession,
    *,
    model_id: int,
    strategy_branch: str,
) -> dict:
    """Re-run Polymarket OOT PnL for one persisted ModelRegistry candidate."""
    branch = strategy_branch.strip().upper()
    if branch not in {"OUTSIDER_ONLY", "FAVORITE_ONLY", "COMBINED"}:
        raise HTTPException(status_code=422, detail="strategy_branch must be OUTSIDER_ONLY, FAVORITE_ONLY or COMBINED")
    model = (await db.execute(
        select(ModelRegistry).where(ModelRegistry.id == model_id, ModelRegistry.model_type == "lgbm")
    )).scalar_one_or_none()
    if model is None:
        raise HTTPException(status_code=404, detail=f"LightGBM model {model_id} not found")
    artifact = (await db.execute(
        select(ModelRegistryOOFArtifact).where(
            ModelRegistryOOFArtifact.model_registry_id == model.id,
            ModelRegistryOOFArtifact.schema_version == OOF_ARTIFACT_SCHEMA_VERSION,
        )
    )).scalar_one_or_none()
    if artifact is None:
        raise HTTPException(status_code=409, detail={
            "error": "OOF_ARTIFACT_MISSING",
            "model_id": model_id,
            "message": "Retrain this candidate to persist reproducible OOF rows and quotes.",
        })
    try:
        payload = deserialize_oof_artifact(artifact.artifact_blob)
        backtest_options = _backtest_options_for_model(model.training_params or {})
        result = compute_oof_polymarket_backtest(
            payload["frame"], payload["oof_scores"], payload["quotes"],
            strategy_branch=branch,
            **backtest_options,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"error": "OOF_ARTIFACT_INVALID", "message": str(exc)}) from exc
    return {
        **{key: value for key, value in result.items() if key != "trades"},
        "model_id": model.id,
        "model_asset": model.asset,
        "model_version": model.version,
        "feature_set": (model.training_params or {}).get("feature_set", "A"),
        "pnl_mode": "POLYMARKET_OOF_SAVED_CANDIDATE",
        "artifact_rows": artifact.row_count,
        "artifact_schema_version": artifact.schema_version,
    }


@router.get("/api/backtest", dependencies=[Depends(verify_api_key)])
async def crypto_backtest(
    symbol: str = "BTCUSDT",
    interval: str = "15m",
    min_edge: float | None = Query(None),
    commission: float | None = Query(None),
    feature_set: str = "A",
    pnl_mode: Literal["BINANCE", "POLYMARKET"] = "BINANCE",
    strategy_branch: str = "OUTSIDER_ONLY",
    model_id: int | None = Query(None),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Запускает walk-forward backtest и возвращает детальные метрики и PnL-кривую.
    Результат кэшируется на 5 минут для предотвращения перегрузки CPU.
    """
    # Direct unit-test calls do not pass through FastAPI's dependency parser;
    # normalize Query(None) to the actual default before checking model_id.
    if not isinstance(model_id, int):
        model_id = getattr(model_id, "default", None)
    try:
        normalized_feature_set = normalize_feature_set(feature_set)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if model_id is not None:
        if pnl_mode != "POLYMARKET":
            raise HTTPException(status_code=422, detail="model_id requires pnl_mode=POLYMARKET")
        return await _saved_lgbm_model_polymarket_backtest(
            db, model_id=model_id, strategy_branch=strategy_branch
        )
    if pnl_mode == "POLYMARKET":
        return await _stored_lgbm_polymarket_backtest(
            db,
            symbol=symbol,
            feature_set=normalized_feature_set,
            strategy_branch=strategy_branch,
        )

    cache_key = f"backtest_{pnl_mode}_{symbol}_{interval}_{min_edge}_{commission}_{normalized_feature_set}"
    now = time.time()
    if cache_key in _cache and now - _cache[cache_key]["ts"] < 300:
        return _cache[cache_key]["data"]

    from polyflip.services.settings_service import get_float, get_int

    async with async_session() as session:
        if min_edge is None:
            min_edge = await get_float(session, "BACKTEST_MIN_EDGE")
        epsilon_quantile = await get_float(session, "LGBM_EPSILON_QUANTILE")

        lgbm_params = {
            "learning_rate": await get_float(
                session, "CRYPTO_LGBM_LEARNING_RATE"
            ),
            "subsample": await get_float(session, "CRYPTO_LGBM_SUBSAMPLE"),
            "colsample_bytree": await get_float(
                session, "CRYPTO_LGBM_COLSAMPLE_BYTREE"
            ),
            "num_leaves": await get_int(session, "CRYPTO_LGBM_NUM_LEAVES"),
            "max_depth": await get_int(session, "CRYPTO_LGBM_MAX_DEPTH"),
            "min_child_samples": await get_int(
                session, "CRYPTO_LGBM_MIN_CHILD_SAMPLES"
            ),
            "n_estimators": await get_int(session, "CRYPTO_LGBM_N_ESTIMATORS"),
            "reg_alpha": await get_float(session, "CRYPTO_LGBM_REG_ALPHA"),
            "reg_lambda": await get_float(session, "CRYPTO_LGBM_REG_LAMBDA"),
        }

        candles = await get_recent_candles(session, symbol, interval, limit=10_000)

    if len(candles) < 600:
        return {
            "error": f"Недостаточно свечей: {len(candles)} < 600. Пожалуйста, сделайте backfill.",
            "symbol": symbol,
        }

    df = build_features(candles)

    # Запускаем backtest в пуле потоков (CPU-bound)
    result = await asyncio.to_thread(
        run_backtest,
        df,
        symbol,
        min_edge,
        commission,
        lgbm_params=lgbm_params,
        epsilon_quantile=epsilon_quantile,
        feature_set=normalized_feature_set,
        closed_candles=candles,
    )

    data = {
        "symbol": result.symbol,
        "feature_set": normalized_feature_set,
        "feature_set_version": get_feature_set(normalized_feature_set).version,
        "n_candles_total": result.n_candles_total,
        "n_candles_test": result.n_candles_test,
        "n_trades": result.n_trades,
        "win_rate": round(result.win_rate, 4),
        "total_return_net": round(result.total_return_net, 5),
        "sharpe_ratio": round(result.sharpe_ratio, 3),
        "max_drawdown": round(result.max_drawdown, 5),
        "edge_rate": round(result.edge_rate, 4),
        "epsilon": round(result.epsilon, 6),
        "train_auc": round(result.train_auc, 4),
        "is_profitable": result.is_profitable(),
        "summary": result.summary(),
        "pnl_curve": result.pnl_curve,
    }

    _cache[cache_key] = {"ts": now, "data": data}
    return data



@router.post("/api/train", dependencies=[Depends(verify_api_key)])
async def crypto_train(
    background_tasks: Any = None,
    symbol: str = "BTCUSDT",
    interval: str = "15m",
    feature_set: str = "A",
    activate_after_train: bool = False,
    experiment_config_id: int | None = None,
    db: AsyncSession = Depends(get_db_session),
):
    """Queue training in the durable database-backed worker."""
    if not isinstance(db, AsyncSession):
        db = None

    symbol = str(symbol).upper().strip()
    interval = str(interval).strip() or "15m"
    if symbol not in CRYPTO_SYMBOLS and db is not None:
        raise HTTPException(status_code=422, detail=f"unsupported symbol: {symbol}")
    try:
        normalized_feature_set = normalize_feature_set(feature_set)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if _active_trainings.get(symbol, {}).get("status") == "training":
        return {
            "status": "already_running",
            "symbol": symbol,
            "message": f"Training for {symbol} is already running.",
        }

    if db is None:
        _active_trainings[symbol] = {
            "status": "training",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "feature_set": normalized_feature_set,
            "activate_after_train": bool(activate_after_train),
            "experiment_config_id": experiment_config_id,
        }
        _cache.pop("status", None)
        return {
            "status": "started",
            "symbol": symbol,
            "experiment_config_id": experiment_config_id,
            "message": f"Training for {symbol} was queued.",
        }

    if experiment_config_id is not None:
        config_row = await db.get(LGBMExperimentConfig, experiment_config_id)
        if config_row is None or config_row.is_archived:
            raise HTTPException(status_code=404, detail="experiment config not found or archived")
        normalized_feature_set = normalize_feature_set(config_row.feature_set)

    running = (
        await db.execute(
            select(LGBMTrainingJob)
            .where(
                LGBMTrainingJob.symbol == symbol,
                LGBMTrainingJob.status.in_({"QUEUED", "RUNNING"}),
            )
            .order_by(LGBMTrainingJob.created_at.desc(), LGBMTrainingJob.id.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if running is not None:
        return {
            "status": "already_running",
            "symbol": symbol,
            "job_id": running.id,
            "queue_status": running.status,
            "message": f"Training for {symbol} is already queued.",
        }

    job = LGBMTrainingJob(
        symbol=symbol,
        interval=interval,
        feature_set=normalized_feature_set,
        activate_after_train=bool(activate_after_train),
        experiment_config_id=experiment_config_id,
        status="QUEUED",
        created_at=datetime.now(timezone.utc),
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    # The durable DB row is the source of truth. Do not keep a second
    # in-process lock: the worker runs in another container and cannot clear it.
    _cache.pop("status", None)
    return {
        "status": "started",
        "symbol": symbol,
        "job_id": job.id,
        "queue_status": job.status,
        "experiment_config_id": experiment_config_id,
        "message": f"Training for {symbol} was queued.",
    }


@router.get("/api/train-jobs/{job_id}", dependencies=[Depends(verify_api_key)])
async def crypto_train_job(job_id: int, db: AsyncSession = Depends(get_db_session)):
    job = await db.get(LGBMTrainingJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="training job not found")
    return {
        "job_id": job.id,
        "symbol": job.symbol,
        "interval": job.interval,
        "feature_set": job.feature_set,
        "status": "training" if job.status in {"QUEUED", "RUNNING"} else job.status.lower(),
        "queue_status": job.status,
        "activate_after_train": bool(job.activate_after_train),
        "experiment_config_id": job.experiment_config_id,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "result": job.result,
        "error": job.error,
        "error_traceback": getattr(job, "error_traceback", None),
    }


@router.get("/api/models/coverage", dependencies=[Depends(verify_api_key)])
async def crypto_models_coverage(db: AsyncSession = Depends(get_db_session)):
    """
    P1: Таблица покрытия LightGBM-моделей по режимам волатильности.
    Показывает для каждого (актив × режим): активная версия или None.
    Используется для диагностики REGIME_UNAVAILABLE / MODEL_NOT_LOADED.
    """
    cache_key = "crypto_models_coverage"
    now = time.time()
    if cache_key in _cache and now - _cache[cache_key]["ts"] < 60:
        return _cache[cache_key]["data"]

    regimes = ["low_vol", "mid_vol", "high_vol"]
    result = {}

    allowed_assets = [f"{sym.upper()}_{regime}" for sym in CRYPTO_SYMBOLS for regime in regimes]
    stmt = (
        select(
            ModelRegistry.asset,
            ModelRegistry.version,
            ModelRegistry.is_active,
            ModelRegistry.trained_at,
            ModelRegistry.quality_gate_passed,
            ModelRegistry.activation_source,
            ModelRegistry.training_params,
            ModelRegistry.features,
        )
        .where(ModelRegistry.asset.in_(allowed_assets))
        .order_by(ModelRegistry.asset, ModelRegistry.version.desc())
    )
    all_rows = (await db.execute(stmt)).all()

    asset_groups = {}
    for r in all_rows:
        asset_groups.setdefault(r.asset, []).append(r)

    for sym in CRYPTO_SYMBOLS:
        sym_upper = sym.upper()
        result[sym_upper] = {}
        for regime in regimes:
            asset_key = f"{sym_upper}_{regime}"
            rows = asset_groups.get(asset_key, [])[:3]
            
            active_row = next((r for r in rows if r.is_active), None)
            all_versions = [r.version for r in rows]

            result[sym_upper][regime] = {
                "active_version": active_row.version if active_row else None,
                "active_trained_at": (
                    active_row.trained_at.isoformat() if active_row and active_row.trained_at else None
                ),
                "quality_gate_passed": active_row.quality_gate_passed if active_row else None,
                "activation_source": active_row.activation_source if active_row else None,
                "target_source": (active_row.training_params or {}).get("target_source") if active_row else None,
                "feature_set": (active_row.training_params or {}).get("feature_set", "A") if active_row else None,
                "feature_set_version": (active_row.training_params or {}).get("feature_set_version", "legacy") if active_row else None,
                "is_loadable": bool(active_row and (active_row.training_params or {}).get("target_source") == "POLYMARKET_FINAL_OUTCOME"),
                "recent_versions": all_versions,
                "status": (
                    ("ACTIVE" if (active_row and (active_row.training_params or {}).get("target_source") == "POLYMARKET_FINAL_OUTCOME") else
                     "ACTIVE_UNLOADABLE" if active_row else
                     ("HAS_INACTIVE" if all_versions else "MISSING"))
                ),
            }

    data = {"coverage": result, "regimes": regimes, "symbols": list(CRYPTO_SYMBOLS)}
    _cache[cache_key] = {"data": data, "ts": now}
    return data


@router.get("/api/models/analytics", dependencies=[Depends(verify_api_key)])
async def crypto_models_analytics(
    requested_mode: str = "PAPER",
    date_from: str = None,
    date_to: str = None,
    db: AsyncSession = Depends(get_db_session)
):
    if requested_mode not in {"PAPER", "LIVE", "SHADOW"}:
        raise HTTPException(
            status_code=422,
            detail="requested_mode должен быть PAPER, SHADOW или LIVE",
        )

    # ── Валидация дат (баг #2) ──────────────────────────────────────────────
    from datetime import timedelta
    import re as _re

    def _parse_date(s: str, field: str) -> datetime:
        if not _re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            raise HTTPException(status_code=422, detail=f"{field}: ожидается формат YYYY-MM-DD")
        try:
            return datetime.strptime(s, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=422, detail=f"{field}: несуществующая дата '{s}'")

    dt_from = _parse_date(date_from, "date_from") if date_from else None
    dt_to   = _parse_date(date_to,   "date_to")   if date_to   else None
    if dt_from and dt_to and dt_from > dt_to:
        raise HTTPException(status_code=422, detail="date_from не может быть позже date_to")
    # ────────────────────────────────────────────────────────────────────────

    cache_key = f"crypto_model_analytics_{requested_mode}_{date_from}_{date_to}"
    now = time.time()
    if cache_key in _cache:
        c = _cache[cache_key]
        if now - c["ts"] < 30:
            return c["data"]

    allowed_assets = []
    for s in CRYPTO_SYMBOLS:
        allowed_assets.extend([f"{s}_low_vol", f"{s}_mid_vol", f"{s}_high_vol", s])

    # 1. Запрашиваем модели
    stmt = (
        select(
            ModelRegistry.asset,
            ModelRegistry.version,
            ModelRegistry.accuracy,
            ModelRegistry.precision_at_threshold,
            ModelRegistry.recall_at_threshold,
            ModelRegistry.f1_at_threshold,
            ModelRegistry.brier_score,
            ModelRegistry.activation_source,
            ModelRegistry.training_params,
            ModelRegistry.features,
            ModelRegistry.quality_gate_passed,
            ModelRegistry.quality_override,
            ModelRegistry.backtest_pnl,
            ModelRegistry.backtest_trades,
            ModelRegistry.backtest_wr,
        )
        .where(ModelRegistry.asset.in_(allowed_assets))
        .order_by(ModelRegistry.asset, ModelRegistry.version.desc())
    )
    models = (await db.execute(stmt)).all()

    # Фильтры дат
    params = {"mode": requested_mode}
    date_filter = ""
    veto_date_filter = ""

    if dt_from:
        params["date_from"] = dt_from
        date_filter     += " AND created_at >= :date_from"
        veto_date_filter += " AND created_at >= :date_from"

    if dt_to:
        dt_to_exclusive = dt_to + timedelta(days=1)
        params["date_to_exclusive"] = dt_to_exclusive
        date_filter     += " AND created_at < :date_to_exclusive"
        veto_date_filter += " AND created_at < :date_to_exclusive"

    # 2. PRIMARY CTE
    primary_sql = text(f"""
        WITH trades AS (
            SELECT 
                model_key,
                model_version,
                COALESCE(realized_pnl_usdc, pnl) as pnl,
                created_at,
                id
            FROM trade_history
            WHERE mode = :mode
              AND position_status = 'CLOSED'
              AND model_key IS NOT NULL
              AND model_attribution_source IN ('EXACT', 'RECONSTRUCTED')
              AND COALESCE(realized_pnl_usdc, pnl) IS NOT NULL
              {date_filter}
        ),
        zero_trade AS (
            SELECT DISTINCT 
                model_key, 
                model_version, 
                CAST(0 AS NUMERIC) as pnl, 
                CAST('1970-01-01 00:00:00+00' AS TIMESTAMP) as created_at, 
                CAST(0 AS INTEGER) as id 
            FROM trades
        ),
        all_trades AS (
            SELECT model_key, model_version, pnl, created_at, id FROM trades
            UNION ALL
            SELECT model_key, model_version, pnl, created_at, id FROM zero_trade
        ),
        cumulatives AS (
            SELECT 
                model_key, model_version, pnl, created_at, id,
                SUM(pnl) OVER (PARTITION BY model_key, model_version ORDER BY created_at ASC, id ASC) as equity
            FROM all_trades
        ),
        peaks AS (
            SELECT 
                model_key, model_version, pnl, equity,
                MAX(equity) OVER (PARTITION BY model_key, model_version ORDER BY created_at ASC, id ASC) as running_peak
            FROM cumulatives
        ),
        drawdowns AS (
            SELECT 
                model_key, model_version, pnl, equity, running_peak,
                (equity - running_peak) as drawdown
            FROM peaks
        ),
        agg_trades AS (
            SELECT 
                model_key, model_version,
                COUNT(*) as total_trades,
                SUM(pnl) as total_pnl,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as win_count,
                SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) as gross_profit,
                ABS(SUM(CASE WHEN pnl < 0 THEN pnl ELSE 0 END)) as gross_loss
            FROM trades
            GROUP BY model_key, model_version
        ),
        agg_drawdowns AS (
            SELECT 
                model_key, model_version,
                MIN(drawdown) as max_drawdown
            FROM drawdowns
            GROUP BY model_key, model_version
        )
        SELECT 
            t.model_key, t.model_version,
            t.total_trades, t.total_pnl, t.win_count, t.gross_profit, t.gross_loss,
            d.max_drawdown
        FROM agg_trades t
        LEFT JOIN agg_drawdowns d ON t.model_key = d.model_key AND t.model_version = d.model_version
    """)
    
    primary_rows = (await db.execute(primary_sql, params)).fetchall()
    primary_stats = {(row.model_key, row.model_version): row for row in primary_rows}

    # 3. DIRECTION CTE
    direction_sql = text(f"""
        WITH trades AS (
            SELECT 
                COALESCE(direction_model_key, confirm_model_key) as model_key,
                COALESCE(direction_model_version, confirm_model_version) as model_version,
                COALESCE(realized_pnl_usdc, pnl) as pnl,
                created_at,
                id
            FROM trade_history
            WHERE mode = :mode
              AND position_status = 'CLOSED'
              AND COALESCE(direction_model_key, confirm_model_key) IS NOT NULL
              AND COALESCE(realized_pnl_usdc, pnl) IS NOT NULL
              {date_filter}
        ),
        unique_trades AS (
            SELECT DISTINCT id, model_key, model_version, pnl, created_at FROM trades
        ),
        zero_trade AS (
            SELECT DISTINCT 
                model_key, 
                model_version, 
                CAST(0 AS NUMERIC) as pnl, 
                CAST('1970-01-01 00:00:00+00' AS TIMESTAMP) as created_at, 
                CAST(0 AS INTEGER) as id 
            FROM trades
        ),
        all_trades AS (
            SELECT model_key, model_version, pnl, created_at, id FROM unique_trades
            UNION ALL
            SELECT model_key, model_version, pnl, created_at, id FROM zero_trade
        ),
        cumulatives AS (
            SELECT 
                model_key, model_version, pnl, created_at, id,
                SUM(pnl) OVER (PARTITION BY model_key, model_version ORDER BY created_at ASC, id ASC) as equity
            FROM all_trades
        ),
        peaks AS (
            SELECT 
                model_key, model_version, pnl, equity,
                MAX(equity) OVER (PARTITION BY model_key, model_version ORDER BY created_at ASC, id ASC) as running_peak
            FROM cumulatives
        ),
        drawdowns AS (
            SELECT 
                model_key, model_version, pnl, equity, running_peak,
                (equity - running_peak) as drawdown
            FROM peaks
        ),
        agg_trades AS (
            SELECT 
                model_key, model_version,
                COUNT(*) as total_trades,
                SUM(pnl) as total_pnl,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as win_count,
                SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) as gross_profit,
                ABS(SUM(CASE WHEN pnl < 0 THEN pnl ELSE 0 END)) as gross_loss
            FROM unique_trades
            GROUP BY model_key, model_version
        ),
        agg_drawdowns AS (
            SELECT 
                model_key, model_version,
                MIN(drawdown) as max_drawdown
            FROM drawdowns
            GROUP BY model_key, model_version
        )
        SELECT 
            t.model_key, t.model_version,
            t.total_trades, t.total_pnl, t.win_count, t.gross_profit, t.gross_loss,
            d.max_drawdown
        FROM agg_trades t
        LEFT JOIN agg_drawdowns d ON t.model_key = d.model_key AND t.model_version = d.model_version
    """)
    direction_rows = (await db.execute(direction_sql, params)).fetchall()
    direction_stats = {(row.model_key, row.model_version): row for row in direction_rows}

    # 3.5 VETO CTE
    # Используем COUNT(DISTINCT decision_run_id) для дедупликации исторических записей:
    # старые строки могли иметь дубли ML + COMBINED для одного цикла решения (баг #3)
    veto_params = dict(params)
    veto_params["decision_mode"] = "PAPER" if requested_mode in ("LIVE", "PAPER") else requested_mode

    veto_sql = text(f"""
        SELECT
            COALESCE(direction_model_key, confirm_model_key) as model_key,
            COALESCE(direction_model_version, confirm_model_version) as model_version,
            COUNT(DISTINCT CASE WHEN confirm_passed = true  THEN decision_run_id END) as confirm_passed_count,
            COUNT(DISTINCT CASE WHEN confirm_passed = false THEN decision_run_id END) as veto_count
        FROM decision_funnel_log
        WHERE COALESCE(direction_model_key, confirm_model_key) IS NOT NULL
          AND execution_mode = :decision_mode
          AND decision_run_id IS NOT NULL
          AND direction_status IN ('OK', 'READY')
          {veto_date_filter}
        GROUP BY
            COALESCE(direction_model_key, confirm_model_key),
            COALESCE(direction_model_version, confirm_model_version)
    """)
    veto_rows = (await db.execute(veto_sql, veto_params)).fetchall()
    veto_stats = {(row.model_key, row.model_version): row for row in veto_rows}

    # 4. Считаем метрики для каждой версии в реестре
    result = {}
    for m in models:
        key = f"{m.asset}_v{m.version}"
        p = primary_stats.get((m.asset, m.version))
        c = direction_stats.get((m.asset, m.version))
        v = veto_stats.get((m.asset, m.version))

        def calc_pf(profit, loss):
            if not profit and not loss: return 0.0
            if not loss: return None  # Infinity: прибыль есть, убытков нет
            return round(float(profit) / float(loss), 2)

        metrics = {
            "pnl": round(float(p.total_pnl), 4) if p else 0.0,
            "win_rate": round(float(p.win_count) / float(p.total_trades) * 100, 1) if p and p.total_trades > 0 else None,
            "total_trades": int(p.total_trades) if p else 0,
            "max_drawdown": round(float(p.max_drawdown), 4) if p and p.max_drawdown is not None else 0.0,
            "profit_factor": calc_pf(p.gross_profit, p.gross_loss) if p else 0.0,

            "direction_pnl": round(float(c.total_pnl), 4) if c else 0.0,
            "direction_win_rate": round(float(c.win_count) / float(c.total_trades) * 100, 1) if c and c.total_trades > 0 else None,
            "direction_trades": int(c.total_trades) if c else 0,
            "direction_max_drawdown": round(float(c.max_drawdown), 4) if c and c.max_drawdown is not None else 0.0,
            "direction_profit_factor": calc_pf(c.gross_profit, c.gross_loss) if c else 0.0,

            "confirm_passed_count": int(v.confirm_passed_count) if v else 0,
            "veto_count": int(v.veto_count) if v else 0,

            # Метрики качества из ModelRegistry
            "auc_lift": round(m.accuracy - 0.5, 4) if m.accuracy else None,
            "feature_set": (m.training_params or {}).get("feature_set", "A"),
            "feature_set_version": (m.training_params or {}).get("feature_set_version", "legacy"),
            "precision": round(m.precision_at_threshold, 4) if getattr(m, "precision_at_threshold", None) is not None else None,
            "recall": round(m.recall_at_threshold, 4) if getattr(m, "recall_at_threshold", None) is not None else None,
            "f1": round(m.f1_at_threshold, 4) if getattr(m, "f1_at_threshold", None) is not None else None,
            "brier_score": round(m.brier_score, 4) if getattr(m, "brier_score", None) is not None else None,
            "backtest_pnl": round(float(m.backtest_pnl), 6) if getattr(m, "backtest_pnl", None) is not None else None,
            "backtest_trades": int(m.backtest_trades) if getattr(m, "backtest_trades", None) is not None else 0,
            "backtest_wr": round(float(m.backtest_wr), 6) if getattr(m, "backtest_wr", None) is not None else None,
            "backtest_pnl_mode": (m.training_params or {}).get("backtest_pnl_mode"),

            # Аудит активации
            "activation_source": m.activation_source,
            "quality_gate_passed": m.quality_gate_passed,
            "quality_override": getattr(m, "quality_override", None),
        }

        result[key] = metrics

    _cache[cache_key] = {"ts": now, "data": result}
    return result


class ActivateModelRequest(BaseModel):
    force: bool = False
    reason: str | None = None


@router.post(
    "/api/models/{asset}/activate/{version}", dependencies=[Depends(verify_api_key)]
)
async def activate_crypto_model(
    asset: str,
    version: int,
    payload: ActivateModelRequest = Body(default_factory=ActivateModelRequest),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Активирует указанную версию крипто-модели, деактивируя остальные.

    Если модель не прошла Quality Gate и force=False → HTTP 409.
    Если force=True → активация помечается как DASHBOARD.
    """
    allowed_assets = []
    for s in CRYPTO_SYMBOLS:
        allowed_assets.extend([f"{s}_low_vol", f"{s}_mid_vol", f"{s}_high_vol", s])

    if asset not in allowed_assets:
        raise HTTPException(status_code=404, detail=f"Актив {asset} не найден")

    # 1. Загружаем целевую модель
    target_stmt = select(ModelRegistry).where(
        ModelRegistry.asset == asset,
        ModelRegistry.version == version,
    )
    model = (await db.execute(target_stmt)).scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail=f"Версия {version} не найдена")

    # 1.5 Technical artifact validation. Performance metrics are advisory and
    # must not block manual experiments.
    params = model.training_params or {}
    if params.get("target_source") != "POLYMARKET_FINAL_OUTCOME":
        raise HTTPException(
            status_code=422,
            detail={
                "code": "NON_CANONICAL_TARGET",
                "message": "Only POLYMARKET_FINAL_OUTCOME models can be activated",
                "target_source": params.get("target_source"),
            },
        )
    try:
        model_features = validate_feature_schema(
            parse_feature_names(model.features) or tuple(CONTROL_FEATURES)
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    from polyflip.crypto.trainer import _model_smoke_test
    smoke_error = _model_smoke_test(model.model_blob, model_features)
    if smoke_error:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "MODEL_ARTIFACT_INVALID",
                "message": "Model artifact is incompatible with its feature schema",
                "error": smoke_error,
            },
        )

    # 3. Запоминаем предыдущую активную версию
    prev_stmt = select(ModelRegistry).where(
        ModelRegistry.asset == asset,
        ModelRegistry.is_active == True,
    )
    prev_model = (await db.execute(prev_stmt)).scalar_one_or_none()
    previous_version = prev_model.version if prev_model else None

    # 4. Атомарная активация
    # Семантика (баг #4): activation_source показывает КТО активировал,
    # quality_override показывает, был ли обойдён Quality Gate
    now = datetime.now(timezone.utc)
    is_quality_override = False

    await db.execute(
        update(ModelRegistry)
        .where(ModelRegistry.asset == asset)
        .values(is_active=False)
    )
    await db.execute(
        update(ModelRegistry)
        .where(ModelRegistry.asset == asset, ModelRegistry.version == version)
        .values(
            is_active=True,
            activation_source="DASHBOARD",
            quality_override=is_quality_override,
            activated_at=now,
            activated_by="dashboard",
            activation_reason=payload.reason,
        )
    )

    await db.commit()

    # 5. Инвалидируем кэш предиктора (в рамках того же процесса API)
    symbol = asset.split("_")[0]
    try:
        CryptoPredictor.invalidate_all(symbol)
        logger.info("predictor_cache_invalidated_after_manual_activation", asset=asset, version=version)
    except Exception as exc:
        logger.warning("predictor_invalidate_failed", error=str(exc))

    _cache.clear()

    # 6. Формируем ответ
    response: dict = {
        "status": "success",
        "asset": asset,
        "active_version": version,
        "previous_version": previous_version,
        "activation_source": "DASHBOARD",
        "quality_gate_passed": model.quality_gate_passed,
        "quality_override": is_quality_override,
    }
    if is_quality_override:
        response["warning"] = (
            f"Модель {asset} v{version} активирована через DASHBOARD с обходом Quality Gate."
        )
    return response


@router.delete("/api/models/{asset}/{version}", dependencies=[Depends(verify_api_key)])
async def delete_crypto_model(
    asset: str, version: int, db: AsyncSession = Depends(get_db_session)
):
    """Удаляет указанную версию крипто-модели из БД."""
    allowed_assets = []
    for s in CRYPTO_SYMBOLS:
        allowed_assets.extend([f"{s}_low_vol", f"{s}_mid_vol", f"{s}_high_vol", s])

    if asset not in allowed_assets:
        raise HTTPException(status_code=404, detail=f"Актив {asset} не найден")

    # Check if active
    stmt = select(ModelRegistry).where(
        ModelRegistry.asset == asset, ModelRegistry.version == version
    )
    model = (await db.execute(stmt)).scalar_one_or_none()

    if not model:
        raise HTTPException(
            status_code=404, detail=f"Модель {asset} v{version} не найдена"
        )

    if model.is_active:
        raise HTTPException(
            status_code=400,
            detail="Нельзя удалить активную модель. Сначала активируйте другую.",
        )

    # Delete
    del_stmt = delete(ModelRegistry).where(
        ModelRegistry.asset == asset, ModelRegistry.version == version
    )
    await db.execute(del_stmt)
    await db.commit()

    _cache.clear()
    return {"status": "success", "detail": f"Модель {asset} v{version} удалена"}
