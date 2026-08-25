"""Dynamic OpenCode model catalog with a DB-backed cache.

The dashboard and the independent research agent must be able to select any
model the configured OpenCode endpoint actually exposes.  Discovery results
are normalized, cached in ``ai_llm_model_catalog`` and served from cache with
``stale=true`` when the provider endpoint is unreachable.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.ai_lab.llm import get_llm_model_catalog
from polyflip.config import settings as default_settings
from polyflip.db.models import AILLMModelCatalog

logger = structlog.get_logger("polyflip.ai_lab.llm_catalog")

SUPPORTED_PROTOCOLS = {"responses", "chat_completions"}
DEFAULT_TTL_SECONDS = 3600


def normalize_models(payload: Any) -> list[dict[str, Any]]:
    """Normalize the supported discovery payload shapes into catalog rows.

    Supported inputs::

        {"data": [{"id": "model-a"}]}
        {"models": [{"id": "model-a", "name": "Model A"}]}

    Rows without an ``id``/``name`` are skipped; unknown protocols fall back
    to ``responses``.
    """
    if not isinstance(payload, dict):
        return []
    rows = payload.get("data") or payload.get("models") or []
    if not isinstance(rows, list):
        return []
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_id = str(row.get("id") or row.get("name") or "").strip()
        if not raw_id:
            continue
        protocol = str(row.get("protocol") or "responses").strip().lower()
        if protocol not in SUPPORTED_PROTOCOLS:
            protocol = "responses"
        result[raw_id] = {
            "model_id": raw_id,
            "display_name": str(
                row.get("display_name") or row.get("name") or raw_id
            ),
            "protocol": protocol,
            "supports_structured_output": bool(
                row.get("supports_structured_output", True)
            ),
        }
    return [result[key] for key in sorted(result)]


async def fetch_opencode_models(
    endpoint_url: str,
    api_key: str,
    *,
    timeout_seconds: float = 10.0,
) -> Any:
    """Fetch the raw discovery payload from the provider endpoint.

    Normalization happens in :func:`normalize_models` so callers can inspect
    or mock the transport independently of the payload shape.
    """
    if not endpoint_url.strip():
        raise RuntimeError("AI_LAB_OPENCODE_MODELS_ENDPOINT is not configured")
    headers = {"Content-Type": "application/json"}
    if api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.get(endpoint_url.strip(), headers=headers)
        response.raise_for_status()
        return response.json()


def _as_utc(value: datetime | None) -> datetime | None:
    """Normalize DB datetimes to aware UTC (SQLite drops tzinfo)."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _row_to_item(row: AILLMModelCatalog) -> dict[str, Any]:
    return {
        "id": row.model_id,
        "label": row.display_name or row.model_id,
        "protocol": row.protocol,
        "supports_structured_output": bool(row.supports_structured_output),
        "is_available": bool(row.is_available),
    }


def _defaults_for(models: list[dict[str, Any]]) -> dict[str, str]:
    available_ids = [
        item["id"] for item in models if item.get("is_available", True)
    ]
    if not available_ids:
        return {"research_model": "", "summary_model": ""}
    return {
        "research_model": available_ids[0],
        "summary_model": available_ids[-1],
    }


def _catalog_response(
    static_shape: dict[str, Any],
    models: list[dict[str, Any]],
    *,
    source: str,
    stale: bool,
    checked_at: datetime,
    error: str | None = None,
) -> dict[str, Any]:
    payload = dict(static_shape)
    if models:
        payload["models"] = models
        payload["defaults"] = _defaults_for(models)
    payload.update({
        "stale": stale,
        "source": source,
        "checked_at": checked_at.isoformat(),
    })
    if error:
        payload["error"] = error[:500]
    return payload


async def _upsert_live_rows(
    db: AsyncSession,
    provider: str,
    items: list[dict[str, Any]],
    *,
    now: datetime,
    ttl_seconds: int,
) -> list[dict[str, Any]]:
    existing = (
        await db.execute(
            select(AILLMModelCatalog).where(AILLMModelCatalog.provider == provider)
        )
    ).scalars().all()
    by_id = {row.model_id: row for row in existing}
    fresh_ids: set[str] = set()
    expires_at = now + timedelta(seconds=max(ttl_seconds, 1))
    for item in items:
        model_id = item["model_id"]
        fresh_ids.add(model_id)
        row = by_id.get(model_id)
        if row is None:
            row = AILLMModelCatalog(
                provider=provider,
                model_id=model_id,
                discovered_at=now,
            )
            db.add(row)
        row.display_name = item["display_name"]
        row.protocol = item["protocol"]
        row.supports_structured_output = item["supports_structured_output"]
        row.is_available = True
        row.expires_at = expires_at
    for model_id, row in by_id.items():
        if model_id not in fresh_ids:
            row.is_available = False
    await db.flush()
    refreshed = (
        await db.execute(
            select(AILLMModelCatalog)
            .where(AILLMModelCatalog.provider == provider)
            .order_by(AILLMModelCatalog.model_id)
        )
    ).scalars().all()
    return [_row_to_item(row) for row in refreshed]


async def refresh_model_catalog(
    db: AsyncSession,
    *,
    provider: str | None = None,
    refresh: bool = False,
    settings_obj: Any | None = None,
) -> dict[str, Any]:
    """Return the merged catalog for one provider, refreshing when needed."""
    cfg = settings_obj or default_settings
    now = datetime.now(timezone.utc)
    provider_name = (provider or "").strip().lower() or "opencode"

    def static_shape() -> dict[str, Any]:
        return get_llm_model_catalog(provider_name)

    if provider_name != "opencode":
        # Only OpenCode has a dynamic discovery endpoint today; other
        # providers keep their static configuration-driven catalog.
        return _catalog_response(
            static_shape(), [], source="static", stale=False, checked_at=now
        )

    endpoint = str(getattr(cfg, "AI_LAB_OPENCODE_MODELS_ENDPOINT", "") or "")
    ttl_seconds = int(
        getattr(cfg, "AI_LAB_OPENCODE_CATALOG_TTL_SECONDS", DEFAULT_TTL_SECONDS)
        or DEFAULT_TTL_SECONDS
    )
    rows = (
        await db.execute(
            select(AILLMModelCatalog).where(
                AILLMModelCatalog.provider == provider_name
            )
        )
    ).scalars().all()
    fresh_rows = [
        row for row in rows
        if row.is_available
        and (_as_utc(row.expires_at) is None or _as_utc(row.expires_at) > now)
    ]

    models: list[dict[str, Any]] | None = None
    source = "cache"
    stale = False
    live_error: str | None = None
    live_failed = False

    should_refresh_live = bool(endpoint) and (refresh or not fresh_rows)
    if should_refresh_live:
        api_key = str(
            getattr(cfg, "AI_LAB_LLM_API_KEY", "")
            or getattr(cfg, "OPENAI_API_KEY", "")
            or ""
        )
        try:
            raw_payload = await fetch_opencode_models(endpoint, api_key)
            fetched = normalize_models(raw_payload)
            models = await _upsert_live_rows(
                db,
                provider_name,
                fetched,
                now=now,
                ttl_seconds=ttl_seconds,
            )
            source = "live"
        except Exception as exc:
            live_failed = True
            live_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "opencode_model_discovery_failed",
                endpoint=endpoint,
                error=live_error,
            )

    if models is None:
        if fresh_rows and not live_failed:
            # Healthy TTL cache served without hitting the endpoint.
            models = [
                _row_to_item(row)
                for row in sorted(fresh_rows, key=lambda r: r.model_id)
            ]
            source = "cache"
            stale = False
        elif rows:
            # Live fetch failed (or cache expired): serve the last known
            # catalog and flag it stale so the UI can warn operators.
            cached_available = [
                row for row in rows if row.is_available
            ]
            if cached_available:
                models = [
                    _row_to_item(row)
                    for row in sorted(cached_available, key=lambda r: r.model_id)
                ]
                source = "cache"
                stale = True
        if models is None:
            fallback_csv = str(
                getattr(cfg, "AI_LAB_OPENCODE_MODELS_FALLBACK", "") or ""
            )
            fallback_ids = [item.strip() for item in fallback_csv.split(",") if item.strip()]
            if fallback_ids:
                models = [
                    {
                        "id": model_id,
                        "label": model_id,
                        "protocol": "responses",
                        "supports_structured_output": True,
                        "is_available": True,
                    }
                    for model_id in fallback_ids
                ]
                source = "fallback"
                stale = True
            else:
                return _catalog_response(
                    static_shape(),
                    [],
                    source="static",
                    stale=False,
                    checked_at=now,
                    error=live_error,
                )
    return _catalog_response(
        static_shape(),
        models,
        source=source,
        stale=stale,
        checked_at=now,
        error=live_error,
    )


async def get_available_catalog_models(
    db: AsyncSession,
    provider: str | None = None,
    *,
    settings_obj: Any | None = None,
) -> dict[str, dict[str, Any]]:
    """Return available dynamic rows keyed by model id (no network calls)."""
    cfg = settings_obj or default_settings
    provider_name = (provider or "").strip().lower() or str(
        getattr(cfg, "AI_LAB_LLM_PROVIDER", "mock")
    ).lower()
    rows = (
        await db.execute(
            select(AILLMModelCatalog).where(
                AILLMModelCatalog.provider == provider_name,
                AILLMModelCatalog.is_available.is_(True),
            )
        )
    ).scalars().all()
    return {row.model_id: row for row in rows}
