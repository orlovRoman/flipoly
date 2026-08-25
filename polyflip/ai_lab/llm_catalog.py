"""Dynamic OpenCode model catalog with a DB-backed cache.

The dashboard and the independent research agent must be able to select any
model the configured OpenCode endpoint actually exposes.  Discovery results
are normalized, cached in ``ai_llm_model_catalog`` and served from cache with
``stale=true`` when the provider endpoint is unreachable.

It also owns the lightweight availability probe used before a model may be
attached to a research run (no trading data ever leaves the process).
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

import httpx
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.ai_lab.llm import (
    DEFAULT_OPENCODE_CHAT_ENDPOINT,
    DEFAULT_OPENCODE_CHAT_MODELS,
    DEFAULT_OPENCODE_ENDPOINT,
    get_llm_model_catalog,
)
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
    # Expose split discovery/probe fields for the dashboard.
    last = getattr(row, "last_checked_at", None)
    return {
        "id": row.model_id,
        "label": row.display_name or row.model_id,
        "protocol": row.protocol,
        "supports_structured_output": bool(row.supports_structured_output),
        "is_available": bool(row.is_available),
        "is_discovered": bool(getattr(row, "is_discovered", True)),
        "probe_status": str(getattr(row, "probe_status", "UNCHECKED")),
        "last_checked_at": _as_utc(last).isoformat() if last else None,
        "is_unchecked": str(getattr(row, "probe_status", "UNCHECKED")) == "UNCHECKED",
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
        is_new = row is None
        if is_new:
            row = AILLMModelCatalog(
                provider=provider,
                model_id=model_id,
                discovered_at=now,
            )
            db.add(row)
            # New discovery starts as UNCHECKED.
            row.is_discovered = True  # type: ignore[attr-defined]
            row.probe_status = "UNCHECKED"  # type: ignore[attr-defined]
        row.display_name = item["display_name"]
        row.protocol = item["protocol"]
        row.supports_structured_output = item["supports_structured_output"]
        row.is_available = True
        row.is_discovered = True  # type: ignore[attr-defined]
        if is_new:
            row.probe_status = "UNCHECKED"  # type: ignore[attr-defined]
        row.expires_at = expires_at
    for model_id, row in by_id.items():
        if model_id not in fresh_ids:
            row.is_available = False
            row.is_discovered = False  # type: ignore[attr-defined]
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
                AILLMModelCatalog.is_discovered.is_(True),
                AILLMModelCatalog.probe_status == "PASSED",
                AILLMModelCatalog.supports_structured_output.is_(True),
            )
        )
    ).scalars().all()
    # Apply probe TTL filtering.
    ttl = int(getattr(cfg, "AI_LAB_OPENCODE_PROBE_TTL_SECONDS", 86400) or 86400)
    now = datetime.now(timezone.utc)
    fresh: dict[str, AILLMModelCatalog] = {}
    for row in rows:
        last = _as_utc(getattr(row, "last_checked_at", None))
        if last is None:
            continue
        if (now - last).total_seconds() > ttl:
            continue
        fresh[row.model_id] = row
    return fresh


# ---------------------------------------------------------------------------
# Availability probe (T03)
# ---------------------------------------------------------------------------
PROBE_INSTRUCTIONS = "Return JSON only"
PROBE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}
ProbeSender = Callable[..., Awaitable[Any]]


def _probe_body(protocol: str, model_id: str) -> dict[str, Any]:
    if protocol == "chat_completions":
        return {
            "model": model_id,
            "messages": [
                {"role": "system", "content": PROBE_INSTRUCTIONS},
                {"role": "user", "content": json.dumps({
                    "task": PROBE_INSTRUCTIONS,
                    "schema": {"ok": "boolean"},
                })},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "model_probe",
                    "strict": True,
                    "schema": PROBE_SCHEMA,
                },
            },
        }
    return {
        "model": model_id,
        "input": [
            {"role": "system", "content": PROBE_INSTRUCTIONS},
            {"role": "user", "content": json.dumps({
                "task": PROBE_INSTRUCTIONS,
                "schema": {"ok": "boolean"},
            })},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "model_probe",
                "strict": True,
                "schema": PROBE_SCHEMA,
            }
        },
        "store": False,
    }


def _probe_candidates(provider_name: str, settings_obj: Any) -> list[dict[str, Any]]:
    """Ordered probe targets; a custom endpoint override always wins."""
    override = str(getattr(settings_obj, "AI_LAB_LLM_ENDPOINT", "") or "").strip()
    if override:
        protocol = (
            "chat_completions"
            if override.rstrip("/").endswith("/chat/completions")
            else "responses"
        )
        return [{"url": override.rstrip("/"), "protocol": protocol}]
    if provider_name == "openai":
        return [
            {
                "url": "https://api.openai.com/v1/responses",
                "protocol": "responses",
            }
        ]
    return [
        {"url": DEFAULT_OPENCODE_ENDPOINT, "protocol": "responses"},
        {"url": DEFAULT_OPENCODE_CHAT_ENDPOINT, "protocol": "chat_completions"},
    ]


async def _default_probe_sender(url: str, headers: dict[str, str], body: dict[str, Any]) -> Any:
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(url, headers=headers, json=body)
        response.raise_for_status()
        return response.json()


def _extract_output_text(data: Any, *, is_chat_completion: bool) -> str:
    from polyflip.ai_lab.llm import OpenAIResponsesProvider

    if not isinstance(data, dict):
        return ""
    return OpenAIResponsesProvider._response_text(
        data, is_chat_completion=is_chat_completion
    )


async def check_model_availability(
    provider_name: str,
    model_id: str,
    *,
    settings_obj: Any | None = None,
    sender: ProbeSender | None = None,
) -> dict[str, Any]:
    """Probe one model with a tiny structured request (no trading data).

    A model counts as available only when some supported transport returns
    valid JSON matching ``{"ok": bool}``.
    """
    cfg = settings_obj or default_settings
    provider = (provider_name or "").strip().lower()
    model = (model_id or "").strip()
    checked_at = datetime.now(timezone.utc)
    report: dict[str, Any] = {
        "provider": provider,
        "model_id": model,
        "available": False,
        "protocol": None,
        "latency_ms": None,
        "checked_at": checked_at.isoformat(),
        "error": None,
    }
    if provider == "mock":
        report.update({"available": True, "protocol": "mock", "latency_ms": 0})
        return report
    if provider not in {"openai", "opencode"}:
        raise ValueError(f"Unsupported AI Lab LLM provider: {provider}")
    api_key = str(
        getattr(cfg, "AI_LAB_LLM_API_KEY", "")
        or getattr(cfg, "OPENAI_API_KEY", "")
        or ""
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    send = sender or _default_probe_sender
    last_error = "no candidate endpoint responded"
    for candidate in _probe_candidates(provider, cfg):
        started = time.monotonic()
        try:
            data = await send(
                url=candidate["url"],
                headers=headers,
                body=_probe_body(candidate["protocol"], model),
            )
            raw_text = _extract_output_text(
                data, is_chat_completion=candidate["protocol"] == "chat_completions"
            ).strip()
            payload = json.loads(raw_text)
            if not (
                isinstance(payload, dict)
                and payload.get("ok") is True
                and len(payload) >= 1
            ):
                last_error = "structured output did not match {\"ok\": true}"
                continue
            latency_ms = int((time.monotonic() - started) * 1000)
            report.update({
                "available": True,
                "protocol": candidate["protocol"],
                "latency_ms": latency_ms,
                "error": None,
            })
            return report
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
    report["error"] = last_error[:500]
    return report


async def persist_model_check_result(
    db: AsyncSession,
    *,
    provider: str,
    model_id: str,
    report: dict[str, Any],
) -> AILLMModelCatalog:
    """Store the probe outcome on the catalog row (creating it if needed)."""
    now = datetime.now(timezone.utc)
    row = (
        await db.execute(
            select(AILLMModelCatalog).where(
                AILLMModelCatalog.provider == provider,
                AILLMModelCatalog.model_id == model_id,
            )
        )
    ).scalar_one_or_none()
    if row is None:
        row = AILLMModelCatalog(
            provider=provider,
            model_id=model_id,
            display_name=model_id,
            discovered_at=now,
        )
        db.add(row)
    is_passed = bool(report.get("available"))
    row.is_available = is_passed
    row.is_discovered = True  # type: ignore[attr-defined]
    row.probe_status = "PASSED" if is_passed else "FAILED"  # type: ignore[attr-defined]
    # last_checked_at is the authoritative probe timestamp
    try:
        checked_iso = report.get("checked_at")
        parsed = datetime.fromisoformat(str(checked_iso).replace("Z", "+00:00")) if checked_iso else now
    except Exception:
        parsed = now
    row.last_checked_at = _as_utc(parsed)  # type: ignore[attr-defined]
    if report.get("protocol") in SUPPORTED_PROTOCOLS | {"mock"}:
        row.protocol = str(report["protocol"])
    metadata = dict(row.raw_metadata or {})
    metadata["last_check"] = {
        "available": is_passed,
        "latency_ms": report.get("latency_ms"),
        "protocol": report.get("protocol"),
        "checked_at": report.get("checked_at"),
        "error": report.get("error"),
    }
    row.raw_metadata = metadata
    await db.flush()
    return row


# ---------------------------------------------------------------------------
# Immutable run-time LLM selection snapshot (T05)
# ---------------------------------------------------------------------------
async def resolve_llm_snapshot(
    db: AsyncSession,
    *,
    provider: str | None,
    research_model: str | None,
    summary_model: str | None,
    settings_obj: Any | None = None,
) -> dict[str, Any]:
    """Resolve and validate the immutable LLM selection for a new run.

    Raises ``ValueError`` when a dynamically known model is missing from the
    catalog or its latest availability check failed. When no dynamic catalog
    exists for the provider the legacy static configuration catalog is used.
    """
    cfg = settings_obj or default_settings
    provider_name = (provider or "").strip().lower() or str(
        getattr(cfg, "AI_LAB_LLM_PROVIDER", "mock")
    ).strip().lower()
    checked_at = datetime.now(timezone.utc)

    def snapshot(
        research: str,
        summary: str,
        *,
        protocol: str | None,
        status_value: str,
        catalog_time: datetime | None,
    ) -> dict[str, Any]:
        return {
            "provider": provider_name,
            "research_model": research,
            "summary_model": summary,
            "catalog_checked_at": (
                catalog_time.isoformat() if catalog_time else None
            ),
            "catalog_model_status": status_value,
            "protocol": protocol,
        }

    if provider_name == "mock":
        research = (research_model or "").strip() or "mock-gpt-5"
        summary = (summary_model or "").strip() or "mock-gpt-5"
        return snapshot(
            research, summary, protocol="mock", status_value="available",
            catalog_time=checked_at,
        )

    by_id: dict[str, AILLMModelCatalog] = {}
    if provider_name == "opencode":
        # Only OpenCode has a dynamic discovery cache today.
        rows = (
            await db.execute(
                select(AILLMModelCatalog).where(
                    AILLMModelCatalog.provider == provider_name
                )
            )
        ).scalars().all()
        by_id = {row.model_id: row for row in rows}

    if not by_id:
        # Legacy/static path keeps pre-catalog behavior working.
        resolved = get_llm_model_catalog(provider_name)
        allowed = {str(item["id"]) for item in resolved["models"]}
        defaults = resolved["defaults"]
        research = (research_model or "").strip() or str(defaults["research_model"])
        summary = (summary_model or "").strip() or str(defaults["summary_model"])
        if research not in allowed or summary not in allowed:
            raise ValueError(
                f"Unknown model for provider {provider_name}: "
                f"research={research!r}, summary={summary!r}"
            )
        protocol = (
            "chat_completions"
            if provider_name == "opencode"
            and research in set(DEFAULT_OPENCODE_CHAT_MODELS)
            else "responses"
        )
        return snapshot(
            research, summary, protocol=protocol,
            status_value="legacy_static", catalog_time=None,
        )

    PROBE_TTL_SECONDS = int(getattr(cfg, "AI_LAB_OPENCODE_PROBE_TTL_SECONDS", 86400) or 86400)

    def validated(model_id: str | None) -> AILLMModelCatalog:
        clean = (model_id or "").strip()
        row = by_id.get(clean)
        if row is None:
            raise ValueError(
                f"Model '{clean}' is not present in the {provider_name} catalog"
            )
        # New semantics: must be discovered, probe PASSED, and support structured output.
        is_disc = bool(getattr(row, "is_discovered", True))
        probe = str(getattr(row, "probe_status", "UNCHECKED"))
        supports = bool(row.supports_structured_output)
        # Backward compat: if probe is UNCHECKED but is_available True, treat as legacy discovered
        # without probe. For new code, probe must be PASSED.
        # Keep legacy rows that haven't been probed yet as UNCHECKED -> require probe.
        # However tests that seed only is_available should still pass if they also set probe_status.
        # We enforce strict check.
        if not is_disc:
            raise ValueError(f"Model '{clean}' is not discovered")
        if probe != "PASSED":
            raise ValueError(
                f"Model '{clean}' probe status is {probe} (expected PASSED)"
            )
        if not supports:
            raise ValueError(f"Model '{clean}' does not support structured output")
        # Probe TTL: stale probes require re-check.
        last = _as_utc(getattr(row, "last_checked_at", None))
        if last is None:
            raise ValueError(f"Model '{clean}' probe is stale (never checked)")
        age = (checked_at - last).total_seconds()
        if age > PROBE_TTL_SECONDS:
            raise ValueError(f"Model '{clean}' probe is stale (age {int(age)}s > {PROBE_TTL_SECONDS}s)")
        return row

    research_row = validated(research_model)
    summary_row = validated(summary_model)
    return snapshot(
        research_row.model_id,
        summary_row.model_id,
        protocol=research_row.protocol,
        status_value="available",
        catalog_time=_as_utc(getattr(research_row, "last_checked_at", None) or research_row.discovered_at),
    )
