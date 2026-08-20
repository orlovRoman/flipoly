from datetime import datetime, timezone
from types import SimpleNamespace

from polyflip.api.ai_lab import _safe_text, verify_deployment_event_chain


def _event(previous, event_hash="hash", event_id=1):
    return SimpleNamespace(
        id=event_id,
        revision_id=7,
        event_type="CREATED",
        actor="test",
        reason="safe reason",
        payload={},
        previous_hash=previous,
        event_hash=event_hash,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_audit_text_truncates_tracebacks_and_redacts_keys():
    value = "api_key=super-secret Traceback (most recent call last):\nsecret details"
    result = _safe_text(value)
    assert "super-secret" not in result
    assert "secret details" not in result
    assert len(result) < 1000


def test_verify_deployment_event_chain_reports_previous_hash_break():
    first = _event("0" * 64, event_hash="not-a-real-service-hash")
    second = _event("wrong-predecessor", event_hash="also-invalid", event_id=2)
    result = verify_deployment_event_chain([first, second])
    assert result["valid"] is False
    assert {item["reason"] for item in result["broken_events"]} == {
        "previous_hash_mismatch",
        "event_hash_mismatch",
    }
