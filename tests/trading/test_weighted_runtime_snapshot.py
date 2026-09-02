import scripts.weighted_policy_runtime_snapshot as runtime_snapshot
from scripts.weighted_policy_runtime_snapshot import build_snapshot


def test_runtime_snapshot_captures_weighted_controls_without_secrets(monkeypatch):
    monkeypatch.setenv("TRADING_POLICY_MODE", "WEIGHTED_SHADOW")
    monkeypatch.setenv("WEIGHTED_SIZING_MODE", "LOWER_BOUND_KELLY")
    monkeypatch.setenv("WEIGHTED_STANDARD_ERROR", "0.12")
    monkeypatch.setenv("PAPER_FEE_MODEL", "POLYMARKET_PRICE_DEPENDENT")
    monkeypatch.setenv("OPENROUTER_API_KEY", "do-not-copy")
    snapshot = build_snapshot()
    environment = snapshot["safe_environment"]
    assert environment["TRADING_POLICY_MODE"] == "WEIGHTED_SHADOW"
    assert environment["WEIGHTED_SIZING_MODE"] == "LOWER_BOUND_KELLY"
    assert environment["WEIGHTED_STANDARD_ERROR"] == "0.12"
    assert environment["PAPER_FEE_MODEL"] == "POLYMARKET_PRICE_DEPENDENT"
    assert "OPENROUTER_API_KEY" not in environment
    assert snapshot["active_models"] == []
    assert snapshot["active_models_source"] == "not_requested"
    assert snapshot["secrets_omitted"] is True


def test_runtime_snapshot_uses_build_metadata_when_git_is_not_in_image(monkeypatch):
    monkeypatch.setenv("POLYFLIP_BUILD_SHA", "abc123")
    monkeypatch.setenv("POLYFLIP_BUILD_BRANCH", "codex/weighted-trading-policy")
    monkeypatch.setattr(
        runtime_snapshot.subprocess,
        "check_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            runtime_snapshot.subprocess.CalledProcessError(1, "git")
        ),
    )
    snapshot = build_snapshot()
    assert snapshot["git"] == {
        "sha": "abc123",
        "branch": "codex/weighted-trading-policy",
    }
