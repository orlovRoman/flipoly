from types import SimpleNamespace

import pytest

from polyflip.ai_lab.llm import (
    DEFAULT_OPENCODE_CHAT_ENDPOINT,
    DEFAULT_OPENCODE_ENDPOINT,
    OpenAIResponsesProvider,
    get_llm_model_catalog,
    get_llm_provider,
    normalize_llm_selection,
)


def test_catalog_contains_safe_provider_metadata(monkeypatch):
    import polyflip.config as config

    monkeypatch.setattr(
        config,
        "settings",
        SimpleNamespace(
            AI_LAB_LLM_PROVIDER="opencode",
            AI_LAB_MODEL_RESEARCH="gpt-5.6-sol",
            AI_LAB_MODEL_SUMMARY="gpt-5.6-luna",
            AI_LAB_LLM_AVAILABLE_PROVIDERS="mock,opencode",
            AI_LAB_ALLOWED_MODELS="",
            AI_LAB_LLM_API_KEY="secret-value",
            OPENAI_API_KEY="",
        ),
    )
    catalog = get_llm_model_catalog()
    assert catalog["provider"] == "opencode"
    assert {item["id"] for item in catalog["models"]} == {
        "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
        "muse-spark-1.2-contributor-free", "big-pickle", "nemotron-3-ultra-free",
    }
    labels = {item["id"]: item["label"] for item in catalog["models"]}
    assert labels["big-pickle"] == "Big Pickle"
    assert labels["muse-spark-1.2-contributor-free"] == "Muse Spark 1.2 Free"
    assert labels["nemotron-3-ultra-free"] == "Nemotron 3 Ultra Free"
    assert catalog["providers"][-1]["configured"] is True
    assert "secret-value" not in str(catalog)


def test_selection_is_validated_against_provider_catalog(monkeypatch):
    import polyflip.config as config

    monkeypatch.setattr(
        config,
        "settings",
        SimpleNamespace(
            AI_LAB_LLM_PROVIDER="mock",
            AI_LAB_MODEL_RESEARCH="gpt-5.6",
            AI_LAB_MODEL_SUMMARY="gpt-5.6-mini",
            AI_LAB_LLM_AVAILABLE_PROVIDERS="mock,openai,opencode",
            AI_LAB_ALLOWED_MODELS="",
            AI_LAB_LLM_API_KEY="",
            OPENAI_API_KEY="",
        ),
    )
    assert normalize_llm_selection("mock", None, None) == ("mock", "mock-gpt-5", "mock-gpt-5")
    with pytest.raises(ValueError, match="Unknown model"):
        normalize_llm_selection("mock", "gpt-5.6-sol", "mock-gpt-5")


def test_opencode_factory_uses_compatible_responses_endpoint(monkeypatch):
    import polyflip.config as config

    monkeypatch.setattr(
        config,
        "settings",
        SimpleNamespace(
            AI_LAB_LLM_PROVIDER="opencode",
            AI_LAB_MODEL_RESEARCH="gpt-5.6-sol",
            AI_LAB_MODEL_SUMMARY="gpt-5.6-luna",
            AI_LAB_LLM_API_KEY="universal-key",
            OPENAI_API_KEY="",
            AI_LAB_LLM_ENDPOINT="",
        ),
    )
    provider = get_llm_provider("opencode")
    assert isinstance(provider, OpenAIResponsesProvider)
    assert provider.provider_name == "opencode"
    assert provider.endpoint_url == DEFAULT_OPENCODE_ENDPOINT
    assert provider.route_opencode_models is True
    assert provider.model_research == "gpt-5.6-sol"
    assert provider._endpoint_for_model("gpt-5.6-sol") == DEFAULT_OPENCODE_ENDPOINT
    assert provider._endpoint_for_model("big-pickle") == DEFAULT_OPENCODE_CHAT_ENDPOINT

def test_opencode_factory_requires_key(monkeypatch):
    import polyflip.config as config

    monkeypatch.setattr(
        config,
        "settings",
        SimpleNamespace(
            AI_LAB_LLM_PROVIDER="opencode",
            AI_LAB_LLM_API_KEY="",
            OPENAI_API_KEY="",
            AI_LAB_MODEL_RESEARCH="gpt-5.6-sol",
            AI_LAB_MODEL_SUMMARY="gpt-5.6-luna",
            AI_LAB_LLM_ENDPOINT="",
        ),
    )
    with pytest.raises(RuntimeError, match="AI_LAB_LLM_API_KEY"):
        get_llm_provider("opencode")


def test_run_request_persists_selected_llm_fields():
    from polyflip.api.ai_lab import RunCreateRequest

    request = RunCreateRequest(
        objective="Compare a selected provider",
        mode="RESEARCH",
        config_ids=[1],
        llm_provider="opencode",
        research_model="gpt-5.6-sol",
        summary_model="gpt-5.6-luna",
    )
    assert request.llm_provider == "opencode"
    assert request.research_model == "gpt-5.6-sol"
    assert request.summary_model == "gpt-5.6-luna"