from scripts.weighted_policy_benchmark import _policy_config


def test_benchmark_policy_config_matches_runtime_mrf_controls(monkeypatch):
    monkeypatch.setenv("WEIGHTED_MODELS_AGREE_BETA", "0.35")
    monkeypatch.setenv("WEIGHTED_MRF_APPLICATION", "stake")
    monkeypatch.setenv("WEIGHTED_MRF_SIZING_GAMMA", "0.4")

    config = _policy_config()

    assert config.models_agree_beta == 0.35
    assert config.mrf_application == "STAKE"
    assert config.mrf_sizing_gamma == 0.4


def test_benchmark_policy_config_bounds_invalid_mrf_controls(monkeypatch):
    monkeypatch.setenv("WEIGHTED_MODELS_AGREE_BETA", "9")
    monkeypatch.setenv("WEIGHTED_MRF_APPLICATION", "unsupported")
    monkeypatch.setenv("WEIGHTED_MRF_SIZING_GAMMA", "-9")

    config = _policy_config()

    assert config.models_agree_beta == 2.0
    assert config.mrf_application == "PROBABILITY"
    assert config.mrf_sizing_gamma == -1.0
