from polyflip.settings_registry import registry_defaults, registry_keys


def test_weighted_runtime_controls_have_registry_defaults():
    defaults = registry_defaults()
    expected = {
        "WEIGHTED_MODELS_AGREE_BETA": "0.0",
        "WEIGHTED_MRF_APPLICATION": "PROBABILITY",
        "WEIGHTED_MRF_SIZING_GAMMA": "0.0",
        "WEIGHTED_POLICY_ARTIFACT_PATH": "",
        "WEIGHTED_SIZING_MODE": "FIXED",
        "WEIGHTED_STANDARD_ERROR": "0.0",
        "WEIGHTED_KELLY_FRACTION": "0.025",
        "WEIGHTED_SIZE_CAP_USDC": "3.0",
    }
    assert set(expected).issubset(registry_keys())
    assert {key: defaults[key] for key in expected} == expected
