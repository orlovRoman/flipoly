from polyflip.crypto.edge import compute_crypto_signal_strength


def test_down_threshold_is_lower_bound_on_p_up_and_overlap_is_rejected():
    assert compute_crypto_signal_strength(0.10, 0.70, 0.30)[1] == "DOWN"
    assert compute_crypto_signal_strength(0.50, 0.70, 0.30)[1] == "NONE"
    assert compute_crypto_signal_strength(0.90, 0.70, 0.30)[1] == "UP"
    assert compute_crypto_signal_strength(0.50, 0.30, 0.70)[1] == "NONE"
