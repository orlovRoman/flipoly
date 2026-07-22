from polyflip.crypto.risk_guard import check_funding_veto, FUNDING_EXTREME_THRESHOLD

def test_funding_extreme_computation():
    """Проверяет корректность расчета вето по экстремальной ставке фандинга (>= 0.0005 / 0.05%)."""
    # При экстремальной ставке 0.0006 с позицией по толпе -> vetoed == True
    veto_extreme = check_funding_veto(0.0006, "UP")
    assert veto_extreme.vetoed is True
    assert veto_extreme.stake_multiplier == 0.0

    # При нормальной ставке 0.0001 -> vetoed == False
    veto_normal = check_funding_veto(0.0001, "UP")
    assert veto_normal.vetoed is False
    assert veto_normal.stake_multiplier == 1.0
