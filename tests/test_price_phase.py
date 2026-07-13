from polyflip.constants import get_price_phase, PRICE_PHASE_BOUNDARIES

def test_phase_boundaries():
    assert get_price_phase(0.50) == "contested"   # dev=0.00 → нижняя граница
    assert get_price_phase(0.59) == "contested"   # dev=0.09
    assert get_price_phase(0.60) == "leaning"     # dev=0.10 → граница включительно
    assert get_price_phase(0.74) == "leaning"     # dev=0.24
    assert get_price_phase(0.75) == "decided"     # dev=0.25
    assert get_price_phase(0.99) == "decided"     # dev=0.49
    assert get_price_phase(0.01) == "decided"     # dev=0.49, симметрия
    assert get_price_phase(0.40) == "leaning"     # dev=0.10 → leaning

def test_phase_symmetry():
    # Рынок симметричен: цена 0.3 и 0.7 должны давать одну фазу
    assert get_price_phase(0.30) == get_price_phase(0.70)
    assert get_price_phase(0.20) == get_price_phase(0.80)
    assert get_price_phase(0.10) == get_price_phase(0.90)
