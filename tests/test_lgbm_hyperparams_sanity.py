import pytest
from polyflip.settings_registry import registry_defaults

def test_num_leaves_consistent_with_max_depth():
    """
    num_leaves не должен превышать 2^max_depth.
    Если num_leaves=31 при max_depth=5 (2^5 = 32) — всё ок.
    """
    defaults = registry_defaults()
    max_depth = int(defaults.get("LGBM_MAX_DEPTH", 5))
    num_leaves = int(defaults.get("LGBM_NUM_LEAVES", 31))
    
    if max_depth > 0:
        max_possible_leaves = 2 ** max_depth
        assert num_leaves <= max_possible_leaves, (
            f"num_leaves={num_leaves} > 2^max_depth={max_possible_leaves}."
        )

def test_min_child_samples_not_too_high_for_dataset():
    defaults = registry_defaults()
    min_child = int(defaults.get("LGBM_MIN_CHILD_SAMPLES", 30))
    assert min_child <= 30
