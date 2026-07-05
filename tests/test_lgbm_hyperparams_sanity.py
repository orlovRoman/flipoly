import pytest

def test_num_leaves_consistent_with_max_depth():
    """
    num_leaves не должен превышать 2^max_depth.
    При max_depth=4 → max 16 листьев.
    Если num_leaves=31, LightGBM молча игнорирует max_depth.
    """
    from polyflip.constants import LGBM_MAX_DEPTH, LGBM_NUM_LEAVES
    
    max_depth = LGBM_MAX_DEPTH
    num_leaves = LGBM_NUM_LEAVES
    
    if max_depth > 0:
        max_possible_leaves = 2 ** max_depth
        assert num_leaves <= max_possible_leaves, (
            f"num_leaves={num_leaves} > 2^max_depth={max_possible_leaves}. "
            f"Либо убери max_depth ограничение, либо уменьши num_leaves до {max_possible_leaves}"
        )

def test_min_child_samples_not_too_high_for_dataset():
    """
    min_child_samples=50 при малом датасете (< 500 строк тест-части)
    не даст модели обучиться на редких паттернах.
    """
    from polyflip.constants import LGBM_MIN_CHILD_SAMPLES
    min_child = LGBM_MIN_CHILD_SAMPLES
    # Для крипто-моделей с 15m свечами датасет обычно 3000-8000 строк
    # При 70/30 split тест = ~900-2400 строк. min_child=50 — на грани.
    assert min_child <= 30, (
        f"min_child_samples={min_child} слишком высокий для малых датасетов. "
        f"Рекомендуется ≤ 20 для начала"
    )
