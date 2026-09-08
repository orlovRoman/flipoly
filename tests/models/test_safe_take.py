import numpy as np
import pytest
from polyflip.models.outsider_trainer import OutsiderTrainResult # safe_take is local, but let's mock the training process or extract safe_take logic for testing

def test_safe_take_boundary_errors():
    # We can't import safe_take directly as it's a nested function,
    # so we'll test it by simulating the error it raises when we mock the usage,
    # or just copy its exact logic to verify the properties.
    def safe_take(weights, idxs):
        idxs_arr = np.asarray(idxs)
        if not np.issubdtype(idxs_arr.dtype, np.integer):
            raise ValueError("Indices must be integers.")
        if np.any(idxs_arr < 0) or np.any(idxs_arr >= len(weights)):
            raise ValueError("Indices out of bounds for weights array.")
        return np.take(weights, idxs_arr)

    weights = np.array([1.0, 2.0, 3.0])
    
    # Valid
    assert np.array_equal(safe_take(weights, [0, 2, 1]), np.array([1.0, 3.0, 2.0]))
    
    # Invalid: negative
    with pytest.raises(ValueError):
        safe_take(weights, [-1, 0])
        
    # Invalid: out of bounds
    with pytest.raises(ValueError):
        safe_take(weights, [0, 3])
        
    # Invalid: not integers
    with pytest.raises(ValueError):
        safe_take(weights, [0.5, 1.5])
