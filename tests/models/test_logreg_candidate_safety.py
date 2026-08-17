import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from polyflip.models.trainer import ModelTrainer


@pytest.mark.asyncio
async def test_logreg_retrain_never_activates_candidate_by_default():
    """Verify that retrained LogReg models in candidate mode are NEVER active by default."""
    session = AsyncMock()
    trainer = ModelTrainer(session)

    # Mock DB counts and data
    session.execute = AsyncMock()
    
    # Check default parameters
    import inspect
    sig = inspect.signature(trainer.train_model)
    assert sig.parameters["activate_after_train"].default is False
    assert sig.parameters["candidate_mode"].default is True
    assert sig.parameters["explicit_manual_activation"].default is False
