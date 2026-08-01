import pytest
import asyncio
import polyflip.api.analytics as analytics_module

@pytest.mark.asyncio
async def test_invalidate_clears_cache():
    """invalidate_analytics_cache сбрасывает time_left_dist_cache."""
    analytics_module._time_left_dist_cache = {"BTC": {"n": 100}}
    analytics_module._time_left_dist_cache_time = 9999999.0
    await analytics_module.invalidate_analytics_cache()
    assert analytics_module._time_left_dist_cache is None

@pytest.mark.asyncio
async def test_invalidate_is_safe_to_call_multiple_times():
    """Повторный вызов invalidate не бросает исключений."""
    for _ in range(10):
        await analytics_module.invalidate_analytics_cache()
