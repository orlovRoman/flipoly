import pytest
import pickle
from datetime import datetime, timezone, timedelta
from sqlalchemy import select
from unittest.mock import patch, AsyncMock
from polyflip.db.models import RuntimeSettings, LiveMarket, ModelRegistry, TradeHistory
from polyflip.trading.engine import trade_worker_cycle

class MockModel:

    def __init__(self, proba):
        self.proba = proba
        self.feature_names_in_ = ['mid_price']

    def predict_proba(self, X):
        return [self.proba]

@pytest.mark.asyncio
async def test_recommended_thresholds_api(db_session):
    from polyflip.api.settings import get_recommended_thresholds
    now = datetime.now(timezone.utc)
    db_settings = [RuntimeSettings(key='DEAD_ZONE_WIDTH', value='0.15', updated_at=now, updated_by='test'), RuntimeSettings(key='TRADE_NO_FLIP_THRESHOLD', value='0.20', updated_at=now, updated_by='test'), RuntimeSettings(key='AUTO_FLIP_THRESHOLD_BTC', value='0.65', updated_at=now, updated_by='test')]
    db_session.add_all(db_settings)
    await db_session.commit()

    class DummyAsyncContextManager:

        def __init__(self, session):
            self.session = session

        async def __aenter__(self):
            return self.session

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    def dummy_session_creator():
        return DummyAsyncContextManager(db_session)
    from polyflip.config import Settings
    from unittest.mock import PropertyMock
    with patch('polyflip.api.settings.async_session', dummy_session_creator), patch.object(Settings, 'asset_list', new_callable=PropertyMock) as mock_prop:
        mock_prop.return_value = ['BTC', 'ETH']
        response = await get_recommended_thresholds()
        assert response['global']['dead_zone'] == 0.15
        assert response['global']['current_no_flip'] == 0.2
        assert 'BTC' in response['per_asset']
        assert response['per_asset']['BTC']['flip_threshold'] == 0.65
        assert response['per_asset']['BTC']['recommended_no_flip'] == 0.5
        assert response['per_asset']['BTC']['is_auto_calibrated'] is True
        assert 'ETH' not in response['per_asset'], 'Для ETH порога в БД нет, не должно быть в per_asset'