import pytest
import pickle
from datetime import datetime, timezone, timedelta
from sqlalchemy import select
from unittest.mock import MagicMock, AsyncMock
from polyflip.db.models import RuntimeSettings, LiveMarket, ModelRegistry, TradeHistory
from polyflip.trading.engine import trade_worker_cycle

class DynamicMockModel:

    def __init__(self, prob_yes: float=0.5):
        self.prob_yes = prob_yes
        self.feature_names_in_ = ['mid_price']

    def predict_proba(self, X):
        return [[1.0 - self.prob_yes, self.prob_yes]]