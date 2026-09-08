import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from polyflip.models.outsider_feature_sets import (
    MODEL_A_FEATURES,
    OUTSIDER_FEATURE_SETS,
    get_outsider_feature_set,
)


def test_outsider_feature_sets_contracts():
    """1.8: Feature contracts must be immutable and deterministic."""
    model_a = get_outsider_feature_set("MODEL_A")
    assert model_a.features == MODEL_A_FEATURES
    assert model_a.features == ("mid_price", "time_left_min", "spread")
    assert len(model_a.schema_hash) == 16

    legacy = get_outsider_feature_set("LEGACY")
    assert "mid_price" in legacy.features
    assert "spread" in legacy.features


def test_target_alignment_truth_table():
    """1.7: Canonical target truth table for favourite x final outcome."""
    # favourite = YES if mid_price > 0.5 else NO (mid != 0.5)
    # flip = True if favourite != final_outcome else False
    cases = [
        # (mid_price, final_outcome, expected_target, desc)
        (0.60, "NO", 1, "favourite YES loses -> target 1"),
        (0.60, "YES", 0, "favourite YES wins -> target 0"),
        (0.40, "YES", 1, "favourite NO loses -> target 1"),
        (0.40, "NO", 0, "favourite NO wins -> target 0"),
    ]
    for mid, final_outcome, expected_target, desc in cases:
        target = 1 if ((mid > 0.5) != (final_outcome == "YES")) else 0
        assert target == expected_target, desc


def test_mid_05_exclusion():
    """1.7: mid_price == 0.5 must be excluded from target decision rows."""
    mid_prices = pd.Series([0.60, 0.50, 0.40, 0.50, 0.70])
    valid_mask = mid_prices != 0.5
    assert valid_mask.tolist() == [True, False, True, False, True]


def test_candidate_side_consistency():
    """1.7: Candidate trading side matches target definition."""
    # If target == 1 (flip), buy outsider. If target == 0, no flip.
    # When mid_price > 0.5, favourite is YES, outsider is NO.
    # When mid_price < 0.5, favourite is NO, outsider is YES.
    test_cases = [
        (0.65, "NO", 1, "NO"),   # favourite is YES, loses -> buy NO
        (0.65, "YES", 0, "NO"),  # favourite is YES, wins -> no flip
        (0.35, "YES", 1, "YES"), # favourite is NO, loses -> buy YES
        (0.35, "NO", 0, "YES"),  # favourite is NO, wins -> no flip
    ]
    for mid, final_outcome, target, expected_outsider in test_cases:
        favourite_side = "YES" if mid > 0.5 else "NO"
        outsider_side = "NO" if favourite_side == "YES" else "YES"
        actual_flip = 1 if favourite_side != final_outcome else 0
        assert actual_flip == target
        assert outsider_side == expected_outsider


@pytest.mark.asyncio
async def test_trainer_contract_locks_features():
    """1.8: Passing MODEL_A locks active_features and prevents auto-expansion."""
    from polyflip.models.trainer import ModelTrainer
    from polyflip.db.models import MarketSnapshot

    now = datetime.now(timezone.utc)
    mock_db = AsyncMock()

    # Mock count query returning sufficient samples
    mock_db.execute.return_value.scalar.return_value = 100

    # Create mock snapshots for 20 markets across time (20 * 4 = 80 samples >= 50)
    snaps = []
    for m_idx in range(20):
        m_id = f"m_{m_idx}"
        is_even = (m_idx % 2 == 0)
        # Flip occurs on markets where m_idx % 4 in (0, 1)
        has_flip = (m_idx % 4 in (0, 1))
        mid = 0.65 if is_even else 0.35
        if has_flip:
            outcome = "NO" if is_even else "YES"
        else:
            outcome = "YES" if is_even else "NO"

        market_base_time = now - timedelta(hours=30 - m_idx)
        for t_min in [14.0, 10.0, 5.0, 2.0]:
            s = MarketSnapshot(
                id=len(snaps) + 1,
                market_id=m_id,
                asset="BTC",
                recorded_at=market_base_time + timedelta(minutes=15 - int(t_min)),
                time_left_min=t_min,
                mid_price=mid,
                spread=0.01,
                best_bid=mid - 0.01,
                best_ask=mid + 0.01,
                price_velocity=0.0,
                volume_5min=100.0,
                hour_of_day=12,
                final_outcome=outcome,
                flip_vs_final=has_flip,
            )
            snaps.append(s)

    mock_scalars = MagicMock()
    mock_scalars.all.return_value = snaps

    async def fake_execute(stmt):
        res = MagicMock()
        stmt_str = str(stmt).lower()
        if "count" in stmt_str:
            res.scalar.return_value = len(snaps)
        elif "runtimesettings" in stmt_str:
            res.scalar_one_or_none.return_value = None
        else:
            res.scalars.return_value = mock_scalars
        return res

    mock_db.execute = AsyncMock(side_effect=fake_execute)

    trainer = ModelTrainer(mock_db)

    from polyflip.models.trainer import _fit_and_serialize as real_fit
    captured_X = None
    captured_y = None

    def spy_fit(X, y, *args, **kwargs):
        nonlocal captured_X, captured_y
        captured_X = X.copy()
        captured_y = y.copy()
        return real_fit(X, y, *args, **kwargs)

    def mock_float(db, k):
        if "min_time" in k.lower():
            return 1.0
        if "max_time" in k.lower():
            return 12.0
        if "min_auc" in k.lower():
            return 0.50
        return 0.05

    with patch("polyflip.models.trainer._fit_and_serialize", side_effect=spy_fit), \
         patch("polyflip.services.settings_service.get_float", AsyncMock(side_effect=mock_float)), \
         patch("polyflip.services.settings_service.get_int", AsyncMock(return_value=2)), \
         patch("polyflip.services.settings_service.get_setting", AsyncMock(return_value="uniform")):

        # Train with MODEL_A
        res = await trainer.train_model(
            asset="BTC",
            save_settings=False,
            feature_set="MODEL_A",
            activate_after_train=False,
        )

        assert res is True, f"Failed with: {trainer.status_messages.get('BTC')}"
        assert captured_X is not None
        # Must only contain MODEL_A features, NO DERIVED FEATURES auto-expansion!
        assert list(captured_X.columns) == list(MODEL_A_FEATURES)
        assert len(captured_X) > 0
        assert len(captured_y) == len(captured_X)
