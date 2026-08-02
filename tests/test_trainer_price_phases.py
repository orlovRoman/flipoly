import pytest
from datetime import datetime, timezone
from unittest.mock import patch
from sqlalchemy import select

from polyflip.config import settings
from polyflip.db.models import MarketSnapshot, ModelRegistry, RuntimeSettings
from polyflip.models.trainer import ModelTrainer, _fit_and_serialize


@pytest.mark.asyncio
async def test_trainer_creates_phase_models_with_sufficient_data(db_session):
    """
    Проверяем, что при наличии достаточного числа рынков и снимков в фазе (например, decided: mid_price >= 0.75 или <= 0.25)
    создается запись в ModelRegistry с соответствующим суффиксом (например, BTC_decided).
    """
    snaps = []
    # Создаем 6 разных рынков (> CV_N_SPLITS=5), в каждом по 10 снимков
    for m_idx in range(6):
        market_id = f"test_market_decided_{m_idx}"
        for i in range(10):
            # mid_price = 0.85 -> |0.85 - 0.5| = 0.35 -> phase 'decided'
            snaps.append(MarketSnapshot(
                market_id=market_id,
                asset="BTC",
                time_left_min=float(i + 1),
                mid_price=0.85,
                spread=0.01,
                volume_5min=1000.0,
                price_velocity=0.0,
                hour_of_day=12,
                final_outcome="YES" if (m_idx % 2 == 0) else "NO",
                flip_vs_final=True if (i % 2 == 0) else False,
                recorded_at=datetime.now(timezone.utc)
            ))
    db_session.add_all(snaps)
    # Устанавливаем минимальные требования для фазовых моделей в тесте
    db_session.add(RuntimeSettings(key="MIN_SAMPLES_FOR_PHASE_MODEL", value="10", updated_at=datetime.now(timezone.utc), updated_by="test"))
    db_session.add(RuntimeSettings(key="LR_MIN_AUC_FOR_DEPLOY", value="0.40", updated_at=datetime.now(timezone.utc), updated_by="test"))
    db_session.add(RuntimeSettings(key="BACKTEST_MIN_TRADES", value="1", updated_at=datetime.now(timezone.utc), updated_by="test"))
    db_session.add(RuntimeSettings(key="BACKTEST_MIN_PNL", value="-1000.0", updated_at=datetime.now(timezone.utc), updated_by="test"))
    await db_session.commit()

    trainer = ModelTrainer(db_session)
    with patch.object(settings, "MIN_SAMPLES_FOR_MODEL", 10):
        res = await trainer.train_model("BTC")

    assert res is True

    # Проверяем, что создалась запись BTC_decided
    stmt = select(ModelRegistry).where(ModelRegistry.asset == "BTC_decided")
    phase_models = (await db_session.execute(stmt)).scalars().all()
    assert len(phase_models) >= 1
    assert phase_models[0].asset == "BTC_decided"
    assert phase_models[0].is_active is True
    assert phase_models[0].model_blob is not None

    # Проверяем, что в status_messages отражены фазы
    assert "BTC" in trainer.status_messages
    assert "Фазы:" in trainer.status_messages["BTC"]
    assert "decided: ok" in trainer.status_messages["BTC"]


@pytest.mark.asyncio
async def test_fit_and_serialize_receives_nonempty_phase_data(db_session):
    """
    Проверяем, что при вызове _fit_and_serialize для фазовой модели
    передаются непустые X_phase, y_phase и grp_phase.
    """
    snaps = []
    for m_idx in range(6):
        market_id = f"test_market_{m_idx}"
        for i in range(10):
            snaps.append(MarketSnapshot(
                market_id=market_id,
                asset="BTC",
                time_left_min=float(i + 1),
                mid_price=0.50, # phase 'contested'
                spread=0.01,
                volume_5min=1000.0,
                price_velocity=0.0,
                hour_of_day=12,
                final_outcome="YES" if (m_idx % 2 == 0) else "NO",
                flip_vs_final=True if (i % 2 == 0) else False,
                recorded_at=datetime.now(timezone.utc)
            ))
    db_session.add_all(snaps)
    db_session.add(RuntimeSettings(key="MIN_SAMPLES_FOR_PHASE_MODEL", value="10", updated_at=datetime.now(timezone.utc), updated_by="test"))
    db_session.add(RuntimeSettings(key="LR_MIN_AUC_FOR_DEPLOY", value="0.40", updated_at=datetime.now(timezone.utc), updated_by="test"))
    db_session.add(RuntimeSettings(key="BACKTEST_MIN_TRADES", value="1", updated_at=datetime.now(timezone.utc), updated_by="test"))
    db_session.add(RuntimeSettings(key="BACKTEST_MIN_PNL", value="-1000.0", updated_at=datetime.now(timezone.utc), updated_by="test"))
    await db_session.commit()

    trainer = ModelTrainer(db_session)
    
    fit_calls = []
    original_fit = _fit_and_serialize

    def spy_fit(X, y, groups, **kwargs):
        fit_calls.append({
            "X_len": len(X),
            "y_len": len(y),
            "grp_len": len(groups),
            "X_cols": list(X.columns),
        })
        return original_fit(X, y, groups, **kwargs)

    with patch("polyflip.models.trainer._fit_and_serialize", side_effect=spy_fit), \
         patch.object(settings, "MIN_SAMPLES_FOR_MODEL", 10):
        res = await trainer.train_model("BTC")

    assert res is True
    # Первый вызов — базовая модель, второй вызов — фазовая contested
    assert len(fit_calls) >= 2
    phase_call = fit_calls[1]
    assert phase_call["X_len"] > 0
    assert phase_call["y_len"] > 0
    assert phase_call["grp_len"] > 0
    assert len(phase_call["X_cols"]) > 0


@pytest.mark.asyncio
async def test_phase_error_reflected_in_status_messages(db_session):
    """
    Проверяем, что исключение внутри обучения конкретной фазы не проглатывается молча,
    а попадает в status_messages.
    """
    snaps = []
    for m_idx in range(6):
        market_id = f"test_market_err_{m_idx}"
        for i in range(10):
            snaps.append(MarketSnapshot(
                market_id=market_id,
                asset="BTC",
                time_left_min=float(i + 1),
                mid_price=0.50,
                spread=0.01,
                volume_5min=1000.0,
                price_velocity=0.0,
                hour_of_day=12,
                final_outcome="YES" if (m_idx % 2 == 0) else "NO",
                flip_vs_final=True if (i % 2 == 0) else False,
                recorded_at=datetime.now(timezone.utc)
            ))
    db_session.add_all(snaps)
    db_session.add(RuntimeSettings(key="MIN_SAMPLES_FOR_PHASE_MODEL", value="10", updated_at=datetime.now(timezone.utc), updated_by="test"))
    db_session.add(RuntimeSettings(key="LR_MIN_AUC_FOR_DEPLOY", value="0.40", updated_at=datetime.now(timezone.utc), updated_by="test"))
    await db_session.commit()

    trainer = ModelTrainer(db_session)
    call_idx = [0]
    original_fit = _fit_and_serialize

    def fit_with_phase_error(X, y, groups, **kwargs):
        call_idx[0] += 1
        if call_idx[0] == 2:  # ошибка на фазовой модели
            raise RuntimeError("Simulated phase training crash")
        return original_fit(X, y, groups, **kwargs)

    with patch("polyflip.models.trainer._fit_and_serialize", side_effect=fit_with_phase_error), \
         patch.object(settings, "MIN_SAMPLES_FOR_MODEL", 10):
        res = await trainer.train_model("BTC")

    assert res is True
    assert "BTC" in trainer.status_messages
    assert "failed: Simulated phase training crash" in trainer.status_messages["BTC"]
