import pytest
import json
import asyncio
from polyflip.db.connection import async_session
from polyflip.services.preset_service import PresetService
from polyflip.db.models import ConfigPreset, RuntimeSettings
from sqlalchemy import select

@pytest.mark.asyncio
async def test_preset_service_lifecycle():
    async with async_session() as db:
        # 1. Снятие и сохранение пресета
        preset = await PresetService.save_preset(
            db=db,
            name="test_preset_unit",
            description="Unit test preset",
            preset_type="manual",
            capital_at_save=1000.0,
            pnl_at_save=50.0,
            created_by="pytest"
        )
        assert preset.id is not None
        assert preset.name == "test_preset_unit"

        snap = json.loads(preset.snapshot)
        assert isinstance(snap, dict)

        # 2. Восстановление
        changed = await PresetService.restore_preset(db, preset.id, restored_by="pytest")
        assert changed >= 0

        # 3. ATH-проверка (первый запуск)
        ath = await PresetService.check_and_save_ath(db, current_capital=1500.0, current_pnl=500.0, min_pnl_diff=1.0, min_interval_hours=0)
        if ath:
            assert ath.preset_type in ["ath_capital", "ath_pnl"]

        # 4. Софт-удаление
        preset.is_active = False
        await db.commit()
