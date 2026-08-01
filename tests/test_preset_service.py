import json
import asyncio
from polyflip.db.connection import async_session
from polyflip.services.preset_service import PresetService

async def run_test():
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
        print(f"OK: Preset created id={preset.id}, params={len(snap)}")

        # 2. Восстановление
        changed = await PresetService.restore_preset(db, preset.id, restored_by="pytest")
        assert changed >= 0
        print(f"OK: Preset restored, changed={changed}")

        # 3. ATH-проверка (первый запуск)
        ath = await PresetService.check_and_save_ath(db, current_capital=1500.0, current_pnl=500.0, min_pnl_diff=1.0, min_interval_hours=0)
        if ath:
            assert ath.preset_type in ["ath_capital", "ath_pnl"]
            print(f"OK: ATH preset saved name={ath.name}")

        # 4. Софт-удаление
        preset.is_active = False
        await db.commit()
        print("OK: All PresetService lifecycle checks PASSED!")

if __name__ == "__main__":
    asyncio.run(run_test())
