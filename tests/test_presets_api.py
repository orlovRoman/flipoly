import json
import asyncio
from polyflip.db.connection import async_session
from polyflip.services.preset_service import PresetService
from polyflip.db.models import ConfigPreset

async def run_test():
    async with async_session() as db:
        # Создание пресета через PresetService (эквивалентно POST /api/presets/)
        preset = await PresetService.save_preset(
            db=db,
            name="test_api_preset",
            description="API test preset",
            preset_type="manual",
            created_by="test_api"
        )
        assert preset.id is not None

        # Проверка diff
        current_snap = await PresetService.capture_snapshot(db)
        preset_snap = json.loads(preset.snapshot)
        assert isinstance(preset_snap, dict)
        assert len(preset_snap) > 0

        # Очистка
        preset.is_active = False
        await db.commit()
        print("OK: Presets API unit test PASSED!")

if __name__ == "__main__":
    asyncio.run(run_test())
