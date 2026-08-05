import json
import pytest
from unittest.mock import AsyncMock
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.services.preset_service import PresetService
from polyflip.db.models import ConfigPreset, RuntimeSettings


def test_preset_contains_dead_zone_but_not_operational_switches():
    filtered = PresetService.sanitize_snapshot(
        {
            "AUTO_DEAD_ZONE": "false",
            "DEAD_ZONE_WIDTH": "0.05",
            "TRADE_FLIP_THRESHOLD": "0.85",
            "FLIP_THRESHOLD": "0.48",
            "TRADING_MODE": "combined",
            "TRADING_ENABLED": "true",
            "EXECUTION_MODE": "PAPER",
            "LIVE_TRADING_ENABLED": "false",
            "LIVE_MIRROR_ENABLED": "false",
            "LIVE_RELEASE_MODE": "DISABLED",
        }
    )

    assert filtered["AUTO_DEAD_ZONE"] == "false"
    assert filtered["DEAD_ZONE_WIDTH"] == "0.05"
    assert filtered["TRADE_FLIP_THRESHOLD"] == "0.85"
    assert filtered["FLIP_THRESHOLD"] == "0.48"
    assert filtered["TRADING_MODE"] == "combined"
    assert "TRADING_ENABLED" not in filtered
    assert "EXECUTION_MODE" not in filtered
    assert "LIVE_TRADING_ENABLED" not in filtered
    assert "LIVE_MIRROR_ENABLED" not in filtered
    assert "LIVE_RELEASE_MODE" not in filtered


@pytest.mark.asyncio
async def test_restore_preset_updates_auto_dead_zone():
    mock_preset = ConfigPreset(
        id=9,
        name="mz 1",
        snapshot=json.dumps(
            {
                "AUTO_DEAD_ZONE": "false",
                "DEAD_ZONE_WIDTH": "0.01",
                "FLIP_THRESHOLD": "0.48",
                "TRADING_ENABLED": "true",
                "EXECUTION_MODE": "PAPER",
            }
        ),
        is_active=True,
    )

    row_auto_dz = RuntimeSettings(key="AUTO_DEAD_ZONE", value="true")
    row_dz_width = RuntimeSettings(key="DEAD_ZONE_WIDTH", value="0.05")
    row_flip = RuntimeSettings(key="FLIP_THRESHOLD", value="0.50")

    settings_map = {
        "AUTO_DEAD_ZONE": row_auto_dz,
        "DEAD_ZONE_WIDTH": row_dz_width,
        "FLIP_THRESHOLD": row_flip,
    }

    mock_db = AsyncMock(spec=AsyncSession)
    mock_db.get.side_effect = lambda model, ident: (
        mock_preset
        if model == ConfigPreset and ident == 9
        else settings_map.get(ident)
    )

    changed, updated_params = await PresetService.restore_preset(
        mock_db, 9, restored_by="test"
    )

    assert changed == 3
    assert updated_params["AUTO_DEAD_ZONE"] == "false"
    assert updated_params["DEAD_ZONE_WIDTH"] == "0.01"
    assert updated_params["FLIP_THRESHOLD"] == "0.48"
    assert "TRADING_ENABLED" not in updated_params
    assert "EXECUTION_MODE" not in updated_params

    assert row_auto_dz.value == "false"
    assert row_dz_width.value == "0.01"
    assert row_flip.value == "0.48"


def test_diff_preset_logic():
    preset_raw = {
        "AUTO_DEAD_ZONE": "false",
        "DEAD_ZONE_WIDTH": "0.01",
        "FLIP_THRESHOLD": "0.48",
        "TRADING_ENABLED": "true",
        "UNKNOWN_KEY": "foo",
    }
    current_settings = {
        "AUTO_DEAD_ZONE": "true",
        "DEAD_ZONE_WIDTH": "0.05",
        "FLIP_THRESHOLD": "0.48",
    }

    preset_snap = PresetService.sanitize_snapshot(preset_raw)
    current_snap = PresetService.sanitize_snapshot(current_settings)

    diff = {}
    for key in sorted(set(preset_snap) | set(current_snap)):
        preset_value = preset_snap.get(key)
        current_value = current_snap.get(key)
        if str(preset_value) != str(current_value):
            diff[key] = {
                "preset": preset_value,
                "current": current_value,
            }

    assert "AUTO_DEAD_ZONE" in diff
    assert diff["AUTO_DEAD_ZONE"] == {"preset": "false", "current": "true"}
    assert "DEAD_ZONE_WIDTH" in diff
    assert diff["DEAD_ZONE_WIDTH"] == {"preset": "0.01", "current": "0.05"}
    assert "FLIP_THRESHOLD" not in diff
    assert "TRADING_ENABLED" not in diff
    assert "UNKNOWN_KEY" not in diff
