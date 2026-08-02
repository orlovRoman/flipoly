import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.api.presets import diff_preset
from polyflip.db.models import ConfigPreset, RuntimeSettings


@pytest.mark.asyncio
async def test_api_diff_preset_sanitization():
    mock_preset = ConfigPreset(
        id=1,
        name="test_preset",
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

    mock_db = AsyncMock(spec=AsyncSession)
    mock_db.get.return_value = mock_preset

    # Mock rows for capture_snapshot
    row1 = RuntimeSettings(key="AUTO_DEAD_ZONE", value="true")
    row2 = RuntimeSettings(key="DEAD_ZONE_WIDTH", value="0.05")
    row3 = RuntimeSettings(key="FLIP_THRESHOLD", value="0.48")
    row4 = RuntimeSettings(key="TRADING_ENABLED", value="false")

    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [
        row1,
        row2,
        row3,
        row4,
    ]
    mock_db.execute.return_value = mock_result

    res = await diff_preset(1, db=mock_db)

    assert res["preset_id"] == 1
    diff = res["diff"]
    assert "AUTO_DEAD_ZONE" in diff
    assert diff["AUTO_DEAD_ZONE"] == {"preset": "false", "current": "true"}
    assert "DEAD_ZONE_WIDTH" in diff
    assert diff["DEAD_ZONE_WIDTH"] == {"preset": "0.01", "current": "0.05"}
    assert "FLIP_THRESHOLD" not in diff
    assert "TRADING_ENABLED" not in diff
    assert "EXECUTION_MODE" not in diff
