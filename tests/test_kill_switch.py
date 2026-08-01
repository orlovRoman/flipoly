import pytest
from decimal import Decimal
from datetime import datetime, timezone
from sqlalchemy import select
from polyflip.db.models import RuntimeSettings
from polyflip.execution.risk_checks import check_risk_limits


@pytest.mark.asyncio
async def test_kill_switch_rejects_live_open(db_session):
    """При LIVE_TRADING_ENABLED=false check_risk_limits отклоняет LIVE OPEN."""
    setting = await db_session.scalar(
        select(RuntimeSettings).where(RuntimeSettings.key == "LIVE_TRADING_ENABLED")
    )
    if setting:
        setting.value = "false"
        setting.updated_at = datetime.now(timezone.utc)
        setting.updated_by = "system"
    else:
        db_session.add(
            RuntimeSettings(
                key="LIVE_TRADING_ENABLED",
                value="false",
                updated_at=datetime.now(timezone.utc),
                updated_by="system",
            )
        )
    await db_session.commit()

    risk_err = await check_risk_limits(
        db_session,
        intent="OPEN",
        max_spend_usdc=Decimal("10.0"),
        requested_mode="LIVE",
    )
    assert risk_err is not None
    assert "kill switch" in risk_err.lower()


@pytest.mark.asyncio
async def test_kill_switch_allows_close_even_when_off(db_session):
    """Закрытие позиций (CLOSE) разрешено даже при выключенном kill switch."""
    setting = await db_session.scalar(
        select(RuntimeSettings).where(RuntimeSettings.key == "LIVE_TRADING_ENABLED")
    )
    if setting:
        setting.value = "false"
        setting.updated_at = datetime.now(timezone.utc)
        setting.updated_by = "system"
    else:
        db_session.add(
            RuntimeSettings(
                key="LIVE_TRADING_ENABLED",
                value="false",
                updated_at=datetime.now(timezone.utc),
                updated_by="system",
            )
        )
    await db_session.commit()

    risk_err = await check_risk_limits(
        db_session,
        intent="CLOSE",
        max_spend_usdc=Decimal("10.0"),
        requested_mode="LIVE",
    )
    assert risk_err is None
