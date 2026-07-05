import pytest
from datetime import datetime, timezone
from polyflip.db.models import TradeHistory
from polyflip.api.dashboard import get_trade_logs

@pytest.mark.asyncio
async def test_trade_logs_pagination(db_session):
    # Создаём 60 записей TradeHistory
    now = datetime.now(timezone.utc)
    for i in range(60):
        t = TradeHistory(
            market_id=f"m_{i}",
            asset="BTC",
            outcome_bought="YES",
            amount_usdc=10.0,
            executed_price=0.5,
            predicted_flip_prob=0.8,
            active_features="test",
            status="SUCCESS",
            created_at=now
        )
        db_session.add(t)
    await db_session.commit()

    # GET /api/dashboard/trade_logs?page=1&page_size=25 → items=25, total=60, pages=3
    res1 = await get_trade_logs(db_session, page=1, page_size=25)
    assert len(res1["items"]) == 25
    assert res1["total"] == 60
    assert res1["pages"] == 3
    assert res1["page"] == 1
    assert res1["page_size"] == 25
    assert "edge" in res1["items"][0]

    # GET /api/dashboard/trade_logs?page=3&page_size=25 → items=10 (остаток)
    res3 = await get_trade_logs(db_session, page=3, page_size=25)
    assert len(res3["items"]) == 10
    assert res3["total"] == 60
    assert res3["pages"] == 3

    # GET /api/dashboard/trade_logs?page=99&page_size=25 → items=0
    res99 = await get_trade_logs(db_session, page=99, page_size=25)
    assert len(res99["items"]) == 0
    assert res99["total"] == 60


@pytest.mark.asyncio
async def test_max_edge_mapping(db_session):
    from polyflip.api.settings import update_setting, SettingValue
    from polyflip.db.models import RuntimeSettings
    from sqlalchemy import select

    # Обновляем настройку с ключом "MAX_EDGE"
    await update_setting("MAX_EDGE", SettingValue(value="15.0"), db=db_session, request=None)

    # Проверяем, что в базе данных значение записалось именно в "MAX_BET_EDGE"
    row = (await db_session.execute(
        select(RuntimeSettings).where(RuntimeSettings.key == "MAX_BET_EDGE")
    )).scalar_one_or_none()
    assert row is not None
    assert row.value == "0.15"  # 15% переведено в долю 0.15


