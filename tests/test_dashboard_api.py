import pytest
from types import SimpleNamespace
from datetime import datetime, timezone
from polyflip.db.models import TradeHistory
from polyflip.api.dashboard import get_trade_logs, _mrf_audit_payload


def test_mrf_audit_payload_prefers_json_and_preserves_policy():
    funnel = SimpleNamespace(
        mrf_audit_json='{"global_phase":"MIXED","policy":{"allow":true,"multiplier":0.5}}',
        mrf_mode="ACTIVE",
        mrf_evaluated=True,
        mrf_phase="SIDEWAYS",
        mrf_asset_phase="WEAK_UP",
        mrf_strength=0.1,
        mrf_confidence=0.9,
        mrf_multiplier=0.5,
        mrf_applied=True,
        mrf_failure_reason=None,
        mrf_final_action="BUY_YES",
    )

    payload = _mrf_audit_payload(funnel)

    assert payload["global_phase"] == "MIXED"
    assert payload["asset_phase"] == "WEAK_UP"
    assert payload["policy"] == {"allow": True, "multiplier": 0.5}


def test_mrf_audit_payload_falls_back_to_scalar_telemetry():
    funnel = SimpleNamespace(
        mrf_audit_json=None,
        mrf_mode="ACTIVE",
        mrf_evaluated=False,
        mrf_phase="UNKNOWN",
        mrf_asset_phase="UNKNOWN",
        mrf_strength=None,
        mrf_confidence=None,
        mrf_multiplier=None,
        mrf_applied=False,
        mrf_failure_reason="not_ready",
        mrf_final_action="SKIP",
    )

    payload = _mrf_audit_payload(funnel)

    assert payload["mode"] == "ACTIVE"
    assert payload["global_phase"] == "UNKNOWN"
    assert payload["failure_reason"] == "not_ready"
    assert payload["policy"]["allow"] is False

@pytest.mark.asyncio
async def test_trade_logs_pagination(db_session):
    now = datetime.now(timezone.utc)
    for i in range(60):
        t = TradeHistory(market_id=f'm_{i}', asset='BTC', outcome_bought='YES', amount_usdc=10.0, remaining_shares=20.0, executed_price=0.5, predicted_flip_prob=0.8, active_features='test', status='SUCCESS', created_at=now)
        db_session.add(t)
    await db_session.commit()
    res1 = await get_trade_logs(db_session, page=1, page_size=25)
    assert len(res1['items']) == 25
    assert res1['total'] == 60
    assert res1['pages'] == 3
    assert res1['page'] == 1
    assert res1['page_size'] == 25
    assert 'edge' in res1['items'][0]
    res3 = await get_trade_logs(db_session, page=3, page_size=25)
    assert len(res3['items']) == 10
    assert res3['total'] == 60
    assert res3['pages'] == 3
    res99 = await get_trade_logs(db_session, page=99, page_size=25)
    assert len(res99['items']) == 0
    assert res99['total'] == 60

@pytest.mark.asyncio
async def test_get_daily_pnl(db_session):
    from polyflip.api.dashboard import get_daily_pnl
    now = datetime.now(timezone.utc)
    t = TradeHistory(market_id='m_pnl_1', asset='BTC', outcome_bought='YES', amount_usdc=10.0, remaining_shares=20.0, executed_price=0.6, predicted_flip_prob=0.8, active_features='other', status='SUCCESS', position_status='CLOSED', pnl=1.5, created_at=now)
    db_session.add(t)
    await db_session.commit()
    res = await get_daily_pnl(db=db_session)
    assert res['status'] == 'success'
    assert len(res['data']) == 1
    assert res['data'][0]['strategy'] == 'Фаворит'