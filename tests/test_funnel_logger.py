import pytest
from unittest.mock import AsyncMock, MagicMock
from polyflip.trading.funnel_logger import log_funnel

@pytest.mark.asyncio
async def test_funnel_logger_no_raise():
    """FunnelLogger не бросает исключений при ошибке записи."""
    mock_session = MagicMock()
    mock_session.commit = AsyncMock()
    mock_session.commit.side_effect = Exception("DB Connection Error")

    # Должен молча обработать исключение и записать warning в лог, не выбрасывая ошибку
    await log_funnel(
        mock_session, market_id="x" * 200, asset="BTC",
        trading_mode="ML", used_model=None,
        p_flip=0.3, edge=0.05, fresh_price=0.7,
        threshold_lower=0.35, threshold_upper=0.65,
        min_edge_used=0.05, final_action="SKIP",
    )


@pytest.mark.asyncio
async def test_funnel_logger_writes_v3_gate_fields_without_multiplier():
    mock_session = MagicMock()
    mock_session.commit = AsyncMock()

    await log_funnel(
        mock_session,
        market_id="m1",
        asset="BTC",
        trading_mode="COMBINED",
        p_flip=0.2,
        edge=0.1,
        fresh_price=0.5,
        threshold_lower=0.4,
        threshold_upper=0.6,
        min_edge_used=0.04,
        mrf_mode="SHADOW",
        mrf_policy_version=3,
        mrf_regime_evidence=-0.22,
        mrf_gate_threshold=0.15,
        mrf_edge_margin=0.01,
        mrf_gate_would_block=True,
        mrf_gate_reason="regime_veto",
        mrf_multiplier=None,
        final_action="BUY_NO",
    )

    row = mock_session.add.call_args.args[0]
    assert row.mrf_policy_version == 3
    assert row.mrf_regime_evidence == -0.22
    assert row.mrf_gate_would_block is True
    assert row.mrf_gate_reason == "regime_veto"
    assert row.mrf_multiplier is None
