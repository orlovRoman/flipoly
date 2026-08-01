
@pytest.mark.asyncio
async def test_resolve_no_fill_batch_safe_and_unsafe(db_session):
    from polyflip.api.main import app
    from polyflip.db.connection import get_db_session
    from polyflip.api.auth import verify_api_key

    # Safe request
    trade1 = make_trade()
    db_session.add(trade1)
    await db_session.flush()

    req1 = make_req(trade1.id)
    db_session.add(req1)
    await db_session.flush()

    res1 = ExposureReservation(
        id=uuid.uuid4(),
        request_id=req1.id,
        trade_history_id=trade1.id,
        market_id="test_market",
        amount_usdc=Decimal("10.0"),
        created_at=datetime.now(timezone.utc), expires_at=datetime.now(timezone.utc),
    )
    db_session.add(res1)
    await db_session.flush()

    # Unsafe request (has execution fills)
    trade2 = make_trade()
    db_session.add(trade2)
    await db_session.flush()

    req2 = make_req(trade2.id)
    db_session.add(req2)
    await db_session.flush()

    attempt = make_attempt(req2.id)
    db_session.add(attempt)
    await db_session.flush()

    fill = ExecutionFill(
        id=uuid.uuid4(),
        attempt_id=attempt.id,
        provider_trade_id="x",
        shares=Decimal("1.0"),
        price=Decimal("0.5"),
        timestamp=datetime.now(timezone.utc),
    )
    db_session.add(fill)
    await db_session.commit()

    app.dependency_overrides[get_db_session] = lambda: db_session
    app.dependency_overrides[verify_api_key] = lambda: True

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/execution/requests/resolve-no-fill-batch",
            json={
                "request_ids": [str(req1.id), str(req2.id)],
                "operator": "batch_op",
                "note": "batch_test"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert str(req1.id) in data["resolved"]
        assert any(s["request_id"] == str(req2.id) for s in data["skipped"])

    await db_session.refresh(req1)
    assert req1.state == "MANUAL_REVIEW_FAILED"

    await db_session.refresh(req2)
    assert req2.state == "MANUAL_REVIEW_REQUIRED"

    from sqlalchemy import select, func
    event_count = await db_session.scalar(
        select(func.count(ExecutionEvent.id))
        .where(ExecutionEvent.request_id == req1.id)
        .where(ExecutionEvent.event_type == "MANUAL_REVIEW_BATCH_NO_FILL")
    )
    assert event_count == 1
