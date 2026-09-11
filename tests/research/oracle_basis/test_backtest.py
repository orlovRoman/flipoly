"""Executable economics: documented fee, both-side settlement, sizing, fills."""

from decimal import Decimal

import pytest

from polyflip.research.oracle_basis.backtest import (
    EconomicsError,
    block_bootstrap_mean_ci,
    ledger_row,
    settle_pnl_usdc,
    size_for_budget,
    take_liquidity,
    taker_fee,
)


def test_documented_taker_fee_formula():
    assert taker_fee(10, Decimal("0.5")) == Decimal("0.175000")
    assert taker_fee(0, Decimal("0.9")) == Decimal("0")
    assert taker_fee(10, Decimal("0.5"), scheme="ZERO") == Decimal("0")
    with pytest.raises(TypeError):
        taker_fee(10, 0.5)
    with pytest.raises(EconomicsError):
        taker_fee(10, Decimal("0.5"), scheme="FIXED_020")


def test_settlement_pnl_both_sides():
    fee = taker_fee(10, Decimal("0.5"))
    up_win = settle_pnl_usdc(
        side="UP", contracts=10, ask_price=Decimal("0.5"), outcome_up=True, fee=fee
    )
    assert up_win == Decimal("10") * (Decimal(1) - Decimal("0.5")) - fee
    down_win = settle_pnl_usdc(
        side="DOWN", contracts=10, ask_price=Decimal("0.4"), outcome_up=False, fee=fee
    )
    assert down_win == Decimal("10") * (Decimal(1) - Decimal("0.4")) - fee
    down_loss = settle_pnl_usdc(
        side="DOWN", contracts=10, ask_price=Decimal("0.4"), outcome_up=True, fee=fee
    )
    assert down_loss == Decimal("10") * (Decimal(0) - Decimal("0.4")) - fee


def test_sizing_includes_fee_step_and_depth():
    # Unit cash at a=0.5 is 0.5175, so $10 affords 19 contracts before lots/depth.
    assert (
        size_for_budget(
            budget_usdc=10, ask_price=Decimal("0.5"), min_size=1, size_step=1
        )
        == 19
    )
    assert (
        size_for_budget(
            budget_usdc=10, ask_price=Decimal("0.5"), min_size=1, size_step=5
        )
        == 15
    )
    assert (
        size_for_budget(
            budget_usdc=10,
            ask_price=Decimal("0.5"),
            min_size=1,
            size_step=1,
            depth_available=7,
        )
        == 7
    )
    assert (
        size_for_budget(
            budget_usdc=Decimal("0.1"),
            ask_price=Decimal("0.5"),
            min_size=1,
            size_step=1,
        )
        == 0
    )


def test_take_liquidity_reports_shortfall_and_vwap():
    fill = take_liquidity([(Decimal("0.5"), 6), (Decimal("0.6"), 10)], 10)
    assert fill["filled"] == 10 and fill["shortfall"] == 0
    assert fill["vwap"] == (Decimal("0.5") * 6 + Decimal("0.6") * 4) / 10
    short = take_liquidity([(Decimal("0.5"), 3)], 10)
    assert (short["filled"], short["shortfall"]) == (3, 7)


def test_ledger_row_and_day_block_bootstrap():
    row = ledger_row(
        market_id="m1",
        asset="BTC",
        horizon_sec=300,
        side="UP",
        contracts=10,
        fill_price=Decimal("0.5"),
        outcome_up=True,
        fee=Decimal("0.175"),
        policy="net_ev_threshold",
        post_latency_fill=True,
        depth_limited=False,
    )
    assert Decimal(row["pnl_usdc"]) == Decimal("10") * Decimal("0.5") - Decimal("0.175")
    assert row["post_latency_fill"] is True
    ci = block_bootstrap_mean_ci([1.0, 2.0, 3.0], n_boot=500, seed=0)
    assert ci["mean"] == pytest.approx(2.0)
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]
    assert ci["blocks"] == 3.0


def test_ledger_v2_scaled_integers():
    from polyflip.research.oracle_basis.backtest import LEDGER_COLUMNS, ledger_row_v2

    row = ledger_row_v2(
        market_id="m1",
        asset="btc",
        horizon_sec=300,
        side="UP",
        decision_ms=1_000,
        assumed_fill_ms=1_900,
        p_up=0.6,
        model_version="B0",
        protocol_version="0.2",
        ask_price=Decimal("0.5"),
        quantity=10,
        fee=Decimal("0.175"),
        fee_scheme="DOCUMENTED_CRYPTO_TAKER",
        outcome_up=True,
        price_sources="book@decision+arrival@+900ms",
        spot_status="OK",
        spot_age_ms=800,
        gap_bucket="<3bps",
        policy="net_ev_threshold",
        post_latency_fill=True,
        depth_limited=False,
    )
    assert list(row) == list(LEDGER_COLUMNS)
    assert row["ask_price_e12"] == 500_000_000_000
    assert row["p_up_e6"] == 600_000
    assert row["fee_micros"] == 175_000
    assert row["pnl_micros"] == 4_825_000  # 10 * 0.5 - 0.175
    assert row["spot_age_ms"] == 800
    with pytest.raises(EconomicsError):
        ledger_row_v2(
            market_id="m2",
            asset="BTC",
            horizon_sec=300,
            side="UP",
            decision_ms=1_000,
            assumed_fill_ms=999,
            p_up=0.6,
            model_version="B0",
            protocol_version="0.2",
            ask_price=Decimal("0.5"),
            quantity=10,
            fee=Decimal("0.175"),
            fee_scheme="DOCUMENTED_CRYPTO_TAKER",
            outcome_up=True,
        )


def test_ledger_parquet_roundtrip(tmp_path):
    import pyarrow.parquet as pq

    from polyflip.research.oracle_basis.backtest import (
        ledger_row_v2,
        write_ledger_parquet,
    )

    rows = [
        ledger_row_v2(
            market_id="m1",
            asset="BTC",
            horizon_sec=60,
            side="DOWN",
            decision_ms=2_000,
            assumed_fill_ms=2_500,
            p_up=0.3,
            model_version="B1",
            protocol_version="0.2",
            ask_price=Decimal("0.4"),
            quantity=5,
            fee=Decimal("0.084"),
            fee_scheme="DOCUMENTED_CRYPTO_TAKER",
            outcome_up=False,
            price_sources="book@decision",
            spot_status="STALE",
            spot_age_ms=None,
            gap_bucket="NO_OFFICIAL_REFERENCE",
            policy="t0",
            post_latency_fill=False,
            depth_limited=True,
        )
    ]
    path = str(tmp_path / "ledger.parquet")
    sha = write_ledger_parquet(rows, path)
    assert len(sha) == 64
    table = pq.read_table(path)
    assert table.schema.names[0] == "market_id"
    assert table.num_rows == 1
    assert table.column("pnl_micros")[0].as_py() == 5 * 600_000 - 84_000
    assert table.column("spot_age_ms")[0].as_py() is None
