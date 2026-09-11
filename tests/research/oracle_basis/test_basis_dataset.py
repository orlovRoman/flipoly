"""Loader dedupe/conflicts, horizon checkpoints, dataset rows and day splits."""

import pytest

from polyflip.collector.rtds_collector import RTDSError
from polyflip.research.oracle_basis.basis_dataset import (
    SplitMarket,
    assemble_row,
    book_regime,
    gap_bucket,
    horizon_checkpoints,
    split_markets,
)
from polyflip.research.oracle_basis.proxy_twap import E18
from polyflip.research.oracle_basis.rtds_loader import (
    checkpoint_table,
    load_observations,
    select_checkpoint_observation,
)


def obs(source, symbol, observed, received, price):
    return {
        "source": source,
        "asset": symbol.replace("USDT", "").replace("USD", ""),
        "symbol": symbol,
        "price": price,
        "observed_at": observed,
        "received_at": received,
    }


def test_loader_dedupes_conflicts_and_rejects():
    rows = [
        obs("BINANCE", "BTCUSDT", 1_000, 1_100, "100"),
        obs(
            "BINANCE", "BTCUSDT", 1_000, 1_050, "100"
        ),  # duplicate, earlier receipt wins
        obs("BINANCE", "BTCUSDT", 2_000, 2_100, "101"),
        obs(
            "BINANCE", "BTCUSDT", 2_000, 2_050, "102"
        ),  # conflict, earliest receipt wins
        {
            "source": "BINANCE",
            "asset": "BTC",
            "price": 1.5,
            "observed_at": 3_000,
            "received_at": 3_100,
        },
    ]
    kept, summary = load_observations(rows)
    assert summary["rows_kept"] == 2
    assert summary["duplicates"] == 1
    assert summary["conflicts"] == 1
    assert summary["rejected_count"] == 1
    by_observed = {o.observed_at_ms: o for o in kept}
    assert by_observed[1_000].received_at_ms == 1_050
    assert by_observed[2_000].price_e18 == 102 * E18


def test_checkpoint_selection_marks_stale_not_estimates():
    kept, _ = load_observations([obs("BINANCE", "BTCUSDT", 1_000, 1_100, "100")])
    found, status, age = select_checkpoint_observation(kept, 1_200, 500)
    assert (status, age) == ("OK", 100) and found is not None
    found, status, age = select_checkpoint_observation(kept, 5_000, 500)
    assert (found, status) == (None, "STALE") and age == 3_900
    assert select_checkpoint_observation(kept, 500, 500)[1] == "NO_DATA"
    table = checkpoint_table(kept, [500, 1_200], 500)
    assert [r["status"] for r in table] == ["NO_DATA", "OK"]


def test_horizons_and_row_without_official_reference():
    assert horizon_checkpoints(900_000) == {
        180: 720_000,
        60: 840_000,
        30: 870_000,
        15: 885_000,
        5: 895_000,
    }
    row = assemble_row(
        market_id="m1",
        asset="btc",
        horizon_sec=60,
        checkpoint_ms=840_000,
        spot_status="OK",
        spot_age_ms=100,
        proxy_open_e18=100 * E18,
        proxy_close_e18=101 * E18,
        official_twap_e18=None,
        official_status="MISSING",
        outcome_up=True,
    )
    assert row["reconstruction_available"] is False
    assert row["reconstruction_gap_reason"] == "MISSING"
    assert row["proxy_is_oracle"] is False
    assert row["label_source"] == "OFFICIAL_OUTCOME"
    assert row["flip_error"] == {"proxy_direction": "UP", "flip_error": False}


def _ms(day: str, hour: int, minute: int) -> int:
    from datetime import datetime, timezone

    year, month, dom = (int(p) for p in day.split("-"))
    return int(
        datetime(year, month, dom, hour, minute, tzinfo=timezone.utc).timestamp() * 1000
    )


def test_day_splits_keep_days_together_and_embargo():
    d1, d2, d3 = "2026-09-01", "2026-09-02", "2026-09-03"
    midnight_d2 = _ms(d2, 0, 0)
    midnight_d3 = _ms(d3, 0, 0)
    markets = [
        SplitMarket("t1", d1, _ms(d1, 10, 15), _ms(d1, 10, 0)),  # keep
        SplitMarket(
            "t3", d1, midnight_d2 - 900_000, midnight_d2 - 990_000
        ),  # ends exactly at embargo edge: keep
        SplitMarket(
            "t4", d1, midnight_d2 - 300_000, midnight_d2 - 1_200_000
        ),  # inside window: embargo drop
        SplitMarket("v1", d2, _ms(d2, 10, 15), _ms(d2, 10, 0)),  # keep
        SplitMarket(
            "v2", d2, midnight_d3 - 300_000, midnight_d3 - 1_200_000
        ),  # validation/test embargo drop
        SplitMarket(
            "x2", d3, midnight_d3 + 600_000, midnight_d3 - 300_000
        ),  # crosses into test: drop
        SplitMarket("s1", d3, _ms(d3, 10, 15), _ms(d3, 10, 0)),  # keep
    ]
    split = split_markets(markets, embargo_ms=900_000)
    assert split["days"] == {d1: "train", d2: "validation", d3: "test"}
    assert split["splits"] == {
        "train": ["t1", "t3"],
        "validation": ["v1"],
        "test": ["s1"],
    }
    # Both boundaries enforce the window AND the crossing exclusion.
    assert split["dropped"] == [
        {"market_id": "t4", "reason": "EMBARGO_WINDOW_BEFORE_VALIDATION"},
        {"market_id": "x2", "reason": "CROSSES_SPLIT_BOUNDARY"},
        {"market_id": "v2", "reason": "EMBARGO_WINDOW_BEFORE_TEST"},
    ]
    assert split["excluded"] == []


def test_day_splits_drop_crossing_markets_and_unknown_starts():
    d1, d2, d3 = "2026-09-01", "2026-09-02", "2026-09-03"
    midnight_d2 = _ms(d2, 0, 0)
    markets = [
        SplitMarket("t1", d1, _ms(d1, 10, 15), _ms(d1, 10, 0)),
        # Ends after midnight UTC: utc_day is d2, interval crosses the boundary.
        SplitMarket("x1", d2, midnight_d2 + 600_000, midnight_d2 - 300_000),
        SplitMarket("v1", d2, _ms(d2, 10, 15), _ms(d2, 10, 0)),
        SplitMarket("s1", d3, _ms(d3, 10, 15), _ms(d3, 10, 0)),
        {"market_id": "u1", "utc_day": d1, "end_ms": _ms(d1, 12, 0), "start_ms": None},
    ]
    split = split_markets(markets, embargo_ms=900_000)
    assert split["splits"] == {"train": ["t1"], "validation": ["v1"], "test": ["s1"]}
    assert split["dropped"] == [{"market_id": "x1", "reason": "CROSSES_SPLIT_BOUNDARY"}]
    assert split["excluded"] == [{"market_id": "u1", "reason": "NO_START_MS"}]
    with pytest.raises(RTDSError):
        split_markets([SplitMarket("t1", d1, 10, 10), SplitMarket("t1", d2, 20, 15)])


def test_buckets_and_regimes():
    assert gap_bucket(None) == "NO_OFFICIAL_REFERENCE"
    assert (
        gap_bucket(2) == "<3bps"
        and gap_bucket(10) == "3-10bps"
        and gap_bucket(11) == ">10bps"
    )
    assert book_regime(0.41) == "CONTESTED"
    assert book_regime(0.40) == "FAVORITE"
    assert book_regime(0.9) == "FAVORITE"
    with pytest.raises(RTDSError):
        book_regime(1.5)
