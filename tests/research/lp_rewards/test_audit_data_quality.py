import importlib
from decimal import Decimal
import json
from pathlib import Path
import sys
import pandas as pd
import pytest

audit_04 = importlib.import_module("scripts.research.lp_rewards.04_audit_data_quality")
audit_daily_evaluations = audit_04.audit_daily_evaluations
audit_data_quality = audit_04.audit_data_quality
audit_l2_parquet = audit_04.audit_l2_parquet
audit_trade_parquet = audit_04.audit_trade_parquet
from polyflip.research.lp_rewards.protocol import LPProtocol, load_protocol


def test_audit_l2_parquet_valid(tmp_path):
    p_file = tmp_path / "valid.parquet"
    now_ns = 1_700_000_000_000_000_000
    df = pd.DataFrame([
        {
            "timestamp_ns": now_ns,
            "condition_id": "c1",
            "asset_id": "tok1",
            "side": "bid",
            "price": "0.45",
            "size": "100.0",
            "valid_from_ns": now_ns,
            "valid_to_ns": now_ns + 10_000_000_000,
            "order_age_sec": "0.0",
        },
        {
            "timestamp_ns": now_ns + 1_000_000_000,
            "condition_id": "c1",
            "asset_id": "tok1",
            "side": "ask",
            "price": "0.55",
            "size": "80.0",
            "valid_from_ns": now_ns,
            "valid_to_ns": now_ns + 10_000_000_000,
            "order_age_sec": "1.0",
        },
    ])
    df.to_parquet(p_file, compression="zstd", index=False)

    valid, viols, stats = audit_l2_parquet(p_file, now_ns=now_ns + 5_000_000_000)
    assert valid is True
    assert len(viols) == 0
    assert stats["rows"] == 2
    assert stats["duplicates"] == 0
    assert stats["min_price"] == 0.45
    assert stats["max_price"] == 0.55


def test_audit_l2_parquet_missing_columns(tmp_path):
    p_file = tmp_path / "missing_cols.parquet"
    df = pd.DataFrame([{"timestamp_ns": 123, "condition_id": "c1"}])
    df.to_parquet(p_file, compression="zstd", index=False)

    valid, viols, _ = audit_l2_parquet(p_file)
    assert valid is False
    assert any("Missing required columns" in v for v in viols)


def test_audit_l2_parquet_price_out_of_bounds(tmp_path):
    p_file = tmp_path / "bad_price.parquet"
    now_ns = 1_700_000_000_000_000_000
    df = pd.DataFrame([
        {
            "timestamp_ns": now_ns,
            "condition_id": "c1",
            "asset_id": "tok1",
            "side": "bid",
            "price": "1.45",  # Probability > 1.0 is invalid
            "size": "100.0",
            "valid_from_ns": now_ns,
            "valid_to_ns": now_ns,
            "order_age_sec": "0.0",
        }
    ])
    df.to_parquet(p_file, compression="zstd", index=False)

    valid, viols, _ = audit_l2_parquet(p_file, now_ns=now_ns)
    assert valid is False
    assert any("Prices out of valid probability range" in v for v in viols)


def test_audit_l2_parquet_size_non_positive(tmp_path):
    p_file = tmp_path / "zero_size.parquet"
    now_ns = 1_700_000_000_000_000_000
    df = pd.DataFrame([
        {
            "timestamp_ns": now_ns,
            "condition_id": "c1",
            "asset_id": "tok1",
            "side": "bid",
            "price": "0.45",
            "size": "0.0",  # Zero size is invalid
            "valid_from_ns": now_ns,
            "valid_to_ns": now_ns,
            "order_age_sec": "0.0",
        }
    ])
    df.to_parquet(p_file, compression="zstd", index=False)

    valid, viols, _ = audit_l2_parquet(p_file, now_ns=now_ns)
    assert valid is False
    assert any("Non-positive order sizes" in v for v in viols)


def test_audit_l2_parquet_future_timestamps(tmp_path):
    p_file = tmp_path / "future_ts.parquet"
    now_ns = 1_700_000_000_000_000_000
    df = pd.DataFrame([
        {
            "timestamp_ns": now_ns + 3600 * 1_000_000_000,  # 1 hour in future
            "condition_id": "c1",
            "asset_id": "tok1",
            "side": "bid",
            "price": "0.45",
            "size": "10.0",
            "valid_from_ns": now_ns,
            "valid_to_ns": now_ns,
            "order_age_sec": "0.0",
        }
    ])
    df.to_parquet(p_file, compression="zstd", index=False)

    valid, viols, _ = audit_l2_parquet(p_file, now_ns=now_ns)
    assert valid is False
    assert any("Future timestamps detected" in v for v in viols)


def test_audit_l2_parquet_non_monotonic_timestamps(tmp_path):
    p_file = tmp_path / "non_monotonic.parquet"
    now_ns = 1_700_000_000_000_000_000
    df = pd.DataFrame([
        {
            "timestamp_ns": now_ns + 10_000_000_000,
            "condition_id": "c1",
            "asset_id": "tok1",
            "side": "bid",
            "price": "0.45",
            "size": "10.0",
            "valid_from_ns": now_ns,
            "valid_to_ns": now_ns,
            "order_age_sec": "0.0",
        },
        {
            "timestamp_ns": now_ns,  # Timestamp went backwards
            "condition_id": "c1",
            "asset_id": "tok1",
            "side": "bid",
            "price": "0.46",
            "size": "10.0",
            "valid_from_ns": now_ns,
            "valid_to_ns": now_ns,
            "order_age_sec": "0.0",
        },
    ])
    df.to_parquet(p_file, compression="zstd", index=False)

    valid, viols, _ = audit_l2_parquet(p_file, now_ns=now_ns + 20_000_000_000)
    assert valid is False
    assert any("Non-monotonic timestamps" in v for v in viols)


def test_audit_l2_parquet_unaggregated_duplicates(tmp_path):
    p_file = tmp_path / "duplicates.parquet"
    now_ns = 1_700_000_000_000_000_000
    df = pd.DataFrame([
        {
            "timestamp_ns": now_ns,
            "condition_id": "c1",
            "asset_id": "tok1",
            "side": "bid",
            "price": "0.45",
            "size": "10.0",
            "valid_from_ns": now_ns,
            "valid_to_ns": now_ns,
            "order_age_sec": "0.0",
        },
        {
            "timestamp_ns": now_ns,
            "condition_id": "c1",
            "asset_id": "tok1",
            "side": "bid",
            "price": "0.45",
            "size": "20.0",
            "valid_from_ns": now_ns,
            "valid_to_ns": now_ns,
            "order_age_sec": "0.0",
        },
    ])
    df.to_parquet(p_file, compression="zstd", index=False)

    valid, viols, stats = audit_l2_parquet(p_file, now_ns=now_ns)
    assert valid is False
    assert stats["duplicates"] == 2
    assert any("Unaggregated duplicate" in v for v in viols)


def test_audit_daily_evaluations_hash_mismatch_and_quote_hours(tmp_path):
    eval_dir = tmp_path / "daily_evaluations"
    eval_dir.mkdir()

    # Case 1: Hash mismatch
    f1 = eval_dir / "2026-09-14.json"
    with open(f1, "w", encoding="utf-8") as f:
        json.dump({
            "date": "2026-09-14",
            "protocol_hash": "wrong_hash",
            "quote_hours": "12.5",
            "net_pnl": "1.2",
        }, f)

    valid, viols, _ = audit_daily_evaluations(eval_dir, expected_protocol_hash="correct_hash")
    assert valid is False
    assert any("Protocol hash mismatch" in v for v in viols)

    # Case 2: quote_hours exceeds 24h
    with open(f1, "w", encoding="utf-8") as f:
        json.dump({
            "date": "2026-09-14",
            "protocol_hash": "correct_hash",
            "quote_hours": "25.0",
            "net_pnl": "1.2",
        }, f)

    valid, viols, _ = audit_daily_evaluations(eval_dir, expected_protocol_hash="correct_hash")
    assert valid is False
    assert any("exceeds 24h" in v for v in viols)


def test_audit_data_quality_full_flow(tmp_path):
    protocol = load_protocol()
    storage = tmp_path / "lp_storage"
    storage.mkdir()

    # Empty storage
    res_empty = audit_data_quality(storage, protocol)
    assert res_empty["status"] == "PENDING_DATA_ACCUMULATION"
    assert res_empty["is_valid"] is True

    # Add 1 valid market snapshot
    m1_dir = storage / "l2_snapshots" / "cond1"
    m1_dir.mkdir(parents=True)
    now_ns = 1_700_000_000_000_000_000
    df = pd.DataFrame([
        {
            "timestamp_ns": now_ns,
            "condition_id": "cond1",
            "asset_id": "tok1",
            "side": "bid",
            "price": "0.49",
            "size": "50.0",
            "valid_from_ns": now_ns,
            "valid_to_ns": now_ns,
            "order_age_sec": "0.0",
        }
    ])
    df.to_parquet(m1_dir / "2026-09-14.parquet", compression="zstd", index=False)

    res_acc = audit_data_quality(storage, protocol, now_ns=now_ns)
    assert res_acc["status"] == "DATA_QUALITY_ACCUMULATING"
    assert res_acc["is_valid"] is True
    assert res_acc["total_l2_files"] == 1
    assert res_acc["total_l2_rows"] == 1

    # Corrupt the parquet file by writing bad data
    bad_df = pd.DataFrame([{"invalid_col": 123}])
    bad_df.to_parquet(m1_dir / "2026-09-15.parquet", compression="zstd", index=False)

    res_bad = audit_data_quality(storage, protocol, now_ns=now_ns)
    assert res_bad["status"] == "DATA_INTEGRITY_VIOLATION"
    assert res_bad["is_valid"] is False
    assert len(res_bad["violations"]) > 0


def test_audit_l2_parquet_calendar_coverage_by_timestamp_ns(tmp_path):
    """Verify that multiple files with different filenames within the same UTC day count as 1 day."""
    protocol = load_protocol()
    storage = tmp_path / "lp_storage"
    m1_dir = storage / "l2_snapshots" / "cond_test_cov"
    m1_dir.mkdir(parents=True)

    # Base timestamp: 2026-09-14 10:00:00 UTC = 1789376400 seconds
    base_ts_ns = 1_789_376_400_000_000_000

    # Write 5 files with completely different names, all containing data for the same UTC day
    for i in range(5):
        p_file = m1_dir / f"arbitrary_batch_{i}_{i*100}.parquet"
        file_ts = base_ts_ns + i * 3600 * 1_000_000_000  # 1 hour later, still same UTC day
        df = pd.DataFrame([
            {
                "timestamp_ns": file_ts,
                "condition_id": "cond_test_cov",
                "asset_id": "tok1",
                "side": "bid",
                "price": "0.45",
                "size": "100.0",
                "valid_from_ns": file_ts,
                "valid_to_ns": file_ts + 10_000_000_000,
                "order_age_sec": "0.0",
            }
        ])
        df.to_parquet(p_file, compression="zstd", index=False)

    res = audit_data_quality(storage, protocol, now_ns=base_ts_ns + 86400 * 1_000_000_000)
    assert res["is_valid"] is True
    assert res["total_l2_files"] == 5
    # Crucial: 5 files for the same UTC day must yield exactly 1 calendar day of coverage
    expected_ratio = 1.0 / float(protocol.gates.gate_a.min_calendar_days)
    assert abs(res["min_coverage_ratio"] - expected_ratio) < 1e-6
    assert abs(res["avg_coverage_ratio"] - expected_ratio) < 1e-6


def test_audit_daily_evaluations_simulated_and_actual_quote_hours(tmp_path):
    """Verify audit_daily_evaluations handles both simulated and actual quote-hours."""
    eval_dir = tmp_path / "daily_evaluations"
    eval_dir.mkdir()

    # Case 1: Standard record with simulated_quote_hours and actual_quote_hours
    f1 = eval_dir / "2026-09-14.json"
    with open(f1, "w", encoding="utf-8") as f:
        json.dump({
            "date": "2026-09-14",
            "protocol_hash": "correct_hash",
            "simulated_quote_hours": "14.5",
            "actual_quote_hours": "0.0",
            "quote_hours": "14.5",
            "net_pnl": "2.5",
        }, f)

    valid, viols, stats = audit_daily_evaluations(eval_dir, expected_protocol_hash="correct_hash")
    assert valid is True
    assert len(viols) == 0
    assert stats["total_simulated_quote_hours"] == Decimal("14.5")
    assert stats["total_actual_quote_hours"] == Decimal("0.0")
    assert stats["total_quote_hours"] == Decimal("14.5")

    # Case 2: Legacy record with only quote_hours
    f2 = eval_dir / "2026-09-15.json"
    with open(f2, "w", encoding="utf-8") as f:
        json.dump({
            "date": "2026-09-15",
            "protocol_hash": "correct_hash",
            "quote_hours": "10.0",
            "net_pnl": "1.0",
        }, f)

    valid, viols, stats = audit_daily_evaluations(eval_dir, expected_protocol_hash="correct_hash")
    assert valid is True
    assert stats["total_simulated_quote_hours"] == Decimal("24.5")
    assert stats["total_quote_hours"] == Decimal("24.5")

    # Case 3: Negative actual quote-hours
    f3 = eval_dir / "2026-09-16.json"
    with open(f3, "w", encoding="utf-8") as f:
        json.dump({
            "date": "2026-09-16",
            "protocol_hash": "correct_hash",
            "simulated_quote_hours": "10.0",
            "actual_quote_hours": "-1.0",
            "net_pnl": "1.0",
        }, f)

    valid, viols, _ = audit_daily_evaluations(eval_dir, expected_protocol_hash="correct_hash")
    assert valid is False
    assert any("Negative actual_quote_hours" in v for v in viols)
