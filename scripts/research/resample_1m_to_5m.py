"""
scripts/research/resample_1m_to_5m.py

Resamples crypto_candles_1m.csv to closed 5-minute bars using OHLCV aggregation,
saving result to artifacts/research/crypto_candles_5m_resampled.csv.

Original crypto_candles_5m.csv is preserved as crypto_candles_5m_legacy.csv
(historical control) only if not already backed up.

Usage:
    python scripts/research/resample_1m_to_5m.py
"""
from __future__ import annotations

import sys
import shutil
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def resample_1m_to_5m(candles_1m: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-minute OHLCV bars into closed 5-minute bars.

    Aggregation rules:
      - open:   first 1m open in the 5m window
      - high:   max of 1m highs
      - low:    min of 1m lows
      - close:  last 1m close (last closed 1m bar)
      - volume: sum of 1m volumes

    All 5 constituent 1m bars must be present for a 5m bar to be emitted
    (is_closed=True guarantee: no partial bars).
    """
    df = candles_1m.copy()
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.sort_values("open_time").reset_index(drop=True)

    # Floor to 5-minute grid
    df["bar_5m"] = df["open_time"].dt.floor("5min")

    # Count 1m bars per 5m window — require exactly 5
    counts = df.groupby(["symbol", "bar_5m"])["open_time"].count()

    agg = (
        df.groupby(["symbol", "bar_5m"])
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index()
    )
    agg = agg.rename(columns={"bar_5m": "open_time"})

    # Filter to complete 5m bars only
    complete_mask = counts[
        counts == 5
    ].reset_index()[["symbol", "bar_5m"]].rename(columns={"bar_5m": "open_time"})
    agg = agg.merge(complete_mask, on=["symbol", "open_time"], how="inner")

    agg["interval"] = "5m"
    agg["is_closed"] = True  # all bars are guaranteed closed

    # Reorder columns to match legacy CSV
    agg = agg[["symbol", "interval", "open_time", "open", "high", "low", "close", "volume", "is_closed"]]
    agg["open_time"] = agg["open_time"].dt.strftime("%Y-%m-%d %H:%M:%S+00:00")

    return agg


def main() -> None:
    artifacts = REPO_ROOT / "artifacts" / "research"
    src_1m = artifacts / "crypto_candles_1m.csv"
    out_5m = artifacts / "crypto_candles_5m_resampled.csv"
    legacy_5m = artifacts / "crypto_candles_5m_legacy.csv"
    original_5m = artifacts / "crypto_candles_5m.csv"

    if not src_1m.exists():
        print(f"ERROR: {src_1m} not found. Run fetch_crypto_candles_1m.py first.")
        sys.exit(1)

    # Backup original 5m CSV as legacy if not already done
    if original_5m.exists() and not legacy_5m.exists():
        shutil.copy2(original_5m, legacy_5m)
        print(f"Backed up original 5m candles to {legacy_5m.name}")

    print(f"Loading {src_1m.name} ...")
    df_1m = pd.read_csv(src_1m, encoding="utf-8")
    print(f"  {len(df_1m):,} rows loaded")

    print("Resampling to 5m closed bars ...")
    df_5m = resample_1m_to_5m(df_1m)
    print(f"  {len(df_5m):,} complete 5m bars produced")

    df_5m.to_csv(out_5m, index=False, encoding="utf-8")
    size_mb = out_5m.stat().st_size / 1024 / 1024
    print(f"Saved {out_5m.name} ({size_mb:.2f} MB)")

    # Quick sanity check
    df_check = pd.read_csv(out_5m)
    assert df_check["is_closed"].all(), "All resampled bars must be closed"
    print("Sanity check passed: all bars have is_closed=True")


if __name__ == "__main__":
    main()
