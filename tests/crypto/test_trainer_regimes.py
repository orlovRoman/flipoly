import numpy as np
import pandas as pd
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

def _make_filtered_df(n: int = 600) -> pd.DataFrame:
    """Синтетический df_filtered с vol_ratio для тестирования разбиения."""
    np.random.seed(0)
    return pd.DataFrame({
        "vol_ratio": np.random.exponential(scale=0.5, size=n),
        "ret_1": np.random.normal(0, 0.002, n),
    })


def test_tertile_sizes_balanced():
    """Tertile-разбиение должно давать примерно равные части."""
    df = _make_filtered_df(600)
    p33 = df["vol_ratio"].quantile(0.33)
    p67 = df["vol_ratio"].quantile(0.67)

    low  = df[df["vol_ratio"] <= p33]
    mid  = df[(df["vol_ratio"] > p33) & (df["vol_ratio"] <= p67)]
    high = df[df["vol_ratio"] > p67]

    # Каждая часть ≈ 33% ± 5%
    total = len(df)
    for part in [low, mid, high]:
        ratio = len(part) / total
        assert 0.28 <= ratio <= 0.38, f"Неожиданный размер режима: {ratio:.2%}"


def test_tertile_no_overlap():
    """Строки не должны попадать в два режима одновременно."""
    df = _make_filtered_df(300)
    p33 = df["vol_ratio"].quantile(0.33)
    p67 = df["vol_ratio"].quantile(0.67)

    idx_low  = set(df[df["vol_ratio"] <= p33].index)
    idx_mid  = set(df[(df["vol_ratio"] > p33) & (df["vol_ratio"] <= p67)].index)
    idx_high = set(df[df["vol_ratio"] > p67].index)

    assert idx_low & idx_mid == set()
    assert idx_mid & idx_high == set()
    assert idx_low & idx_high == set()


def test_tertile_covers_all_rows():
    """Объединение трёх tertile должно покрывать все строки."""
    df = _make_filtered_df(300)
    p33 = df["vol_ratio"].quantile(0.33)
    p67 = df["vol_ratio"].quantile(0.67)

    low  = df[df["vol_ratio"] <= p33]
    mid  = df[(df["vol_ratio"] > p33) & (df["vol_ratio"] <= p67)]
    high = df[df["vol_ratio"] > p67]

    assert len(low) + len(mid) + len(high) == len(df)


def test_tertile_min_regime_size():
    """При n=450 каждый режим должен превышать MIN_ROWS=150."""
    df = _make_filtered_df(450)
    p33 = df["vol_ratio"].quantile(0.33)
    p67 = df["vol_ratio"].quantile(0.67)

    low  = df[df["vol_ratio"] <= p33]
    mid  = df[(df["vol_ratio"] > p33) & (df["vol_ratio"] <= p67)]
    high = df[df["vol_ratio"] > p67]

    for part in [low, mid, high]:
        assert len(part) >= 150, f"Режим слишком мал: {len(part)} строк"
