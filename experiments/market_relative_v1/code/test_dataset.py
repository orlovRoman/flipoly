"""Dataset self-tests (T08). Run: pytest test_dataset.py. Env knobs:
MKTREL_DATA/day MKTREL_BUILD dirs; default to ./data and ./build next to builder.
"""
import hashlib
import os

import numpy as np
import pandas as pd
import pytest

import build_dataset as B

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("MKTREL_ROOT", os.path.dirname(BASE))
DATA = os.environ.get("MKTREL_DATA", os.path.join(ROOT, "data"))
BUILD = os.environ.get("MKTREL_BUILD", os.path.join(ROOT, "build"))
DS = os.path.join(BUILD, "dataset.csv")


@pytest.fixture(scope="module")
def df():
    if not os.path.exists(DS):
        pytest.skip("frozen dataset is not present; run the read-only export/build step")
    return pd.read_csv(DS, dtype={"market_id": str})


@pytest.fixture(scope="module")
def snaps():
    if not os.path.exists(os.path.join(DATA, "snaps.csv")):
        pytest.skip("frozen snapshot input is not present; run the read-only export/build step")
    from build_dataset import epoch_of
    s = pd.read_csv(os.path.join(DATA, "snaps.csv"), dtype={"market_id": str},
                    usecols=["market_id", "recorded_at", "poly_up_mid", "mid_price"])
    s["rec"] = epoch_of(pd.to_datetime(s["recorded_at"], utc=True))
    return s


def test_key_unique(df):
    assert df["market_id"].is_unique


def test_columns(df):
    expect = (["market_id", "asset", "end3", "dec_at", "tte_s", "fold", "label"] +
              B.FEATURES + B.FLAGS + ["yes_ask", "no_ask", "synthetic_no", "arb", "outcome"])
    assert list(df.columns) == expect


def test_tte_range(df):
    assert bool(((df["tte_s"] >= 270.0) & (df["tte_s"] <= 330.0)).all())


def test_price_ranges(df):
    assert bool(((df["p_market_yes"] > 0) & (df["p_market_yes"] < 1)).all())
    assert bool(((df["yes_ask"] > 0) & (df["yes_ask"] < 1)).all())
    assert bool(((df["no_ask"] > 0) & (df["no_ask"] < 1)).all())


def test_fold_values(df):
    assert set(df["fold"].unique()) <= {"trainpool", "F1", "F2", "F3", "F4", "F5", "F6"}
    assert "gap" not in set(df["fold"].unique())


def test_label_binary(df):
    assert set(df["label"].unique()) <= {0.0, 1.0}
    assert ((df["label"] == 1.0) == (df["outcome"] == "YES")).all()


def test_flags_binary(df):
    for f in B.FLAGS:
        assert set(df[f].dropna().unique()) <= {0.0, 1.0}, f


def test_flags_match_nan(df):
    for f in B.FEATURES:
        nan = df[f].isna().to_numpy()
        flag = (df[f + "_m"].to_numpy() == 1.0)
        assert bool((nan == flag).all()), f


def test_label_swap_sensitivity():
    """Swapped YES/NO inputs must flip labels (mapping actually drives labels)."""
    import shutil
    if not all(
        os.path.exists(os.path.join(DATA, name))
        for name in ("markets.csv", "snaps.csv", "candles.csv")
    ):
        pytest.skip("frozen dataset inputs are not present; run the read-only export/build step")
    import tempfile
    tmp = tempfile.mkdtemp(prefix="mktrel_swap_")
    for n in ("markets.csv", "snaps.csv", "candles.csv"):
        shutil.copy(os.path.join(DATA, n), tmp)
    m = pd.read_csv(os.path.join(tmp, "markets.csv"), dtype=str)
    m["outcome"] = m["outcome"].map({"YES": "NO", "NO": "YES"}).fillna(m["outcome"])
    m.to_csv(os.path.join(tmp, "markets.csv"), index=False)
    out = os.path.join(tmp, "out")
    B.build(tmp, out)
    base = pd.read_csv(DS, dtype={"market_id": str}, usecols=["market_id", "label"])
    swap = pd.read_csv(os.path.join(out, "dataset.csv"), dtype={"market_id": str},
                       usecols=["market_id", "label"])
    j = base.merge(swap, on="market_id", suffixes=("_b", "_s"))
    assert len(j) == len(base)
    assert bool(((j["label_b"] == 1.0) == (j["label_s"] == 0.0)).all())


def test_lookback_no_future(df, snaps):
    """dprob sources must come from snapshots at most dec-N back (no future fill)."""
    from build_dataset import epoch_of
    grp = {mid: g.sort_values("rec") for mid, g in snaps.groupby("market_id", sort=True)}
    rng = np.random.RandomState(7)
    idx = rng.choice(len(df), size=min(200, len(df)), replace=False)
    for i in idx:
        r = df.iloc[i]
        if r["dprob_90s_m"] == 1.0:
            continue
        g = grp[r["market_id"]]
        dec = float(epoch_of(pd.Timestamp(r["dec_at"], tz="utc")))
        pool = g[(g["rec"] <= dec - 90) & (g["rec"] >= dec - 150)]
        assert len(pool) > 0
        src = pool.iloc[-1]
        p_at = src["poly_up_mid"] if src["poly_up_mid"] == src["poly_up_mid"] and 0 < src["poly_up_mid"] < 1 else src["mid_price"]
        assert abs((r["p_market_yes"] - p_at) - r["dprob_90s"]) < 1e-6


def test_manifest_reconciles():
    import json
    if not all(
        os.path.exists(os.path.join(BUILD, name))
        for name in ("manifest.json", "dataset.sha256", "dataset.csv")
    ):
        pytest.skip("frozen dataset build artifacts are not present; run the read-only export/build step")
    man = json.load(open(os.path.join(BUILD, "manifest.json")))
    df = pd.read_csv(DS, dtype={"market_id": str})
    assert man["n_rows"] == len(df)
    assert man["dataset_sha256"] == open(os.path.join(BUILD, "dataset.sha256")).read().strip()
    h = hashlib.sha256(open(DS, "rb").read()).hexdigest()
    assert h == man["dataset_sha256"]
