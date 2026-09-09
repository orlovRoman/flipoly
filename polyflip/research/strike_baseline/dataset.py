"""Opportunity ledger + dataset assembly for strike-baseline research.

Causal pipeline (protocol P1-08..P1-12):
  1. A decision moment D_min = market_close_utc - entry_minutes.
  2. The judge picks the FIRST snapshot with recorded_at >= D_min and
     observed_delay <= allowed_max_delay_sec. If none, the opportunity status
     is snapshot_missing.
  3. Underlying at decision = last BINANCE 1m candle CLOSED at or before the
     snapshot's recorded_at (candle close_time <= recorded_at). Stale if the
     latest usable close is older than max_price_age_sec.
  4. Volatility uses only closed candles strictly before recorded_at
     (vol_est_window_min rolling). Future data never enters (P1-12).
  5. Strike proxy = first closed BINANCE 1m candle inside the market window
     (close_time <= window_end, open_time >= window_start).
  6. Clean quotes only: best_bid < best_ask, within (0,1). No best-price pick.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd

from polyflip.research.strike_baseline.features import compute_features
from polyflip.research.strike_baseline.market_rules import MarketWindow

STATUS_OK = "ok"
STATUS_SNAPSHOT_MISSING = "snapshot_missing"
STATUS_DELAY_TOO_LARGE = "delay_too_large"
STATUS_RULES_FAIL = "rules_fail"
STATUS_NO_STRIKE = "no_strike"
STATUS_NO_UNDERLYING = "no_underlying_at_decision"
STATUS_NO_QUOTE = "no_quote"
STATUS_NO_OUTCOME = "no_outcome"
STATUS_STALE_PRICE = "stale_price"
STATUS_VOL_INSUFFICIENT = "vol_insufficient"

ENTRY_GRID = (12, 8, 5)
MAX_DELAY_SEC = 30
MAX_PRICE_AGE_SEC = 90
VOL_WINDOW_MIN = 60

ASSET_TO_SYMBOL = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "DOGE": "DOGEUSDT", "XRP": "XRPUSDT"}


@dataclass
class CandleStore:
    """Indexed closed-candle lookup (long format)."""

    df: pd.DataFrame  # columns: symbol, open_time(dt64 UTC), close_time(dt64), open,high,low,close
    by_symbol: dict[str, pd.DataFrame] = field(init=False)
    open_ns: dict[str, np.ndarray] = field(init=False)
    close_ns: dict[str, np.ndarray] = field(init=False)

    def __post_init__(self) -> None:
        frames: dict[str, pd.DataFrame] = {}
        for s, g in self.df.groupby("symbol", sort=False):
            g = g.sort_values("open_time").reset_index(drop=True)
            # Normalize to naive UTC so numpy datetime64 comparisons are tz-clean.
            g["open_time"] = g["open_time"].dt.tz_convert("UTC").dt.tz_localize(None)
            g["close_time"] = g["close_time"].dt.tz_convert("UTC").dt.tz_localize(None)
            frames[s] = g
        self.by_symbol = frames
        self.open_ns = {
            s: g["open_time"].to_numpy(dtype="datetime64[ns]")
            for s, g in frames.items()
        }
        self.close_ns = {
            s: g["close_time"].to_numpy(dtype="datetime64[ns]")
            for s, g in frames.items()
        }

    @staticmethod
    def _naive(at: pd.Timestamp) -> np.datetime64:
        if at.tzinfo is not None:
            at = at.tz_convert("UTC").tz_localize(None)
        return at.to_datetime64()

    def last_close_before(self, symbol: str, at: pd.Timestamp) -> pd.Series | None:
        g = self.by_symbol.get(symbol)
        if g is None:
            return None
        ct = self.close_ns[symbol]
        idx = int(np.searchsorted(ct, self._naive(at), side="right")) - 1
        if idx < 0:
            return None
        return g.iloc[idx]

    def first_close_in_window(self, symbol: str, wstart: pd.Timestamp, wend: pd.Timestamp) -> pd.Series | None:
        g = self.by_symbol.get(symbol)
        if g is None:
            return None
        ot = self.open_ns[symbol]
        i0 = int(np.searchsorted(ot, self._naive(wstart), side="left"))
        if i0 >= len(g):
            return None
        ct = self.close_ns[symbol]
        i1 = int(np.searchsorted(ct, self._naive(wend), side="right")) - 1
        if i1 < i0:
            return None
        return g.iloc[i0]

    def closed_returns_before(self, symbol: str, at: pd.Timestamp, window_minutes: int) -> pd.Series:
        """Log returns of closed candles in (at - window, at]."""
        g = self.by_symbol.get(symbol)
        if g is None:
            return pd.Series(dtype=float)
        ct = self.close_ns[symbol]
        limit = self._naive(at)
        lo = (pd.Timestamp(limit) - pd.Timedelta(minutes=window_minutes)).to_datetime64()
        i_hi = int(np.searchsorted(ct, limit, side="right"))
        i_lo = int(np.searchsorted(ct, lo, side="left"))
        closes = g["close"].to_numpy(dtype=float)[i_lo:i_hi]
        if closes.size < 2:
            return pd.Series(dtype=float)
        return pd.Series(np.log(closes[1:] / closes[:-1]))


def _binary_outcome(final_price: float, strike: float) -> str:
    return "YES" if final_price > strike else "NO"


def _to_ts(value) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value)


def build_ledger(
    markets: pd.DataFrame,
    windows: dict[str, MarketWindow],
    snapshots: pd.DataFrame,
    candles: CandleStore,
    entry_grid: tuple[int, ...] = ENTRY_GRID,
    max_delay_sec: int = MAX_DELAY_SEC,
    vol_window_min: int = VOL_WINDOW_MIN,
    max_price_age_sec: int = MAX_PRICE_AGE_SEC,
) -> pd.DataFrame:
    """Produce one row per (market, entry_variant). Snapshot join is causal."""
    rows: list[dict] = []
    snaps_by_market: dict[str, pd.DataFrame] = {
        mid: g.sort_values("recorded_at")
        for mid, g in snapshots.groupby("market_id")
    }

    for _, m in markets.iterrows():
        mid = m["market_id"]
        win = windows.get(mid)
        if win is None or not win.rules_ok:
            for em in entry_grid:
                rows.append(_row_base(m, win, em, STATUS_RULES_FAIL))
            continue

        wstart = _to_ts(win.window_start_utc)
        wend = _to_ts(win.window_end_utc)
        if wend.tzinfo is None:
            wend = wend.tz_localize("UTC")
        if wstart.tzinfo is None:
            wstart = wstart.tz_localize("UTC")

        # Strike proxy once per market (provenance: PROXY_BINANCE_1M_OPEN)
        sym = ASSET_TO_SYMBOL.get(m["asset"], m["asset"])
        strike_cand = candles.first_close_in_window(sym, wstart, wend)
        strike = float(strike_cand["close"]) if strike_cand is not None else None
        strike_ts = _to_ts(strike_cand["close_time"]) if strike_cand is not None else None

        g = snaps_by_market.get(mid)

        for em in entry_grid:
            row = _row_base(m, win, em, STATUS_OK)
            row["strike"] = strike
            row["strike_ts"] = strike_ts
            row["strike_provenance"] = "PROXY_BINANCE_1M_OPEN" if strike is not None else None
            if strike is None:
                row["status"] = STATUS_NO_STRIKE
                rows.append(row)
                continue

            D = wend - pd.Timedelta(minutes=em)
            row["decision_at"] = D
            if g is None or g.empty:
                row["status"] = STATUS_SNAPSHOT_MISSING
                rows.append(row)
                continue
            cand = g[g["recorded_at"] >= D]
            if cand.empty:
                row["status"] = STATUS_SNAPSHOT_MISSING
                rows.append(row)
                continue
            first = cand.iloc[0]
            delay = (first["recorded_at"] - D).total_seconds()
            if delay > max_delay_sec:
                row["status"] = STATUS_DELAY_TOO_LARGE
                row["snapshot_at"] = first["recorded_at"]
                row["snapshot_delay_sec"] = float(delay)
                rows.append(row)
                continue

            row["snapshot_at"] = first["recorded_at"]
            row["snapshot_delay_sec"] = float(delay)
            rec = first["recorded_at"]
            if not (pd.notna(first["best_bid"]) and pd.notna(first["best_ask"])
                    and 0 < first["best_bid"] < 1 and 0 < first["best_ask"] < 1
                    and first["best_bid"] < first["best_ask"]):
                row["status"] = STATUS_NO_QUOTE
                rows.append(row)
                continue
            row["yes_bid"] = float(first["best_bid"])
            row["yes_ask"] = float(first["best_ask"])
            row["yes_mid"] = (row["yes_bid"] + row["yes_ask"]) / 2.0
            row["spread"] = row["yes_ask"] - row["yes_bid"]

            # Underlying at decision: last closed candle at or before snapshot recorded_at
            ucand = candles.last_close_before(sym, rec)
            if ucand is None:
                row["status"] = STATUS_NO_UNDERLYING
                rows.append(row)
                continue
            underlying = float(ucand["close"])
            rec_naive = rec.tz_convert("UTC").tz_localize(None) if rec.tzinfo is not None else rec
            price_age = (rec_naive - ucand["close_time"]).total_seconds()
            row["underlying_at_decision"] = underlying
            row["underlying_close_time"] = ucand["close_time"]
            row["underlying_price_age_sec"] = float(price_age)
            if price_age > max_price_age_sec:
                row["status"] = STATUS_STALE_PRICE
                rows.append(row)
                continue

            # Vol features (closed candles strictly before decision)
            rets = candles.closed_returns_before(sym, rec, vol_window_min)
            feat = compute_features(underlying, strike, float(em), rets)
            row["features_status"] = feat.status
            row["sigma_min"] = feat.sigma_min
            row["z"] = feat.z
            row["p0"] = feat.p0
            row["n_vol_bars"] = feat.n_vol_bars

            # Outcome: settlement from last closed candle <= window_end vs strike (proxy)
            fend = candles.last_close_before(sym, wend + pd.Timedelta(seconds=5))
            if fend is None:
                row["status"] = STATUS_NO_OUTCOME
                rows.append(row)
                continue
            row["settlement_price"] = float(fend["close"])
            row["settlement_source"] = "BINANCE_1M_PROXY_CLOSE"
            row["outcome"] = _binary_outcome(row["settlement_price"], strike)
            row["canonical_outcome"] = m.get("final_outcome")
            rows.append(row)

    return pd.DataFrame(rows)


def _row_base(m: pd.Series, win: MarketWindow | None, entry_min: int, status: str) -> dict:
    return {
        "market_id": m.get("market_id"),
        "asset": m.get("asset"),
        "end_time_est": m.get("end_time_est"),
        "entry_min_before_close": entry_min,
        "decision_at": None,
        "status": status,
        "opportunity_id": f"{m.get('market_id')}::{entry_min}",
    }