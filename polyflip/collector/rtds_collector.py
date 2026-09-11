"""RTDS oracle observation collector helpers.

Research-only pure helpers for Polymarket real-time data (RTDS) spot/TWAP
observations. No trading decisions are made here and no application database
models are imported, so synthetic replay tests stay hermetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Iterable, Literal, Mapping
import re

E18 = 10**18
RTDSSource = Literal["BINANCE", "CHAINLINK_OFFICIAL"]
SYMBOL_RE = re.compile(r"^[A-Z0-9]+(?:USDT|USD)$")
ASSET_RE = re.compile(r"^[A-Z0-9]{2,10}$")


class RTDSError(ValueError):
    """Raised for malformed RTDS observations or stream misuse."""


def parse_price_e18(value: Any) -> int:
    """Parse a positive decimal price into integer E18 units without floats."""
    if isinstance(value, bool) or value is None or isinstance(value, float):
        raise TypeError("price must be a decimal string or Decimal, never float")
    if isinstance(value, Decimal):
        amount = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("price is empty")
        try:
            amount = Decimal(text)
        except InvalidOperation as exc:
            raise ValueError(f"invalid price: {value!r}") from exc
    else:
        raise TypeError("price must be a decimal string or Decimal, never float")
    if not amount.is_finite() or amount <= 0:
        raise ValueError("price must be finite and positive")
    return int((amount * E18).to_integral_value(rounding=ROUND_HALF_UP))


def round_half_up_div(numerator: int, denominator: int) -> int:
    """Round a non-negative integer division to nearest, halves up."""
    if isinstance(numerator, bool) or not isinstance(numerator, int):
        raise TypeError("numerator must be int")
    if isinstance(denominator, bool) or not isinstance(denominator, int):
        raise TypeError("denominator must be int")
    if numerator < 0 or denominator <= 0:
        raise ValueError("round_half_up_div needs numerator >= 0 and denominator > 0")
    return (numerator + denominator // 2) // denominator


def parse_timestamp_ms(value: Any) -> int:
    """Parse an integer millisecond timestamp without floats."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("timestamp must be integer milliseconds")
    if value < 0:
        raise ValueError("timestamp must be non-negative")
    return value


def normalize_asset(value: Any) -> str:
    text = str(value).strip().upper()
    if not ASSET_RE.match(text):
        raise RTDSError(f"invalid asset: {value!r}")
    return text


def normalize_stream(
    asset: Any, currency: Any = None, symbol: Any = None
) -> tuple[str, str, str]:
    """Normalize (asset, symbol, currency) without mixing USD and USDT."""
    norm_asset = normalize_asset(asset)
    if symbol is not None:
        sym = str(symbol).strip().upper()
        if not SYMBOL_RE.match(sym) or not sym.startswith(norm_asset):
            raise RTDSError(f"invalid symbol for asset {norm_asset}: {symbol!r}")
        suffix = "USDT" if sym.endswith("USDT") else "USD"
        if currency is not None and str(currency).strip().upper() != suffix:
            raise RTDSError(f"symbol/currency conflict: {symbol!r} vs {currency!r}")
        return norm_asset, sym, suffix
    if currency is None:
        raise RTDSError("currency is required when symbol is absent")
    cur = str(currency).strip().upper()
    if cur not in ("USD", "USDT"):
        raise RTDSError(f"invalid currency: {currency!r}")
    return norm_asset, f"{norm_asset}{cur}", cur


@dataclass(frozen=True)
class RTDSObservation:
    source: str
    asset: str
    symbol: str
    currency: str
    price_e18: int
    observed_at_ms: int
    received_at_ms: int
    seq: int | None = None


def validate_observation(raw: Mapping[str, Any]) -> RTDSObservation:
    """Validate one raw RTDS observation mapping into canonical form."""
    if not isinstance(raw, Mapping):
        raise RTDSError("observation must be a mapping")
    try:
        source = str(raw["source"]).strip().upper()
        price_e18 = parse_price_e18(raw["price"])
        observed_at_ms = parse_timestamp_ms(raw["observed_at"])
        received_at_ms = parse_timestamp_ms(raw["received_at"])
        asset, symbol, currency = normalize_stream(
            raw["asset"], raw.get("currency"), raw.get("symbol")
        )
    except KeyError as exc:
        raise RTDSError(f"missing observation field: {exc}") from exc
    if source not in ("BINANCE", "CHAINLINK_OFFICIAL"):
        raise RTDSError(f"invalid source: {raw.get('source')!r}")
    if received_at_ms < observed_at_ms:
        raise RTDSError("received_at precedes observed_at; causality violated")
    seq = raw.get("seq")
    if seq is not None:
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
            raise RTDSError(f"invalid seq: {seq!r}")
    return RTDSObservation(
        source=source,
        asset=asset,
        symbol=symbol,
        currency=currency,
        price_e18=price_e18,
        observed_at_ms=observed_at_ms,
        received_at_ms=received_at_ms,
        seq=seq,
    )


@dataclass
class StreamTracker:
    """Arrival-order accounting for one (source, symbol) stream."""

    source: str
    symbol: str
    gap_threshold_ms: int = 2000
    count: int = 0
    first_received_ms: int | None = None
    last_received_ms: int | None = None
    gaps_over_threshold: int = 0
    max_gap_ms: int = 0
    out_of_order: int = 0

    def add(self, obs: RTDSObservation) -> str:
        if obs.source != self.source or obs.symbol != self.symbol:
            raise RTDSError("observation belongs to another stream")
        if self.count == 0:
            self.first_received_ms = obs.received_at_ms
            self.last_received_ms = obs.received_at_ms
            self.count = 1
            return "FIRST"
        assert self.last_received_ms is not None
        if obs.received_at_ms < self.last_received_ms:
            self.out_of_order += 1
            self.count += 1
            return "OUT_OF_ORDER"
        delta = obs.received_at_ms - self.last_received_ms
        if delta > self.gap_threshold_ms:
            self.gaps_over_threshold += 1
            self.max_gap_ms = max(self.max_gap_ms, delta)
        self.last_received_ms = obs.received_at_ms
        self.count += 1
        return "GAP" if delta > self.gap_threshold_ms else "OK"

    def stale_age_ms(self, at_ms: int) -> int:
        at = parse_timestamp_ms(at_ms)
        if self.last_received_ms is None:
            raise RTDSError("stream has no observations")
        return max(0, at - self.last_received_ms)

    def summary(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "symbol": self.symbol,
            "count": self.count,
            "first_received_ms": self.first_received_ms,
            "last_received_ms": self.last_received_ms,
            "gaps_over_threshold": self.gaps_over_threshold,
            "max_gap_ms": self.max_gap_ms,
            "out_of_order": self.out_of_order,
        }


def select_asof(
    observations: Iterable[RTDSObservation],
    decision_received_ms: int,
) -> RTDSObservation | None:
    """Backward as-of selection: latest observation received at decision time."""
    decision_ms = parse_timestamp_ms(decision_received_ms)
    best: RTDSObservation | None = None
    best_key: tuple[int, int, int] | None = None
    for obs in observations:
        if obs.received_at_ms <= decision_ms:
            key = (obs.received_at_ms, obs.observed_at_ms, obs.price_e18)
            if best_key is None or key > best_key:
                best = obs
                best_key = key
    return best
