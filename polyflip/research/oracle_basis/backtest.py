"""Executable-economics primitives for hold-to-settlement C contracts."""

from __future__ import annotations

import random
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Iterable, Literal, Sequence

DOCUMENTED_CRYPTO_TAKER_RATE = Decimal("0.07")
MONEY_QUANT = Decimal("0.000001")
LEDGER_SCHEMA_VERSION = "1"
PRICE_E12 = 10**12
P_UP_E6 = 10**6
USDC_MICROS = 10**6
Side = Literal["UP", "DOWN"]

LEDGER_COLUMNS = (
    "market_id",
    "asset",
    "horizon_sec",
    "side",
    "decision_ms",
    "assumed_fill_ms",
    "p_up_e6",
    "model_version",
    "protocol_version",
    "ask_price_e12",
    "quantity",
    "fee_micros",
    "other_costs_micros",
    "fee_scheme",
    "outcome_up",
    "pnl_micros",
    "price_sources",
    "spot_status",
    "spot_age_ms",
    "gap_bucket",
    "policy",
    "post_latency_fill",
    "depth_limited",
    "ledger_schema_version",
)


class EconomicsError(ValueError):
    """Raised for invalid economic inputs or unknown fee schemes."""


def to_money(value: object, name: str) -> Decimal:
    if isinstance(value, bool) or value is None or isinstance(value, float):
        raise TypeError(f"{name} must be Decimal/str/int, never float")
    try:
        amount = value if isinstance(value, Decimal) else Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise EconomicsError(f"invalid {name}: {value!r}") from exc
    if not amount.is_finite():
        raise EconomicsError(f"{name} must be finite")
    return amount


def to_price(value: object, name: str = "price") -> Decimal:
    price = to_money(value, name)
    if price <= 0 or price >= 1:
        raise EconomicsError(f"{name} must be in (0, 1)")
    return price


def to_contracts(value: object, name: str = "contracts") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be int")
    if value < 0:
        raise EconomicsError(f"{name} must be non-negative")
    return value


def taker_fee(
    contracts: int, price: Decimal | str | int, scheme: str = "DOCUMENTED_CRYPTO_TAKER"
) -> Decimal:
    """Documented crypto taker fee: C * 0.07 * a * (1 - a), in USDC."""
    count = to_contracts(contracts)
    key = str(scheme).strip().upper()
    if key == "ZERO":
        return Decimal("0")
    if key != "DOCUMENTED_CRYPTO_TAKER":
        raise EconomicsError(f"unknown fee scheme: {scheme!r}")
    ask = to_price(price)
    fee = Decimal(count) * DOCUMENTED_CRYPTO_TAKER_RATE * ask * (Decimal(1) - ask)
    return fee.quantize(MONEY_QUANT, rounding=ROUND_HALF_UP)


def settle_pnl_usdc(
    *,
    side: str,
    contracts: int,
    ask_price: Decimal | str | int,
    outcome_up: bool,
    fee: Decimal | str | int,
    other_costs: Decimal | str | int = 0,
) -> Decimal:
    """Hold-to-settlement PnL for C contracts bought at explicit ask `a`.

    PnL = C * (Y_side - a) - fee - other_costs. DOWN uses its own observed ask;
    callers must never synthesize ask_down as 1 - ask_up.
    """
    key = str(side).strip().upper()
    if key not in ("UP", "DOWN"):
        raise EconomicsError(f"invalid side: {side!r}")
    if not isinstance(outcome_up, bool):
        raise EconomicsError("outcome_up must be bool")
    count = to_contracts(contracts)
    ask = to_price(ask_price, "ask_price")
    fee_amount = to_money(fee, "fee")
    other = to_money(other_costs, "other_costs")
    if fee_amount < 0 or other < 0:
        raise EconomicsError("fee and other_costs must be non-negative")
    up_win = 1 if outcome_up else 0
    payoff = up_win if key == "UP" else 1 - up_win
    return Decimal(count) * (Decimal(payoff) - ask) - fee_amount - other


def size_for_budget(
    *,
    budget_usdc: Decimal | str | int,
    ask_price: Decimal | str | int,
    min_size: int,
    size_step: int,
    depth_available: int | None = None,
) -> int:
    """Contracts fitting cash including fee, lot step, and available depth."""
    budget = to_money(budget_usdc, "budget_usdc")
    ask = to_price(ask_price)
    minimum = to_contracts(min_size, "min_size")
    step = to_contracts(size_step, "size_step")
    if budget <= 0 or minimum < 1 or step < 1:
        raise EconomicsError(
            "budget must be positive; min_size and size_step at least 1"
        )
    unit_cash = ask * (Decimal(1) + DOCUMENTED_CRYPTO_TAKER_RATE * (Decimal(1) - ask))
    affordable = int(budget // unit_cash)
    if affordable < minimum:
        return 0
    sized = (affordable // step) * step
    if sized < minimum:
        return 0
    if depth_available is not None:
        depth = to_contracts(depth_available, "depth_available")
        sized = min(sized, depth)
        if sized < minimum:
            return 0
    return sized


def take_liquidity(
    depth_levels: Sequence[tuple[Decimal | str | int, int]],
    requested_contracts: int,
) -> dict[str, Decimal | int | None]:
    """Take from explicit post-latency depth levels, best price first."""
    requested = to_contracts(requested_contracts, "requested_contracts")
    levels = [
        (to_price(price, "depth price"), to_contracts(size, "depth size"))
        for price, size in depth_levels
    ]
    filled = 0
    cost = Decimal("0")
    for price, size in levels:
        if filled >= requested:
            break
        take = min(size, requested - filled)
        filled += take
        cost += price * take
    return {
        "requested": requested,
        "filled": filled,
        "shortfall": requested - filled,
        "vwap": (cost / filled) if filled else None,
    }


def ledger_row(
    *,
    market_id: str,
    asset: str,
    horizon_sec: int,
    side: str,
    contracts: int,
    fill_price: Decimal | str | int,
    outcome_up: bool,
    fee: Decimal | str | int,
    policy: str,
    post_latency_fill: bool,
    depth_limited: bool,
    other_costs: Decimal | str | int = 0,
) -> dict[str, object]:
    """One preliminary ledger row with explicit fill assumptions."""
    if not str(market_id) or not str(policy):
        raise EconomicsError("market_id and policy are required")
    pnl = settle_pnl_usdc(
        side=side,
        contracts=contracts,
        ask_price=fill_price,
        outcome_up=outcome_up,
        fee=fee,
        other_costs=other_costs,
    )
    return {
        "market_id": str(market_id),
        "asset": str(asset).strip().upper(),
        "horizon_sec": int(horizon_sec),
        "side": str(side).strip().upper(),
        "contracts": to_contracts(contracts),
        "fill_price": str(to_price(fill_price, "fill_price")),
        "outcome_up": bool(outcome_up),
        "fee_usdc": str(to_money(fee, "fee")),
        "other_costs_usdc": str(to_money(other_costs, "other_costs")),
        "pnl_usdc": str(pnl),
        "policy": str(policy),
        "post_latency_fill": bool(post_latency_fill),
        "depth_limited": bool(depth_limited),
    }


def block_bootstrap_mean_ci(
    daily_values: Iterable[float],
    n_boot: int = 5000,
    seed: int = 0,
    ci: float = 0.95,
) -> dict[str, float]:
    """Day-block bootstrap percentile CI for a mean over UTC-day blocks."""
    values = [float(v) for v in daily_values]
    if not values or any(not isinstance(v, float) or v != v for v in values):
        raise EconomicsError("daily_values must be non-empty finite numbers")
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 100:
        raise EconomicsError("n_boot must be an int >= 100")
    if not 0.0 < ci < 1.0:
        raise EconomicsError("ci must be in (0, 1)")
    rng = random.Random(int(seed))
    means = []
    for _ in range(n_boot):
        sample = [rng.choice(values) for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    low_idx = max(0, min(n_boot - 1, int((1.0 - ci) / 2.0 * n_boot)))
    high_idx = max(0, min(n_boot - 1, int((1.0 + ci) / 2.0 * n_boot)))
    return {
        "mean": sum(values) / len(values),
        "ci_low": means[low_idx],
        "ci_high": means[high_idx],
        "n_boot": float(n_boot),
        "seed": float(seed),
        "blocks": float(len(values)),
    }


def _scaled_int(amount: Decimal, scale: int, name: str) -> int:
    scaled = (amount * scale).to_integral_value(rounding=ROUND_HALF_UP)
    return int(scaled)


def ledger_row_v2(
    *,
    market_id: str,
    asset: str,
    horizon_sec: int,
    side: str,
    decision_ms: int,
    assumed_fill_ms: int,
    p_up: Decimal | str | float,
    model_version: str,
    protocol_version: str,
    ask_price: Decimal | str,
    quantity: int,
    fee: Decimal | str,
    fee_scheme: str,
    outcome_up: bool,
    other_costs: Decimal | str | int = 0,
    price_sources: str = "",
    spot_status: str = "UNKNOWN",
    spot_age_ms: int | None = None,
    gap_bucket: str = "NO_OFFICIAL_REFERENCE",
    policy: str = "",
    post_latency_fill: bool = False,
    depth_limited: bool = False,
) -> dict[str, object]:
    """Frozen ledger-schema v1 row with scaled-integer money (no floats stored).

    p_up accepts float only at this audited boundary and is range-checked;
    every stored field is int/str/bool.
    """
    if not str(market_id) or not str(model_version) or not str(protocol_version):
        raise EconomicsError(
            "market_id, model_version and protocol_version are required"
        )
    if isinstance(p_up, bool) or not isinstance(p_up, (Decimal, str, float, int)):
        raise EconomicsError("p_up must be numeric")
    probability = Decimal(str(p_up))
    if probability < 0 or probability > 1:
        raise EconomicsError("p_up must be in [0, 1]")
    ask = to_price(ask_price, "ask_price")
    count = to_contracts(quantity, "quantity")
    fee_amount = to_money(fee, "fee")
    other = to_money(other_costs, "other_costs")
    pnl = settle_pnl_usdc(
        side=side,
        contracts=count,
        ask_price=ask,
        outcome_up=bool(outcome_up),
        fee=fee_amount,
        other_costs=other,
    )
    decision = to_contracts(decision_ms, "decision_ms")
    fill = to_contracts(assumed_fill_ms, "assumed_fill_ms")
    if fill < decision:
        raise EconomicsError("assumed_fill_ms must be >= decision_ms")
    if spot_age_ms is not None:
        spot_age_ms = to_contracts(spot_age_ms, "spot_age_ms")
    return {
        "market_id": str(market_id),
        "asset": str(asset).strip().upper(),
        "horizon_sec": to_contracts(horizon_sec, "horizon_sec"),
        "side": str(side).strip().upper(),
        "decision_ms": decision,
        "assumed_fill_ms": fill,
        "p_up_e6": _scaled_int(probability, P_UP_E6, "p_up"),
        "model_version": str(model_version),
        "protocol_version": str(protocol_version),
        "ask_price_e12": _scaled_int(ask, PRICE_E12, "ask_price"),
        "quantity": count,
        "fee_micros": _scaled_int(fee_amount, USDC_MICROS, "fee"),
        "other_costs_micros": _scaled_int(other, USDC_MICROS, "other_costs"),
        "fee_scheme": str(fee_scheme),
        "outcome_up": bool(outcome_up),
        "pnl_micros": _scaled_int(pnl, USDC_MICROS, "pnl"),
        "price_sources": str(price_sources),
        "spot_status": str(spot_status),
        "spot_age_ms": spot_age_ms,
        "gap_bucket": str(gap_bucket),
        "policy": str(policy),
        "post_latency_fill": bool(post_latency_fill),
        "depth_limited": bool(depth_limited),
        "ledger_schema_version": LEDGER_SCHEMA_VERSION,
    }


def write_ledger_parquet(rows: list[dict[str, object]], path: str) -> str:
    """Write ledger-schema v1 rows to Parquet with an explicit Arrow schema."""
    import hashlib

    import pyarrow as pa
    import pyarrow.parquet as pq

    for row in rows:
        missing = [c for c in LEDGER_COLUMNS if c not in row]
        if missing:
            raise EconomicsError(f"ledger row missing columns: {missing}")
        if str(row.get("ledger_schema_version")) != LEDGER_SCHEMA_VERSION:
            raise EconomicsError("ledger row has wrong schema version")
    schema = pa.schema(
        [
            ("market_id", pa.string()),
            ("asset", pa.string()),
            ("horizon_sec", pa.int64()),
            ("side", pa.string()),
            ("decision_ms", pa.int64()),
            ("assumed_fill_ms", pa.int64()),
            ("p_up_e6", pa.int64()),
            ("model_version", pa.string()),
            ("protocol_version", pa.string()),
            ("ask_price_e12", pa.int64()),
            ("quantity", pa.int64()),
            ("fee_micros", pa.int64()),
            ("other_costs_micros", pa.int64()),
            ("fee_scheme", pa.string()),
            ("outcome_up", pa.bool_()),
            ("pnl_micros", pa.int64()),
            ("price_sources", pa.string()),
            ("spot_status", pa.string()),
            ("spot_age_ms", pa.int64()),
            ("gap_bucket", pa.string()),
            ("policy", pa.string()),
            ("post_latency_fill", pa.bool_()),
            ("depth_limited", pa.bool_()),
            ("ledger_schema_version", pa.string()),
        ]
    )
    table = pa.Table.from_pylist(
        [{c: row[c] for c in LEDGER_COLUMNS} for row in rows], schema=schema
    )
    pq.write_table(table, path)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
