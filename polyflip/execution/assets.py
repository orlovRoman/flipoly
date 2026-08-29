"""Canonical asset selection helpers for LIVE trading sessions."""

from __future__ import annotations

LIVE_TRADING_ASSETS = ("BTC", "ETH", "SOL", "XRP", "DOGE")


def normalize_live_asset(value: object) -> str:
    """Return a canonical asset symbol or raise a user-facing validation error."""
    asset = str(value or "").strip().upper()
    if asset.endswith("USDT"):
        asset = asset[:-4]
    if asset not in LIVE_TRADING_ASSETS:
        available = ", ".join(LIVE_TRADING_ASSETS)
        raise ValueError(
            f"Недопустимый актив {value!r}. Доступны только: {available}"
        )
    return asset


def normalize_live_assets(
    values: object | None,
    *,
    default_all: bool = True,
) -> list[str]:
    """
    Normalize an API/JSON asset list into stable UI order.

    None means all assets for legacy sessions when default_all is true.
    Explicit empty input is rejected so a LIVE session cannot be activated with
    no tradable asset.
    """
    if values is None:
        raw_values = list(LIVE_TRADING_ASSETS) if default_all else []
    elif isinstance(values, str):
        raw_values = values.split(",")
    else:
        try:
            raw_values = list(values)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError("Список активов должен быть массивом строк") from exc

    selected: set[str] = set()
    for value in raw_values:
        if str(value or "").strip():
            selected.add(normalize_live_asset(value))

    if not selected:
        raise ValueError("Выберите хотя бы один актив для LIVE-торговли")

    return [asset for asset in LIVE_TRADING_ASSETS if asset in selected]
