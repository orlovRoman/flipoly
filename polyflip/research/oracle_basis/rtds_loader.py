"""Backward as-of loading for RTDS observations with explicit data-quality ledger."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

from polyflip.collector.rtds_collector import (
    RTDSObservation,
    RTDSError,
    StreamTracker,
    parse_timestamp_ms,
    select_asof,
    validate_observation,
)


def load_observations(
    raw_rows: Iterable[Mapping[str, Any]],
) -> tuple[list[RTDSObservation], dict[str, Any]]:
    """Validate raw rows, resolve duplicates/conflicts deterministically, sort as-of.

    Duplicate key is (source, symbol, observed_at). For each key the keeper is the
    earliest received observation (ties broken by lower price). Same-price extras
    are duplicates; different-price extras are conflicts. The first received value
    wins; conflicts are counted, never silently averaged.
    """
    grouped: dict[tuple[str, str, int], list[RTDSObservation]] = defaultdict(list)
    rejected: list[dict[str, Any]] = []
    for index, raw in enumerate(list(raw_rows)):
        try:
            obs = validate_observation(raw)
        except (RTDSError, TypeError, ValueError) as exc:
            rejected.append({"index": index, "reason": str(exc)})
            continue
        grouped[(obs.source, obs.symbol, obs.observed_at_ms)].append(obs)

    keepers: list[RTDSObservation] = []
    duplicates = 0
    conflicts = 0
    for key in sorted(grouped):
        candidates = sorted(grouped[key], key=lambda o: (o.received_at_ms, o.price_e18))
        keeper = candidates[0]
        keepers.append(keeper)
        for extra in candidates[1:]:
            if extra.price_e18 == keeper.price_e18:
                duplicates += 1
            else:
                conflicts += 1
    keepers.sort(key=lambda o: (o.received_at_ms, o.observed_at_ms, o.price_e18))

    trackers: dict[tuple[str, str], StreamTracker] = {}
    for obs in keepers:
        tracker = trackers.get((obs.source, obs.symbol))
        if tracker is None:
            tracker = StreamTracker(source=obs.source, symbol=obs.symbol)
            trackers[(obs.source, obs.symbol)] = tracker
        tracker.add(obs)
    summary = {
        "rows_in": (
            len(list(raw_rows)) if not isinstance(raw_rows, list) else len(raw_rows)
        ),
        "rows_kept": len(keepers),
        "rejected": rejected,
        "rejected_count": len(rejected),
        "duplicates": duplicates,
        "conflicts": conflicts,
        "streams": {
            f"{s}|{sym}": t.summary() for (s, sym), t in sorted(trackers.items())
        },
    }
    return keepers, summary


def observations_for_stream(
    observations: Iterable[RTDSObservation], source: str, symbol: str
) -> list[RTDSObservation]:
    """Filter one stream in as-of order."""
    src = str(source).strip().upper()
    sym = str(symbol).strip().upper()
    return sorted(
        (o for o in observations if o.source == src and o.symbol == sym),
        key=lambda o: (o.received_at_ms, o.observed_at_ms, o.price_e18),
    )


def select_checkpoint_observation(
    stream_obs: Iterable[RTDSObservation],
    checkpoint_received_ms: int,
    max_age_ms: int,
) -> tuple[RTDSObservation | None, str, int | None]:
    """Latest observation known at checkpoint, allowing only past receipts.

    Returns (observation_or_none, status, age_ms_or_none) with status in
    OK/NO_DATA/STALE. Stale checkpoints are missing inputs, not estimates.
    """
    checkpoint_ms = parse_timestamp_ms(checkpoint_received_ms)
    if (
        isinstance(max_age_ms, bool)
        or not isinstance(max_age_ms, int)
        or max_age_ms < 0
    ):
        raise RTDSError("max_age_ms must be a non-negative int")
    obs = select_asof(stream_obs, checkpoint_ms)
    if obs is None:
        return None, "NO_DATA", None
    age = checkpoint_ms - obs.received_at_ms
    if age > max_age_ms:
        return None, "STALE", age
    return obs, "OK", age


def checkpoint_table(
    stream_obs: Iterable[RTDSObservation],
    checkpoints_ms: Iterable[int],
    max_age_ms: int,
) -> list[dict[str, Any]]:
    ordered = sorted(
        stream_obs, key=lambda o: (o.received_at_ms, o.observed_at_ms, o.price_e18)
    )
    table = []
    for checkpoint in checkpoints_ms:
        obs, status, age = select_checkpoint_observation(
            ordered, checkpoint, max_age_ms
        )
        table.append(
            {
                "checkpoint_received_ms": parse_timestamp_ms(checkpoint),
                "status": status,
                "age_ms": age,
                "price_e18": obs.price_e18 if obs else None,
                "observed_at_ms": obs.observed_at_ms if obs else None,
                "received_at_ms": obs.received_at_ms if obs else None,
            }
        )
    return table
