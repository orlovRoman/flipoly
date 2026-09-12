"""Step 17: CT as a separate policy feature via the FIXED implementation.

Fixed realization (do NOT replace):
- spec: BTC_CT_T5_V1 from ``polyflip.trading.ct_policy`` (branch
  ``feature/paper-ct-outsider``, commit ``2b531a2``; files vendored
  byte-identical into this tree).
- wrapper: ``compute_token_ct_regime`` with the SPEC's own
  ``min_observations=3`` (note: ``classify_local_regime`` alone defaults to
  4 — always call through the wrapper, never the classifier directly).
- history: previous 900 s of the CHOSEN token, causal (<= decision_at).
- entry only on REVERSION; states are REVERSION/TREND/QUIET/UNCERTAIN.

Scope note: the spec asset is BTC. Other assets keep ct_regime=UNCERTAIN
with reason ASSET_OUT_OF_CT_SCOPE until a per-asset spec is pinned.

Statuses INSUFFICIENT_HISTORY / INVALID_SERIES / CALCULATION_ERROR are
kept verbatim (never mapped to a trend regime).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

from polyflip.trading.ct_policy import (
    compute_token_ct_regime,
    get_btc_ct_t5_v1_spec,
)

CT_SPEC_ID = "BTC_CT_T5_V1"
CT_SPEC_COMMIT = "2b531a2"


@dataclass(frozen=True)
class CTResult:
    signal: str | None  # regime state: REVERSION | TREND | QUIET | UNCERTAIN
    regime: str          # same as signal (kept for policy readability)
    status: str          # VALID | INSUFFICIENT_HISTORY | INVALID_SERIES | CALCULATION_ERROR | ASSET_OUT_OF_SCOPE | NO_HISTORY
    version: str         # spec_id
    spec_hash: str
    params: dict
    n_obs: int
    source: str


def _spec():
    return get_btc_ct_t5_v1_spec()


def compute_ct(history: Sequence[Any] | None,
               meta: dict | None = None,
               decision_at: datetime | None = None) -> CTResult:
    meta = dict(meta or {})
    spec = _spec()
    asset = str(meta.get("asset", spec.asset)).upper()
    source = str(meta.get("source", "unknown"))
    if asset != spec.asset.upper():
        return CTResult(None, "UNCERTAIN", "ASSET_OUT_OF_SCOPE",
                        spec.spec_id, spec.spec_hash,
                        dict(spec.classifier_params), 0, source)
    if not history:
        return CTResult(None, "UNCERTAIN", "NO_HISTORY",
                        spec.spec_id, spec.spec_hash,
                        dict(spec.classifier_params), 0, source)
    if decision_at is None:
        # Without a causal boundary the wrapper cannot run: refuse, do not
        # invent a timestamp (step 12: unknown time is never zeroed).
        return CTResult(None, "UNCERTAIN", "NO_HISTORY",
                        spec.spec_id, spec.spec_hash,
                        dict(spec.classifier_params), len(history), source)
    res = compute_token_ct_regime(
        observations=history,
        decision_at=decision_at,
        window_sec=spec.ct_history_window_sec,
        min_observations=spec.min_observations,
        classifier_params=dict(spec.classifier_params),
    )
    return CTResult(res.state, res.state, res.status,
                    spec.spec_id, spec.spec_hash,
                    dict(spec.classifier_params),
                    res.observations_count, source)


def ct_allows_entry(result: CTResult) -> bool:
    """Entry only on a VALID REVERSION signal (spec.required_signal)."""
    return result.status == "VALID" and result.regime == _spec().required_signal
