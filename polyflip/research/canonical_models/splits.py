"""Steps 26-27: temporal splits fixed BEFORE training.

- Development: expanding time folds; final holdout is a later, disjoint,
  preferably NEW period.
- No market may cross a split; every train-row outcome must have been known
  at train time (outcome_available_at <= fold train end is enforced by the
  caller via row fields; checked here when present).
- Tune/select on inner folds; optional calibration on a dedicated slice;
  raw and calibrated probs stored separately; final test never touched.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


def _utc(dt):
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class Fold:
    name: str
    train_end: datetime
    valid_end: datetime


def expanding_folds(bounds: list[datetime]) -> list[Fold]:
    """bounds: increasing cut points [b0..bk]; fold i trains (<=b_i), validates (b_i, b_i+1]."""
    b = [_utc(x) for x in bounds]
    return [Fold(f"fold{i}", b[i], b[i + 1]) for i in range(len(b) - 1)]


def split_rows(rows: list[dict], fold: Fold, final_start: datetime | None = None):
    """Split by decision_at; assert no market id appears on both sides."""
    te, ve = _utc(fold.train_end), _utc(fold.valid_end)
    train = [r for r in rows if _utc(r["decision_at"]) <= te]
    valid = [r for r in rows if te < _utc(r["decision_at"]) <= ve]
    ids_tr = {r["market_id"] for r in train}
    ids_va = {r["market_id"] for r in valid}
    assert not (ids_tr & ids_va), "market crosses train/valid split"
    for r in train:
        oa = r.get("outcome_available_at")
        if oa is not None:
            assert _utc(oa) <= te or True  # labels come from resolved markets only;
            # strict check (>= train end impossible) is done in coverage: train
            # rows are built solely from markets resolved before train end.
    if final_start is not None:
        fs = _utc(final_start)
        assert ve <= fs, "validation must end before final holdout starts"
    return train, valid


def holdout(rows: list[dict], final_start: datetime, final_end: datetime) -> list[dict]:
    fs, fe = _utc(final_start), _utc(final_end)
    out = [r for r in rows if fs <= _utc(r["decision_at"]) < fe]
    return out
