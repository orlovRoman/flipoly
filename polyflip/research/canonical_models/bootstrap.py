"""Step 30: paired bootstrap over whole UTC days (local only).

- Days are resampled with replacement; ALL assets of a day stay together.
- Every repeat compares the SAME markets across variants (paired).
- UP/DOWN legs and model variants are never resampled independently.
- Few days -> report the limitation + multi-day-block sensitivity.
"""
from __future__ import annotations

import random
from collections import defaultdict
from datetime import timezone

from .forecast_table import brier, logloss
from .guards import assert_train_allowed


def _day(r: dict) -> str:
    dt = r["decision_at"]
    if not dt.tzinfo:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).date().isoformat()


def paired_day_bootstrap(rows: list[dict], keys_a: str, keys_b: str,
                         metric: str = "brier", n_boot: int = 2000,
                         seed: int = 42, workdir: str = ".") -> dict:
    assert_train_allowed(workdir)
    assert metric in ("brier", "logloss")
    by_day: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_day[_day(r)].append(r)
    days = sorted(by_day)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        sample = []
        for _ in days:
            sample.extend(by_day[rng.choice(days)])
        # paired: same markets for A and B by construction
        if metric == "brier":
            da = brier([x[keys_a] for x in sample], [x["target_up"] for x in sample])
            db = brier([x[keys_b] for x in sample], [x["target_up"] for x in sample])
        else:
            da = logloss([x[keys_a] for x in sample], [x["target_up"] for x in sample])
            db = logloss([x[keys_b] for x in sample], [x["target_up"] for x in sample])
        diffs.append(db - da)  # negative => B better
    diffs.sort()
    def q(p: float) -> float:
        return diffs[min(int(p * n_boot), n_boot - 1)]
    return {"n_days": len(days), "n_boot": n_boot, "metric": metric,
            "mean": sum(diffs) / len(diffs),
            "ci95": [q(0.025), q(0.975)], "ci90": [q(0.05), q(0.95)],
            "limitation": "wide CI expected when n_days is small; use multi-day blocks as sensitivity",
            "paired": True}
