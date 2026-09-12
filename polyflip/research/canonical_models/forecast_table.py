"""Step 29: forecast table.

Per variant: Brier + log loss (ONCE per market), reliability by predicted-prob
bins, calibration in REAL-ASK bins of width 0.05, splits by UP/DOWN side,
asset and week, with market counts. Price bins and prob bins are never mixed.
"""
from __future__ import annotations

import math
from collections import defaultdict


def brier(ps: list[float], ys: list[int]) -> float:
    return sum((p - y) ** 2 for p, y in zip(ps, ys)) / max(len(ps), 1)


def logloss(ps: list[float], ys: list[int], eps: float = 1e-6) -> float:
    s = 0.0
    for p, y in zip(ps, ys):
        p = min(max(p, eps), 1 - eps)
        s += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return s / max(len(ps), 1)


def reliability(ps: list[float], ys: list[int], n_bins: int = 10) -> list[dict]:
    bins = [{"n": 0, "mean_p": 0.0, "freq": 0.0, "lo": i / n_bins, "hi": (i + 1) / n_bins}
            for i in range(n_bins)]
    for p, y in zip(ps, ys):
        i = min(int(p * n_bins), n_bins - 1)
        b = bins[i]
        b["n"] += 1
        b["mean_p"] += p
        b["freq"] += y
    for b in bins:
        if b["n"]:
            b["mean_p"] /= b["n"]
            b["freq"] /= b["n"]
    return bins


def ask_calibration(rows: list[dict], prob_key: str, width: float = 0.05) -> list[dict]:
    """Calibration of predicted prob within bins of the REAL observed ask."""
    buckets: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for r in rows:
        ask = r.get("exec_ask")
        if ask is None:
            continue
        buckets[int(ask // width)].append((float(r[prob_key]), int(r["target_up"])))
    out = []
    for k in sorted(buckets):
        ps = [p for p, _ in buckets[k]]
        ys = [y for _, y in buckets[k]]
        out.append({"ask_lo": k * width, "ask_hi": k * width + width,
                    "n": len(ps), "mean_p": sum(ps) / len(ps),
                    "freq": sum(ys) / len(ys), "brier": brier(ps, ys)})
    return out


def variant_table(rows: list[dict], prob_key: str) -> dict:
    ps = [float(r[prob_key]) for r in rows]
    ys = [int(r["target_up"]) for r in rows]
    by: dict[str, dict] = {}
    for key in ("asset", "side", "week"):
        groups: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(rows):
            groups[str(r.get(key, "?"))].append(i)
        by[key] = {g: {"n": len(ix), "brier": brier([ps[i] for i in ix], [ys[i] for i in ix]),
                       "logloss": logloss([ps[i] for i in ix], [ys[i] for i in ix])}
                   for g, ix in sorted(groups.items())}
    return {"n": len(rows), "brier": brier(ps, ys), "logloss": logloss(ps, ys),
            "reliability": reliability(ps, ys),
            "ask_calibration": ask_calibration(rows, prob_key),
            "by": by}
