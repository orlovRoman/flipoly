"""Run the oracle-basis model comparison on a built dataset.

Chronological whole-day splits come from markets.json; probability columns present
in dataset.json are scored per split against real outcomes. B0 always runs from
book_mid_up (raw plus a train-fitted Platt calibration). B1/B2/B3 columns are
scored when present; missing model columns are reported as DATA_INSUFFICIENT,
never invented.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

# Project root on sys.path (repo convention for standalone scripts).
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)

from polyflip.research.oracle_basis.basis_dataset import split_markets
from polyflip.research.oracle_basis.models import (
    apply_b0_calibration,
    brier_score,
    compare_incremental,
    fit_b0_calibration,
    hit_rate,
    log_loss,
)

MODEL_COLUMNS = (("B0", "p_b0"), ("B1", "p_b1"), ("B2", "p_b2"), ("B3", "p_b3"))


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def score_rows(rows: list[dict[str, Any]], column: str) -> dict[str, Any] | None:
    eligible = [
        r for r in rows if r.get(column) is not None and r.get("outcome_up") is not None
    ]
    if not eligible:
        return None
    probs = [float(r[column]) for r in eligible]
    labels = [1 if r["outcome_up"] else 0 for r in eligible]
    return {
        "n": len(eligible),
        "brier": brier_score(probs, labels),
        "log_loss": log_loss(probs, labels),
        "hit_rate": hit_rate(probs, labels),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score oracle-basis model probabilities."
    )
    parser.add_argument(
        "--dataset", required=True, help="dataset.json from build_dataset.py."
    )
    parser.add_argument(
        "--markets", required=True, help="markets.json from export_data.py."
    )
    parser.add_argument(
        "--out-dir", required=True, help="Output directory for comparison JSON."
    )
    parser.add_argument("--train-frac", type=float, default=0.60)
    parser.add_argument("--val-frac", type=float, default=0.20)
    parser.add_argument("--test-frac", type=float, default=0.20)
    parser.add_argument("--embargo-ms", type=int, default=900000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = read_json(args.dataset)
    markets = read_json(args.markets)
    split = split_markets(
        [
            {
                "market_id": m["market_id"],
                "utc_day": datetime.fromtimestamp(m["end_ms"] / 1000.0, tz=timezone.utc)
                .date()
                .isoformat(),
                "start_ms": m.get("start_ms"),
                "end_ms": m["end_ms"],
            }
            for m in markets
        ],
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        embargo_ms=args.embargo_ms,
    )
    market_split = {mid: name for name, mids in split["splits"].items() for mid in mids}
    usable = [
        r
        for r in rows
        if r["market_id"] in market_split and not r.get("exclusion_reasons")
    ]
    by_split: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for row in usable:
        by_split[market_split[row["market_id"]]].append(row)

    # B0 from the correct UP mid; calibration fitted on train only.
    for row in usable:
        row["p_b0"] = row.get("book_mid_up")
    train_mids = [
        r["p_b0"]
        for r in by_split["train"]
        if r["p_b0"] is not None and r["outcome_up"] is not None
    ]
    train_labels = [
        1 if r["outcome_up"] else 0
        for r in by_split["train"]
        if r["p_b0"] is not None and r["outcome_up"] is not None
    ]
    calibration = None
    if train_mids:
        model = fit_b0_calibration(train_mids, train_labels)
        for row in usable:
            row["p_b0_cal"] = (
                apply_b0_calibration(model, [float(row["p_b0"])])[0]
                if row["p_b0"] is not None
                else None
            )
        calibration = {
            "n_train": len(train_mids),
            "method": "platt_logistic_train_only",
        }

    comparison: dict[str, Any] = {
        "splits": split,
        "excluded": split["excluded"],
        "models": {},
        "incremental_vs_B1": {},
    }
    for name, column in MODEL_COLUMNS + (("B0_cal", "p_b0_cal"),):
        per_split = {}
        for split_name, split_rows in by_split.items():
            scored = score_rows(split_rows, column)
            per_split[split_name] = scored or {"status": "DATA_INSUFFICIENT"}
        comparison["models"][name] = per_split
    base = comparison["models"].get("B1", {}).get("test")
    if isinstance(base, dict) and "brier" in base:
        for name in ("B2", "B3", "B0", "B0_cal"):
            upgraded = comparison["models"].get(name, {}).get("test")
            if isinstance(upgraded, dict) and "brier" in upgraded:
                comparison["incremental_vs_B1"][name] = compare_incremental(
                    {"brier": base["brier"], "log_loss": base["log_loss"]},
                    {"brier": upgraded["brier"], "log_loss": upgraded["log_loss"]},
                )

    os.makedirs(args.out_dir, exist_ok=True)
    write_json(os.path.join(args.out_dir, "model_comparison.json"), comparison)
    write_json(
        os.path.join(args.out_dir, "calibration.json"),
        calibration or {"status": "DATA_INSUFFICIENT"},
    )
    print(
        f"usable_rows={len(usable)} splits={ {k: len(v) for k, v in by_split.items()} }"
    )
    print(f"wrote {args.out_dir}/model_comparison.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
