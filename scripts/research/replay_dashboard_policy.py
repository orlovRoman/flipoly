"""Run the fixed dashboard-policy replay locally.

The server is used only to export a causal ledger.  This command performs the
heavy calculation locally and writes a per-opportunity CSV plus JSON summary.
No threshold search or model fitting is performed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from polyflip.research.policy_replay import replay_opportunities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    frame = pd.read_parquet(args.input) if args.input.suffix.lower() == ".parquet" else pd.read_csv(args.input)
    rows, summary = replay_opportunities(frame.to_dict(orient="records"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "policy_replay_opportunities.csv", index=False)
    (args.output_dir / "policy_replay_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

