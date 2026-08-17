"""Stage 10: Generate structured comparison reports and model selection summary.

Reads candidate evaluation results or audits the database, producing:
1. logreg_candidate_comparison_YYYYMMDD.json
2. logreg_candidate_comparison_YYYYMMDD.csv
3. logreg_selection_YYYYMMDD.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import select

from polyflip.constants import COMBINED_MODE_SUPPORTED_ASSETS, PRICE_PHASE_BOUNDARIES
from polyflip.db.connection import async_session
from polyflip.db.models import ModelRegistry

try:
    from polyflip.scripts.audit_logreg_models import audit_models
except ImportError:
    from audit_logreg_models import audit_models


async def generate_selection_summary(
    audit_data: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Analyze all audited candidates and recommend runtime configuration."""
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    output_dir.mkdir(parents=True, exist_ok=True)

    models_by_target: dict[str, list[dict[str, Any]]] = {}
    for m in audit_data.get("models", []):
        asset = m.get("asset")
        models_by_target.setdefault(asset, []).append(m)

    recommendations: dict[str, Any] = {}
    phase_keys = list(PRICE_PHASE_BOUNDARIES.keys())

    for base_asset in sorted(COMBINED_MODE_SUPPORTED_ASSETS):
        base_models = models_by_target.get(base_asset, [])
        base_best = next((m for m in base_models if m.get("deployable")), None)
        base_status = "DEPLOYABLE" if base_best else "UNDEPLOYABLE"

        recommendations[base_asset] = {
            "base_model": {
                "status": base_status,
                "recommended_candidate_id": base_best.get("model_id") if base_best else None,
                "version": base_best.get("version") if base_best else None,
                "combined_pnl": base_best.get("combined_branch", {}).get("total_pnl") if base_best else None,
                "median_window_pnl": base_best.get("oot_windows", {}).get("median_pnl") if base_best else None,
            },
            "phases": {},
        }

        for phase in phase_keys:
            phase_asset = f"{base_asset}_{phase}"
            phase_models = models_by_target.get(phase_asset, [])
            phase_best = next((m for m in phase_models if m.get("deployable")), None)

            if phase_best:
                resolution = "PRIMARY"
                rec_id = phase_best.get("model_id")
                rec_ver = phase_best.get("version")
                pnl = phase_best.get("combined_branch", {}).get("total_pnl")
                median_pnl = phase_best.get("oot_windows", {}).get("median_pnl")
                note = "Phase candidate passed all OOT criteria."
            elif base_best:
                resolution = "FALLBACK_BASE"
                rec_id = base_best.get("model_id")
                rec_ver = base_best.get("version")
                pnl = base_best.get("combined_branch", {}).get("total_pnl")
                median_pnl = base_best.get("oot_windows", {}).get("median_pnl")
                note = f"Phase {phase_asset} failed OOT criteria; safely falling back to trusted base model."
            else:
                resolution = "ABSTAIN"
                rec_id = None
                rec_ver = None
                pnl = 0.0
                median_pnl = 0.0
                note = f"Neither phase {phase_asset} nor base {base_asset} passed OOT criteria; trade abstained (NONE)."

            recommendations[base_asset]["phases"][phase] = {
                "runtime_resolution": resolution,
                "recommended_model_id": rec_id,
                "version": rec_ver,
                "combined_pnl": pnl,
                "median_window_pnl": median_pnl,
                "note": note,
            }

    summary_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "summary": "LogReg candidate audit and runtime selection recommendations",
        "total_assets_covered": len(COMBINED_MODE_SUPPORTED_ASSETS),
        "asset_recommendations": recommendations,
    }

    out_file = output_dir / f"logreg_selection_{date_str}.json"
    out_file.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Selection summary saved to {out_file}")
    return summary_payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LogReg selection and comparison reports")
    parser.add_argument("--audit-json", type=Path, help="Path to existing audit JSON")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Output directory")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.audit_json and args.audit_json.exists():
        audit_data = json.loads(args.audit_json.read_text(encoding="utf-8"))
    else:
        audit_data = asyncio.run(audit_models(all_active=False))

    asyncio.run(generate_selection_summary(audit_data, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
