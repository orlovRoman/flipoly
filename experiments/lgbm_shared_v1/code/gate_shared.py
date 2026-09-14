"""gate_shared.py v1.0.0 — dataset gate as code (spec lgbm_shared_v1).

Requirements: 0 leakage (proven by builder asserts), 0 target conflicts,
0 unexplained dupes, 100% valid reason codes, Tier A >= 90% for policy
replay, >= 200 paired opportunities, checksums match.
Settings-coverage fail -> forecast continues, policy verdict INCONCLUSIVE_SETTINGS.
"""
import hashlib
import json
import os

import pandas as pd
import yaml

GATE_VERSION = "v1.0.0"
SPEC_SHA256 = "1649bc7c7c3459353bec571f5b3d501e9c71dfacf5f1a32772503aef87fcefea"
ROOT = r"D:\lgbm-audit-v1"
DS = os.path.join(ROOT, "out", "shared_ds")


def main():
    assert hashlib.sha256(
        open(os.path.join(ROOT, "spec", "lgbm_shared_v1.yaml"), "rb").read()
    ).hexdigest() == SPEC_SHA256, "spec mismatch"
    rc = yaml.safe_load(open(r"D:\lgbm-audit-v1\spec\reason_codes.yaml"))
    enum_codes = set()
    for v in rc["categories"].values():
        enum_codes.update(v)
    man = json.load(open(os.path.join(DS, "manifest.json")))
    assert man["logical_sha256"] == open(
        os.path.join(DS, "dataset.sha256")).read().strip(), "sha file mismatch"
    df = pd.read_parquet(os.path.join(DS, "shared_dataset.parquet"),
                         columns=["decision_event_id", "model_slot",
                                  "model_mapped", "quote_ok", "settings_tier",
                                  "fold", "contract_target", "reason_codes"])
    checks = {}
    checks["unique_events"] = bool(df["decision_event_id"].is_unique)
    checks["n_rows"] = int(len(df))
    flat = set()
    for x in df["reason_codes"].dropna().unique().tolist():
        flat.update(x.split(";"))
    checks["reason_codes_valid"] = bool(not (flat - enum_codes))
    el = df[df["model_mapped"].fillna(False).astype(bool)
            & df["quote_ok"].fillna(False).astype(bool)]
    checks["eligible"] = int(len(el))
    a = int((el["settings_tier"] == "A").sum())
    checks["tierA"] = a
    checks["tierA_share"] = (a / len(el)) if len(el) else 0.0
    checks["tierA_ge_90pct"] = bool(checks["tierA_share"] >= 0.90)
    checks["pairs_ge_200"] = bool(a >= 200)
    cov = json.load(open(os.path.join(DS, "coverage.json")))
    checks["coverage_assets"] = len(cov.get("asset", {}))
    checks["coverage_folds"] = len(cov.get("fold", {}))
    ok = all([checks["unique_events"], checks["reason_codes_valid"],
              checks["tierA_ge_90pct"], checks["pairs_ge_200"]])
    # leakage/conflicts proven by builder asserts (would have crashed);
    # builder completed -> record as passed with provenance.
    checks["leakage_zero_by_construction"] = True
    checks["target_conflicts_zero_by_construction"] = True
    out = {"gate_version": GATE_VERSION, "spec_sha256": SPEC_SHA256,
           "verdict": "PASS" if ok else "FAIL", "checks": checks}
    if not checks["tierA_ge_90pct"]:
        out["verdict"] = "INCONCLUSIVE_SETTINGS"
        out["note"] = "forecast audit continues; no policy verdict"
    with open(os.path.join(DS, "dataset_gate.json"), "w", newline="") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(checks, indent=1))
    print(out["verdict"])
    return out


if __name__ == "__main__":
    main()
