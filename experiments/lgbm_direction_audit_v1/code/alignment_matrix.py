"""E3 target alignment (recorded directions only, no replay).
Legacy direction vs contract outcome by TTE buckets + UP/DOWN split.
Output: TARGET_ALIGNMENT_MATRIX.csv
"""
import os
import numpy as np
import pandas as pd

ROOT = r"D:\lgbm-audit-v1"
DS = os.path.join(ROOT, "out", "shared_ds")
OUT = os.path.join(ROOT, "out", "direction")
os.makedirs(OUT, exist_ok=True)

BUCKETS = [(0, 3), (3, 5), (5, 8), (8, 12), (12, 1e9)]
BLABEL = ["0-3", "3-5", "5-8", "8-12", "12+"]

def main():
    df = pd.read_parquet(os.path.join(DS, "shared_dataset.parquet"),
                         columns=["model_slot", "slot_version", "slot_value",
                                  "slot_probability", "contract_target",
                                  "legacy_native", "time_to_expiry_s", "fold"])
    df["tte_min"] = df["time_to_expiry_s"] / 60.0
    df["bucket"] = pd.cut(df["tte_min"], bins=[0, 3, 5, 8, 12, 1e9],
                          labels=BLABEL, right=False)
    df["ct"] = pd.to_numeric(df["contract_target"], errors="coerce")
    df["ln"] = pd.to_numeric(df["legacy_native"], errors="coerce")
    rows = []
    sub = df[df["slot_value"].isin(["UP", "DOWN"]) & df["ct"].isin([0.0, 1.0])].copy()
    # contract YES=1 means UP wins (YES/UP mapping per audit scope)
    sub["dir_correct"] = ((sub["slot_value"] == "UP") & (sub["ct"] == 1.0)) | \
                         ((sub["slot_value"] == "DOWN") & (sub["ct"] == 0.0))
    for b in BLABEL:
        g = sub[sub["bucket"] == b]
        for side in ["UP", "DOWN"]:
            gs = g[g["slot_value"] == side]
            rows.append({"tte_bucket": b, "side": side, "n": int(len(gs)),
                         "contract_acc": float(gs["dir_correct"].mean()) if len(gs) else float("nan")})
        rows.append({"tte_bucket": b, "side": "ALL", "n": int(len(g)),
                     "contract_acc": float(g["dir_correct"].mean()) if len(g) else float("nan")})
    # native (next-candle) accuracy by bucket for reference
    for b in BLABEL:
        g = df[(df["bucket"] == b) & df["ln"].isin([0.0, 1.0])].copy()
        if len(g):
            # native label vs own direction implied: use slot_value where present
            rows.append({"tte_bucket": b, "side": "NATIVE_COVERAGE",
                         "n": int(len(g)), "contract_acc": float("nan")})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "TARGET_ALIGNMENT_MATRIX.csv"), index=False)
    print(out.to_string())
    print("wrote TARGET_ALIGNMENT_MATRIX.csv")

if __name__ == "__main__":
    main()
