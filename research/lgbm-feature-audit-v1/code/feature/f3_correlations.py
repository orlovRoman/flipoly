"""f3_correlations.py — F3 feature correlations per train fold (spec
lgbm_feature_audit_v1.yaml correlations_F3).

Per eval fold (F2..F6), scope = train rows of that fold (rule 'scope:
train_part_each_fold'). For every feature pair (79 x 79 / 2):
  - pearson correlation over the COMMON mask of the pair (pairwise-complete,
    means/std computed on the pair's shared observed rows only)
  - spearman correlation: ranks recomputed on the pair's common observed rows
  - missing-mask Jaccard  J = |A & B| / |A | B| on isna masks
      (both columns fully observed -> 1.0; one fully observed -> 0.0)
  - near_duplicate = abs(spearman) >= 0.98 AND jaccard_missing >= 0.99

Memory strategy (machine has ~3-4 GiB free): the cumulative train scope per
fold (up to ~1.3M rows x 79) is streamed fold-by-fold into disk-backed
np.memmap arrays (float32 values + uint8 NaN masks); per-pair stats touch only
two columns at a time. Jaccard is computed in row-chunks.

Aggregated per pair across the 5 folds:
  - n_folds_with_pair (folds where the pair could be computed)
  - mean_pearson, max_abs_spearman, mean_jaccard_missing
  - n_folds_near_dup, near_dup_ge4  (group_verdict_if_both_in_ge_folds: 4)

correlated_also_as_group: near-dup summary is reported both within the same
catalog group and across groups (same-group rows marked same_group=1).

Outputs:
  out/feature/FEATURE_CORRELATIONS_FOLD.parquet   (fold-level long table)
  out/feature/FEATURE_CORRELATIONS.parquet        (pair-level aggregate)
"""
import os, gc, tempfile
from collections import Counter
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FDIR = os.path.join(ROOT, "out", "feature")
MATRIX = os.path.join(FDIR, "FEATURE_MATRIX.parquet")
CATALOG_CSV = os.path.join(FDIR, "FEATURE_CATALOG.csv")
TMP_DIR = r"C:\Users\orlov\AppData\Local\Temp\opencode"

FOLD_CHAIN = ["trainpool", "F1", "F2", "F3", "F4", "F5", "F6"]
EVAL_FOLDS = ["F2", "F3", "F4", "F5", "F6"]

NEAR_DUP_SPEARMAN = 0.98
NEAR_DUP_JACCARD = 0.99
GE_FOLDS = 4
CHUNK = 30000


def _r(x, y):
    """Pearson r computed entirely on the given (already common-mask) rows."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    sx, sy = float(x.std()), float(y.std())
    if sx <= 0.0 or sy <= 0.0:
        return np.nan
    return float((x * y).mean() / (sx * sy))


def jaccard_missing_chunked(M, n0, p):
    """Jaccard on isna masks (uint8 memmap) computed in row-chunks."""
    inter = np.zeros((p, p), dtype=np.float64)
    rowsum = np.zeros(p, dtype=np.float64)
    for s in range(0, n0, CHUNK):
        mm = M[s:s + CHUNK].astype(np.float64)
        inter += mm.T @ mm
        rowsum += mm.sum(axis=0)
        del mm
    union = rowsum[None, :] + rowsum[:, None] - inter
    J = np.full((p, p), np.nan)
    mask_valid = union > 0
    J[mask_valid] = inter[mask_valid] / union[mask_valid]
    J[union == 0] = 1.0
    np.fill_diagonal(J, 1.0)
    return J


def main():
    feats = pd.read_csv(CATALOG_CSV)["feature"].tolist()
    cat = pd.read_csv(CATALOG_CSV, dtype=str)
    gmap = dict(zip(cat["feature"], cat["group"]))
    n = len(feats)

    print("counting folds...", flush=True)
    ctr = Counter(pq.read_table(MATRIX, columns=["fold"]).column("fold").to_pylist())

    rows = []
    for fold in EVAL_FOLDS:
        tr_folds = FOLD_CHAIN[: FOLD_CHAIN.index(fold)]
        n0 = int(sum(ctr[f] for f in tr_folds))
        x_path = os.path.join(TMP_DIR, f"f3_X_{fold}.dat")
        m_path = os.path.join(TMP_DIR, f"f3_M_{fold}.dat")
        with open(x_path, "wb") as fh:
            fh.truncate(n0 * n * np.dtype(np.float32).itemsize)
        with open(m_path, "wb") as fh:
            fh.truncate(n0 * n * np.dtype(np.uint8).itemsize)
        X = np.memmap(x_path, dtype=np.float32, mode="r+", shape=(n0, n))
        M = np.memmap(m_path, dtype=np.uint8, mode="r+", shape=(n0, n))
        try:
            row = 0
            for tr in tr_folds:
                tb = pq.read_table(MATRIX, columns=feats,
                                   filters=[("fold", "==", tr)])
                rn = tb.num_rows
                block = np.empty((rn, n), dtype=np.float32)
                for j in range(n):
                    block[:, j] = tb.column(j).to_numpy(zero_copy_only=False)
                del tb
                X[row:row + rn] = block
                M[row:row + rn] = np.isnan(block)
                del block
                row += rn
            gc.collect()
            print(f"fold {fold}: train rows {n0}", flush=True)

            jac = jaccard_missing_chunked(M, n0, n)
            for a in range(n):
                xa_all = X[:, a]
                ma = M[:, a].astype(bool)
                for b in range(a + 1, n):
                    ok = ~(ma | M[:, b].astype(bool))
                    cnt = int(ok.sum())
                    if cnt < 2:
                        continue
                    xa = xa_all[ok]
                    xb = X[ok, b]
                    rp = _r(xa, xb)
                    rs = _r(rankdata(xa), rankdata(xb))
                    rows.append({
                        "fold": fold,
                        "feature_a": feats[a], "feature_b": feats[b],
                        "group_a": gmap.get(feats[a], ""),
                        "group_b": gmap.get(feats[b], ""),
                        "n_obs": cnt,
                        "pearson": float(rp),
                        "spearman": float(rs),
                        "jaccard_missing": float(jac[a, b]),
                        "near_dup": bool(np.isfinite(rs)
                                         and abs(rs) >= NEAR_DUP_SPEARMAN
                                         and jac[a, b] >= NEAR_DUP_JACCARD),
                    })
            print(f"  fold {fold}: done", flush=True)
        finally:
            X.flush()
            del X, M
            gc.collect()
            for pth in (x_path, m_path):
                try:
                    os.remove(pth)
                except OSError:
                    pass

    fold_df = pd.DataFrame(rows)
    fold_df.to_parquet(os.path.join(FDIR, "FEATURE_CORRELATIONS_FOLD.parquet"),
                       index=False)
    print("wrote FEATURE_CORRELATIONS_FOLD.parquet", len(fold_df))

    # aggregate per pair
    agg = fold_df.groupby(["feature_a", "feature_b", "group_a", "group_b"],
                          as_index=False).agg(
        n_folds=("pearson", "count"),
        mean_pearson=("pearson", "mean"),
        max_abs_spearman=("spearman", lambda s: float(np.abs(s).max())),
        mean_jaccard_missing=("jaccard_missing", "mean"),
        n_folds_near_dup=("near_dup", "sum"),
    )
    agg["same_group"] = (agg["group_a"] == agg["group_b"]).astype(int)
    agg["near_dup_ge4"] = (agg["n_folds_near_dup"] >= GE_FOLDS).astype(int)
    near = agg[agg["n_folds_near_dup"] > 0].sort_values(
        ["n_folds_near_dup", "max_abs_spearman"], ascending=False)
    print(f"\nnear-dup pairs detected: {len(near)} "
          f"(>=1 fold); stable >= {GE_FOLDS} folds: {(agg['near_dup_ge4'] == 1).sum()}")
    for _, r in near.head(20).iterrows():
        print(f"  {r['feature_a']} ~ {r['feature_b']} "
              f"n_folds_nd={int(r['n_folds_near_dup'])} spanomax={r['max_abs_spearman']:.3f} "
              f"jac={r['mean_jaccard_missing']:.3f} grp={r['group_a']}/{r['group_b']}")
    agg.to_parquet(os.path.join(FDIR, "FEATURE_CORRELATIONS.parquet"), index=False)
    print("wrote FEATURE_CORRELATIONS.parquet", len(agg))
    agg.to_csv(os.path.join(FDIR, "FEATURE_CORRELATIONS_AGGREGATE.csv"), index=False)


if __name__ == "__main__":
    main()