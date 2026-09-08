"""
polyflip/models/outsider_dataset.py

Unified decision-level dataset builder for Outsider Models A and B (Item 2.4).
Guarantees:
1. Unique composite key: (market_id, decision_at, candidate_side).
2. Fixed intramarket decision points: T-10, T-5, T-2 minutes prior to expiration.
3. Market-level grouping: all decision points for a given market_id belong to the same validation fold.
4. Identical cohort comparison: Model A and Model B are evaluated on the exact same complete-B cohort.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Sequence, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from polyflip.models.outsider_feature_sets import (
    MODEL_A1_FEATURES,
    MODEL_B1_FEATURES,
    get_outsider_feature_set,
)
from polyflip.models.point_in_time_features import compute_outsider_model_features

DECISION_TIME_POINTS_MIN: tuple[float, ...] = (10.0, 5.0, 2.0)
DECISION_TIME_TOLERANCE_MIN: float = 0.75


@dataclass
class OutsiderCohortData:
    df_full_b: pd.DataFrame
    df_broad_a: pd.DataFrame
    features_a: tuple[str, ...] = MODEL_A1_FEATURES
    features_b: tuple[str, ...] = MODEL_B1_FEATURES
    metadata: dict[str, Any] = field(default_factory=dict)


def build_outsider_decision_rows(
    snapshots: pd.DataFrame,
    target_time_points: Sequence[float] = DECISION_TIME_POINTS_MIN,
    tolerance_min: float = DECISION_TIME_TOLERANCE_MIN,
    fee_rate: float = 0.002,
) -> pd.DataFrame:
    """
    Filters snapshots to fixed decision points T-10, T-5, T-2 per market,
    determines candidate outsider side, target, and executable ask.
    Ensures (market_id, decision_at, candidate_side) is strictly unique.
    """
    if snapshots.empty:
        return pd.DataFrame()

    df = snapshots.copy()
    if "market_id" not in df.columns or "time_left_min" not in df.columns:
        raise ValueError("snapshots must contain 'market_id' and 'time_left_min'")

    # Ensure recorded_at is datetime
    if "recorded_at" in df.columns:
        df["_dt"] = pd.to_datetime(df["recorded_at"], utc=True)
    else:
        df["_dt"] = pd.date_range("2026-01-01", periods=len(df), freq="1min", tz="UTC")

    # Exclude mid_price == 0.5 where outsider is ambiguous (Item 1.7)
    mid_col = "mid_price" if "mid_price" in df.columns else "poly_up_mid"
    df["_mid"] = pd.to_numeric(df[mid_col], errors="coerce")
    df = df[df["_mid"].notna() & (df["_mid"] != 0.5) & (df["_mid"] > 0.0) & (df["_mid"] < 1.0)].copy()

    selected_rows = []

    # For each market and each target decision time (e.g. 10.0, 5.0, 2.0), pick the closest snapshot
    for m_id, grp in df.groupby("market_id", sort=False):
        for target_tl in target_time_points:
            sub = grp[np.abs(grp["time_left_min"] - target_tl) <= tolerance_min]
            if sub.empty:
                continue

            # Pick the row closest to target_tl
            closest_idx = np.abs(sub["time_left_min"] - target_tl).idxmin()
            row = grp.loc[closest_idx].to_dict()

            mid = float(row["_mid"])
            # Outsider is the minority side (< 0.5)
            # If YES mid > 0.5 -> favourite is YES, outsider candidate is NO (DOWN)
            # If YES mid < 0.5 -> favourite is NO, outsider candidate is YES (UP)
            if mid > 0.5:
                candidate_side = "DOWN"
                outsider_mid = 1.0 - mid
                # Executable ask for buying NO
                best_ask = row.get("poly_down_best_ask", row.get("best_ask", outsider_mid + 0.01))
            else:
                candidate_side = "UP"
                outsider_mid = mid
                best_ask = row.get("poly_up_best_ask", row.get("best_ask", outsider_mid + 0.01))

            executable_ask = float(np.clip(best_ask if pd.notna(best_ask) else outsider_mid + 0.01, 0.01, 0.99))
            spread = float(row.get("spread", 0.02))

            # Target: 1 if candidate side wins, 0 if loses
            final_outcome = str(row.get("final_outcome", "")).strip().upper()
            if final_outcome in ("YES", "UP"):
                y_win = 1 if candidate_side in ("UP", "YES") else 0
            elif final_outcome in ("NO", "DOWN"):
                y_win = 1 if candidate_side in ("DOWN", "NO") else 0
            else:
                y_win = int(row.get("target", 0))

            row_out = {
                "market_id": m_id,
                "decision_at": row["_dt"],
                "candidate_side": candidate_side,
                "y_candidate_win": y_win,
                "target": y_win,
                "outsider_mid": outsider_mid,
                "mid_price": outsider_mid,
                "executable_ask": executable_ask,
                "spread": spread,
                "fee_rate": fee_rate,
                "time_left_min": float(row["time_left_min"]),
                "target_time_point": float(target_tl),
                "strike_value": row.get("strike_value", row.get("strike_price", np.nan)),
                "underlying_price": row.get("underlying_price", row.get("binance_spot_mid", np.nan)),
                "underlying_lag_30s": row.get("underlying_lag_30s", np.nan),
                "underlying_lag_120s": row.get("underlying_lag_120s", np.nan),
                "sigma_1m": row.get("sigma_1m", 0.001),
                "strike_source": row.get("strike_source", "UNKNOWN"),
                "strike_effective_at": row.get("strike_effective_at", None),
            }
            selected_rows.append(row_out)

    res_df = pd.DataFrame(selected_rows)
    if res_df.empty:
        return res_df

    # Enforce uniqueness of (market_id, decision_at, candidate_side)
    res_df = res_df.drop_duplicates(subset=["market_id", "decision_at", "candidate_side"]).reset_index(drop=True)
    return res_df


def prepare_outsider_dataset_cohorts(
    snapshots: pd.DataFrame,
    target_time_points: Sequence[float] = DECISION_TIME_POINTS_MIN,
    n_splits: int = 5,
) -> OutsiderCohortData:
    """
    Builds and computes features for Model A and Model B, creates market-level folds,
    and partitions data into:
      - df_full_b: complete cohort where all Model B features are strictly valid
      - df_broad_a: all valid decision rows where Model A features are valid
    """
    base_df = build_outsider_decision_rows(snapshots, target_time_points=target_time_points)
    if base_df.empty:
        return OutsiderCohortData(df_full_b=pd.DataFrame(), df_broad_a=pd.DataFrame())

    # Compute Model A1 and Model B1 features
    df_feat = compute_outsider_model_features(base_df)

    # Assign market-grouped fold indices (GroupKFold by market_id)
    unique_markets = df_feat["market_id"].unique()
    n_splits_adj = min(n_splits, len(unique_markets))

    df_feat["fold"] = 0
    if n_splits_adj >= 2:
        gkf = GroupKFold(n_splits=n_splits_adj)
        for fold_idx, (_, test_indices) in enumerate(gkf.split(df_feat, groups=df_feat["market_id"])):
            df_feat.loc[test_indices, "fold"] = fold_idx

    # Model A cohort: rows where Model A features are non-null and valid
    a_valid = df_feat[list(MODEL_A1_FEATURES)].notna().all(axis=1)
    df_broad_a = df_feat[a_valid].copy().reset_index(drop=True)

    # Model B complete cohort: rows where Model B features AND reference flags are strictly valid
    b_valid = (
        a_valid
        & df_feat[list(MODEL_B1_FEATURES)].notna().all(axis=1)
        & df_feat["has_z_ref"]
        & df_feat["has_ret_30s_ref"]
        & df_feat["has_ret_120s_ref"]
    )
    df_full_b = df_feat[b_valid].copy().reset_index(drop=True)

    metadata = {
        "total_decision_rows": len(df_feat),
        "total_markets": int(df_feat["market_id"].nunique()),
        "broad_a_rows": len(df_broad_a),
        "broad_a_markets": int(df_broad_a["market_id"].nunique()),
        "full_b_rows": len(df_full_b),
        "full_b_markets": int(df_full_b["market_id"].nunique()),
        "n_splits": n_splits_adj,
    }

    return OutsiderCohortData(
        df_full_b=df_full_b,
        df_broad_a=df_broad_a,
        metadata=metadata,
    )
