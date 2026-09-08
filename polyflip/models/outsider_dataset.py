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


@dataclass(frozen=True)
class CandidateQuote:
    is_valid: bool
    executable_ask: float | None
    rejection_reason: str | None = None


def resolve_candidate_quote(
    candidate_side: str,
    yes_best_ask: float | None = None,
    down_best_ask: float | None = None,
    outsider_mid: float | None = None,
    best_ask: float | None = None,
) -> CandidateQuote:
    """
    R3: Resolves executable quote for the candidate outsider side.
    DOWN side strictly requires a valid DOWN ask and must never use YES ask.
    Missing asks return is_valid=False with rejection_reason='MISSING_CANDIDATE_QUOTE'.
    """
    side = str(candidate_side).strip().upper()
    ask = None
    if side in ("DOWN", "NO"):
        if down_best_ask is not None and pd.notna(down_best_ask):
            ask = float(down_best_ask)
    elif side in ("UP", "YES"):
        if yes_best_ask is not None and pd.notna(yes_best_ask):
            ask = float(yes_best_ask)
        elif best_ask is not None and pd.notna(best_ask):
            ask = float(best_ask)

    if ask is None or not np.isfinite(ask) or ask <= 0.0 or ask >= 1.0:
        return CandidateQuote(is_valid=False, executable_ask=np.nan, rejection_reason="MISSING_CANDIDATE_QUOTE")
    return CandidateQuote(is_valid=True, executable_ask=ask, rejection_reason=None)


def candidate_target(
    candidate_side: str,
    final_outcome: str | None = None,
    target: Any = None,
) -> float:
    """
    R3: Returns 1.0 (win), 0.0 (loss), or np.nan (unresolved / pending).
    PENDING/UNRESOLVED markets must never be treated as target=0.
    """
    outcome = str(final_outcome).strip().upper() if final_outcome is not None else ""
    if outcome in ("PENDING", "UNRESOLVED", "OPEN", "UNKNOWN", "NONE", ""):
        if target is not None and pd.notna(target):
            try:
                t_val = float(target)
                if np.isfinite(t_val) and t_val in (0.0, 1.0):
                    return t_val
            except (ValueError, TypeError):
                pass
        return np.nan

    side = str(candidate_side).strip().upper()
    if outcome in ("YES", "UP"):
        return 1.0 if side in ("UP", "YES") else 0.0
    elif outcome in ("NO", "DOWN"):
        return 1.0 if side in ("DOWN", "NO") else 0.0

    if target is not None and pd.notna(target):
        try:
            return float(target)
        except (ValueError, TypeError):
            pass
    return np.nan


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

    # Exclude empty or null market_ids
    df = df[df["market_id"].notna() & (df["market_id"].astype(str).str.strip() != "")].copy()
    if df.empty:
        return pd.DataFrame()

    # Ensure recorded_at is datetime
    if "recorded_at" in df.columns:
        df["_dt"] = pd.to_datetime(df["recorded_at"], utc=True)
    else:
        df["_dt"] = pd.date_range("2026-01-01", periods=len(df), freq="1min", tz="UTC")

    # Exclude mid_price == 0.5 where outsider is ambiguous (Item 1.7)
    mid_col = "mid_price" if "mid_price" in df.columns else "poly_up_mid"
    df["_mid"] = pd.to_numeric(df[mid_col], errors="coerce")
    df = df[df["_mid"].notna() & ((df["_mid"] - 0.5).abs() > 1e-6) & (df["_mid"] > 0.0) & (df["_mid"] < 1.0)].copy()

    selected_rows = []

    # For each market and each target decision time (e.g. 10.0, 5.0, 2.0)
    # Item 1.05: Enforce backward as-of selection (snapshots recorded at or prior to scheduled decision)
    for m_id, grp in df.groupby("market_id", sort=False):
        for target_tl in target_time_points:
            # Backward as-of: snapshot recorded prior to or at decision cutoff (time_left_min >= target_tl)
            sub = grp[(grp["time_left_min"] >= target_tl) & (grp["time_left_min"] <= target_tl + tolerance_min)]
            if sub.empty:
                # Fallback to closest within tolerance if no strictly backward snapshot
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
                quote_res = resolve_candidate_quote(
                    candidate_side=candidate_side,
                    down_best_ask=row.get("poly_down_best_ask", row.get("down_best_ask")),
                    outsider_mid=outsider_mid,
                )
            else:
                candidate_side = "UP"
                outsider_mid = mid
                quote_res = resolve_candidate_quote(
                    candidate_side=candidate_side,
                    yes_best_ask=row.get("poly_up_best_ask", row.get("up_best_ask")),
                    best_ask=row.get("best_ask"),
                    outsider_mid=outsider_mid,
                )

            executable_ask = quote_res.executable_ask
            quote_valid = quote_res.is_valid
            spread = float(row.get("spread", 0.02))

            # Target: 1 if candidate side wins, 0 if loses, np.nan if unresolved
            y_win = candidate_target(
                candidate_side=candidate_side,
                final_outcome=row.get("final_outcome"),
                target=row.get("target"),
            )

            # Item 1.03: Contract of one research row with preserved yes_mid and deterministic row_id
            row_id = f"{m_id}_{float(target_tl):.1f}_{candidate_side}"
            row_out = {
                "row_id": row_id,
                "market_id": m_id,
                "decision_at": row["_dt"],
                "candidate_side": candidate_side,
                "yes_mid": mid,  # Canonical YES mid price preserved (Item 1.03)
                "canonical_yes_mid": mid,
                "y_candidate_win": y_win,
                "target": y_win,
                "outsider_mid": outsider_mid,
                "mid_price": outsider_mid,
                "executable_ask": executable_ask,
                "quote_valid": quote_valid,
                "spread": spread,
                "fee_rate": fee_rate,
                "time_left_min": float(row["time_left_min"]),
                "target_time_point": float(target_tl),
                "strike_value": row.get("strike_value", row.get("strike_price", np.nan)),
                "underlying_price": row.get("underlying_price", row.get("binance_spot_mid", np.nan)),
                "underlying_lag_30s": row.get("underlying_lag_30s", np.nan),
                "underlying_lag_120s": row.get("underlying_lag_120s", np.nan),
                "sigma_1m": row.get("sigma_1m", np.nan),
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

    # Item 1.15: Chronological market walk-forward folds without future leakage
    from polyflip.models.temporal_validation import grouped_walk_forward_folds

    unique_markets = df_feat["market_id"].unique()
    n_splits_adj = min(n_splits, len(unique_markets))

    df_feat["fold"] = 0
    ts_col = "decision_at" if "decision_at" in df_feat.columns else ("recorded_at" if "recorded_at" in df_feat.columns else None)
    if ts_col is not None and len(unique_markets) >= 3 and n_splits_adj >= 2:
        try:
            gwf_folds = grouped_walk_forward_folds(df_feat["market_id"], df_feat[ts_col], n_splits=n_splits_adj)
            if gwf_folds:
                df_feat["fold"] = -1  # Early warmup / training-only block
                for fold_idx, gf in enumerate(gwf_folds):
                    df_feat.loc[gf.validation_index, "fold"] = fold_idx
        except Exception:
            pass

    if df_feat["fold"].nunique() <= 1 and n_splits_adj >= 2:
        # Chronological market grouping without shuffling
        sorted_mkts = list(df_feat.sort_values(ts_col if ts_col else "market_id")["market_id"].unique())
        chunk = max(1, len(sorted_mkts) // n_splits_adj)
        for f_idx in range(n_splits_adj):
            m_set = set(sorted_mkts[f_idx * chunk : (f_idx + 1) * chunk if f_idx < n_splits_adj - 1 else len(sorted_mkts)])
            df_feat.loc[df_feat["market_id"].isin(m_set), "fold"] = f_idx

    # Model A cohort: rows where Model A features are non-null and valid
    a_valid = df_feat[list(MODEL_A1_FEATURES)].notna().all(axis=1)
    df_broad_a = df_feat[a_valid].copy().reset_index(drop=True)

    # Model B complete cohort: rows where Model B features AND reference flags are strictly valid
    has_comp_sig = df_feat["has_computed_sigma"] if "has_computed_sigma" in df_feat.columns else pd.Series(False, index=df_feat.index)
    b_valid = (
        a_valid
        & df_feat[list(MODEL_B1_FEATURES)].notna().all(axis=1)
        & df_feat["has_z_ref"]
        & df_feat["has_ret_30s_ref"]
        & df_feat["has_ret_120s_ref"]
        & has_comp_sig
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
