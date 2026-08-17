"""Stage 6 & 7: Systematic LogReg candidate trainer.

Trains a grid of LogReg candidate models across feature variants,
regularization strengths, class balance, sample weighting, and calibrations.
Evaluates OOF metrics and 3 chronological OOT windows.
Saves all candidates to model_registry strictly as is_active=False.
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sqlalchemy import func, select

from polyflip.constants import COMBINED_MODE_SUPPORTED_ASSETS, PRICE_PHASE_BOUNDARIES
from polyflip.crypto.logreg_polymarket_backtest import compute_logreg_polymarket_backtest
from polyflip.crypto.oof_artifact import OOF_ARTIFACT_SCHEMA_VERSION, serialize_oof_artifact
from polyflip.db.connection import async_session
from polyflip.db.models import MarketSnapshot, ModelRegistry, ModelRegistryOOFArtifact
from polyflip.models.feature_lags import LAG_FEATURE_NAMES, add_lag_features
from polyflip.models.sequence_features import SEQUENCE_CANDLE_FEATURES, attach_closed_candle_features
from polyflip.models.temporal_validation import grouped_walk_forward_folds
from polyflip.models.trainer import DERIVED_FEATURES, add_derived_features


def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    valid = np.isfinite(y_true) & np.isfinite(y_prob)
    y_t = y_true[valid].astype(int)
    y_p = y_prob[valid]
    if len(y_t) == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_p, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    total = len(y_t)
    for b in range(n_bins):
        mask = bin_indices == b
        if np.any(mask):
            bin_acc = np.mean(y_t[mask])
            bin_conf = np.mean(y_p[mask])
            ece += (np.sum(mask) / total) * abs(bin_acc - bin_conf)
    return float(ece)


def _compute_drawdown_series(trade_pnls: Sequence[float]) -> tuple[float, float]:
    """Return max drawdown (USDC) from trade PnL sequence."""
    if not trade_pnls:
        return 0.0, 0.0
    cumulative = np.cumsum(trade_pnls)
    running_max = np.maximum.accumulate(np.insert(cumulative, 0, 0.0))
    drawdowns = running_max[1:] - cumulative
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
    return max_dd, 0.0


def _split_chronological_windows(
    frame: pd.DataFrame,
    p_yes: np.ndarray,
    quotes: pd.DataFrame | None,
    strategy_branch: str,
    **backtest_kwargs: Any,
) -> dict[str, Any]:
    """Split canonical evaluated markets into 3 chronological windows T1/T2/T3 by market_close_at."""
    if frame.empty or len(p_yes) != len(frame):
        return {
            "T1": {"status": "EMPTY", "n_markets": 0, "n_trades": 0, "total_pnl": 0.0},
            "T2": {"status": "EMPTY", "n_markets": 0, "n_trades": 0, "total_pnl": 0.0},
            "T3": {"status": "EMPTY", "n_markets": 0, "n_trades": 0, "total_pnl": 0.0},
            "median_pnl": 0.0,
            "non_negative_windows_count": 0,
        }

    working = frame.copy()
    close_col = None
    for candidate in ("market_close_at", "resolved_at", "end_time_est", "market_start", "recorded_at"):
        if candidate in working.columns and working[candidate].notna().any():
            close_col = candidate
            break

    if close_col:
        working["_close_time"] = pd.to_datetime(working[close_col], utc=True, errors="coerce")
    else:
        working["_close_time"] = pd.to_datetime(working["recorded_at"], utc=True, errors="coerce")

    working["_p_yes"] = p_yes
    working = working.sort_values("_close_time", kind="stable").reset_index(drop=True)

    n = len(working)
    if n < 3:
        res = compute_logreg_polymarket_backtest(
            working, working["_p_yes"].to_numpy(), quotes, strategy_branch=strategy_branch, **backtest_kwargs
        )
        return {
            "T1": {"status": "SPARSE", "n_markets": n, "n_trades": res.get("n_trades", 0), "total_pnl": res.get("total_pnl", 0.0)},
            "T2": {"status": "SPARSE", "n_markets": 0, "n_trades": 0, "total_pnl": 0.0},
            "T3": {"status": "SPARSE", "n_markets": 0, "n_trades": 0, "total_pnl": 0.0},
            "median_pnl": 0.0,
            "non_negative_windows_count": 1 if res.get("total_pnl", 0.0) >= 0 else 0,
        }

    q33 = working["_close_time"].quantile(1.0 / 3.0)
    q67 = working["_close_time"].quantile(2.0 / 3.0)

    w1_mask = working["_close_time"] <= q33
    w2_mask = (working["_close_time"] > q33) & (working["_close_time"] <= q67)
    w3_mask = working["_close_time"] > q67

    if not w1_mask.any() or not w2_mask.any() or not w3_mask.any():
        idx1 = n // 3
        idx2 = 2 * n // 3
        w1_mask = np.zeros(n, dtype=bool)
        w2_mask = np.zeros(n, dtype=bool)
        w3_mask = np.zeros(n, dtype=bool)
        w1_mask[:idx1] = True
        w2_mask[idx1:idx2] = True
        w3_mask[idx2:] = True

    windows = {}
    pnls = []
    for label, mask in (("T1", w1_mask), ("T2", w2_mask), ("T3", w3_mask)):
        sub = working[mask].reset_index(drop=True)
        sub_n = len(sub)
        if sub_n == 0:
            windows[label] = {
                "status": "EMPTY",
                "n_markets": 0,
                "n_trades": 0,
                "total_pnl": 0.0,
                "win_rate": 0.0,
                "max_drawdown": 0.0,
            }
            pnls.append(0.0)
            continue

        sub_quotes = (
            quotes[quotes["market_id"].isin(sub["market_id"])].reset_index(drop=True)
            if quotes is not None and not quotes.empty and "market_id" in quotes.columns
            else None
        )
        res = compute_logreg_polymarket_backtest(
            sub, sub["_p_yes"].to_numpy(), sub_quotes, strategy_branch=strategy_branch, **backtest_kwargs
        )
        trades = res.get("trades", [])
        trade_pnls = [t.get("pnl", 0.0) for t in trades]
        max_dd, _ = _compute_drawdown_series(trade_pnls)
        tot_pnl = float(res.get("total_pnl", 0.0))
        pnls.append(tot_pnl)
        windows[label] = {
            "status": "SPARSE" if sub_n < 30 else "OK",
            "n_markets": sub_n,
            "n_trades": int(res.get("n_trades", 0)),
            "total_pnl": round(tot_pnl, 4),
            "win_rate": round(float(res.get("win_rate", 0.0)), 4),
            "roi": round(float(res.get("roi", 0.0)), 4),
            "max_drawdown": round(max_dd, 4),
            "time_start": sub["_close_time"].min().isoformat() if not sub.empty and pd.notna(sub["_close_time"].min()) else None,
            "time_end": sub["_close_time"].max().isoformat() if not sub.empty and pd.notna(sub["_close_time"].max()) else None,
        }

    windows["median_pnl"] = round(float(np.median(pnls)), 4)
    windows["non_negative_windows_count"] = int(sum(p >= 0 for p in pnls))
    return windows

MIN_SEQUENCE_COVERAGE = 0.95


def _build_feature_matrices(
    df: pd.DataFrame,
    variant: str,
) -> tuple[pd.DataFrame, list[str], float]:
    """Extract requested feature set and evaluate sequence coverage."""
    base_cols = ["mid_price", "spread", "time_left_min"]
    coverage = 1.0

    working = df.copy()
    working = add_derived_features(working)
    working = add_lag_features(working)

    if variant == "BASE":
        raw_features = [c for c in base_cols if c in working.columns]
    elif variant == "BASE+DERIVED":
        raw_features = [c for c in (base_cols + list(DERIVED_FEATURES)) if c in working.columns]
    elif variant == "BASE+LAGS":
        raw_features = [c for c in (base_cols + list(DERIVED_FEATURES) + list(LAG_FEATURE_NAMES)) if c in working.columns]
    elif variant == "BASE+SEQUENCE":
        seq_cols = [c for c in SEQUENCE_CANDLE_FEATURES if c in working.columns]
        if seq_cols:
            non_null_rate = working[seq_cols].notna().all(axis=1).mean()
            coverage = float(non_null_rate)
        else:
            coverage = 0.0

        if coverage < MIN_SEQUENCE_COVERAGE:
            raw_features = [c for c in (base_cols + list(DERIVED_FEATURES) + list(LAG_FEATURE_NAMES)) if c in working.columns]
        else:
            raw_features = [c for c in (base_cols + list(DERIVED_FEATURES) + list(LAG_FEATURE_NAMES) + seq_cols) if c in working.columns]
    else:
        raw_features = [c for c in (base_cols + list(DERIVED_FEATURES)) if c in working.columns]

    features = list(dict.fromkeys(raw_features))
    X = working[features].fillna(0.0)
    return X, features, coverage


def _compute_sample_weights(
    time_left: np.ndarray,
    mode: str,
    tau: float = 5.0,
) -> np.ndarray | None:
    if mode == "uniform" or mode is None:
        return None
    if mode == "time_decay":
        w = 1.0 / (np.asarray(time_left, dtype=float) + 1.0)
    elif mode == "exp_decay":
        w = np.exp(-np.asarray(time_left, dtype=float) / (tau + 1e-9))
    else:
        return None
    w = w / (w.mean() + 1e-9)
    return w.astype(np.float64)


from polyflip.constants import COMBINED_MODE_SUPPORTED_ASSETS, PRICE_PHASE_BOUNDARIES, get_price_phase


async def load_asset_snapshots(
    session: Any,
    asset_name: str,
    min_time_min: float = 0.5,
    max_time_min: float = 14.5,
) -> pd.DataFrame:
    """Load and normalize historical resolved Polymarket snapshots with lightweight column projection."""
    parts = asset_name.split("_", 1)
    base_asset = parts[0].upper()
    phase = parts[1].lower() if len(parts) > 1 else None

    stmt = (
        select(
            MarketSnapshot.market_id,
            MarketSnapshot.asset,
            MarketSnapshot.recorded_at,
            MarketSnapshot.time_left_min,
            MarketSnapshot.mid_price,
            MarketSnapshot.spread,
            MarketSnapshot.best_bid,
            MarketSnapshot.best_ask,
            MarketSnapshot.volume_5min,
            MarketSnapshot.price_velocity,
            MarketSnapshot.hour_of_day,
            MarketSnapshot.flip_vs_final,
            MarketSnapshot.final_outcome,
        )
        .where(
            MarketSnapshot.asset == base_asset,
            MarketSnapshot.final_outcome.in_(["YES", "NO"]),
            MarketSnapshot.flip_vs_final.is_not(None),
            MarketSnapshot.time_left_min >= min_time_min,
            MarketSnapshot.time_left_min <= max_time_min,
        )
        .order_by(MarketSnapshot.recorded_at.asc())
    )
    res = await session.execute(stmt)
    tuples = res.all()
    if not tuples:
        return pd.DataFrame()

    records = []
    for t in tuples:
        mid_p = float(t.mid_price)
        if phase is not None:
            s_phase = get_price_phase(mid_p)
            if s_phase != phase:
                continue

        records.append({
            "market_id": t.market_id,
            "asset": t.asset,
            "recorded_at": t.recorded_at,
            "time_left_min": float(t.time_left_min),
            "mid_price": mid_p,
            "spread": float(t.spread) if t.spread is not None else 0.01,
            "best_bid": float(t.best_bid) if t.best_bid is not None else None,
            "best_ask": float(t.best_ask) if t.best_ask is not None else None,
            "volume_5min": float(t.volume_5min) if t.volume_5min is not None else 0.0,
            "price_velocity": float(t.price_velocity) if t.price_velocity is not None else 0.0,
            "hour_of_day": int(t.hour_of_day) if t.hour_of_day is not None else 0,
            "target": 1 if t.flip_vs_final else 0,
            "final_outcome": t.final_outcome,
            "yes_price": mid_p,
            "no_price": 1.0 - mid_p,
        })
    df = pd.DataFrame(records)
    if not df.empty:
        df["_decision_at"] = pd.to_datetime(df["recorded_at"], utc=True)
    return df


def train_and_eval_candidate(
    df: pd.DataFrame,
    *,
    variant: str,
    C: float,
    class_weight: str | None,
    sample_weight_mode: str,
    sample_weight_tau: float,
    calibration_method: str,
) -> dict[str, Any] | None:
    """Perform chronological grouped walk-forward training & evaluation of one candidate."""
    X, feature_names, seq_coverage = _build_feature_matrices(df, variant)
    y = df["target"].to_numpy().astype(int)
    groups = df["market_id"]
    timestamps = df["_decision_at"]
    time_left = df["time_left_min"].to_numpy()

    folds = grouped_walk_forward_folds(groups, timestamps, n_splits=5)
    if not folds:
        return None

    oof_raw_scores = np.full(len(df), np.nan, dtype=float)
    oof_cal_scores = np.full(len(df), np.nan, dtype=float)

    sample_weights = _compute_sample_weights(time_left, sample_weight_mode, sample_weight_tau)

    for fold in folds:
        X_tr = X.iloc[fold.train_index]
        y_tr = y[fold.train_index]
        w_tr = sample_weights[fold.train_index] if sample_weights is not None else None

        X_val = X.iloc[fold.validation_index]

        if len(np.unique(y_tr)) < 2:
            continue

        base_lr = LogisticRegression(
            C=C,
            class_weight=class_weight,
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )
        base_lr.fit(X_tr, y_tr, sample_weight=w_tr)

        raw_preds = base_lr.predict_proba(X_val)[:, 1]
        oof_raw_scores[fold.validation_index] = raw_preds

        # Calibration
        if calibration_method == "RAW":
            oof_cal_scores[fold.validation_index] = raw_preds
        elif calibration_method == "PLATT":
            try:
                calibrator = CalibratedClassifierCV(estimator=base_lr, method="sigmoid", cv="prefit")
                calibrator.fit(X_tr, y_tr)
                oof_cal_scores[fold.validation_index] = calibrator.predict_proba(X_val)[:, 1]
            except Exception:
                oof_cal_scores[fold.validation_index] = raw_preds
        elif calibration_method == "ISOTONIC":
            try:
                # Isotonic on training predictions
                tr_raw = base_lr.predict_proba(X_tr)[:, 1]
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(tr_raw, y_tr)
                oof_cal_scores[fold.validation_index] = iso.predict(raw_preds)
            except Exception:
                oof_cal_scores[fold.validation_index] = raw_preds

    valid_mask = np.isfinite(oof_cal_scores)
    if valid_mask.sum() < 30:
        return None

    y_val = y[valid_mask]
    cal_val = oof_cal_scores[valid_mask]
    raw_val = oof_raw_scores[valid_mask]

    try:
        val_auc = float(roc_auc_score(y_val, cal_val))
    except Exception:
        val_auc = 0.50

    brier = float(brier_score_loss(y_val, cal_val))
    ece = float(_compute_ece(y_val, cal_val))

    # Fit final full model on entire dataset
    full_weights = sample_weights if sample_weights is not None else None
    final_model = LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )
    final_model.fit(X, y, sample_weight=full_weights)

    # Polymarket canonical evaluation
    quotes = df[["market_id", "recorded_at", "mid_price", "best_bid", "best_ask", "final_outcome"]].copy()
    backtest = compute_logreg_polymarket_backtest(
        df,
        oof_cal_scores,
        quotes,
        strategy_branch="COMBINED",
        min_edge=0.03,
        fee_rate=0.02,
        cost_buffer=0.0,
    )

    # 3 Chronological Windows
    from polyflip.crypto.logreg_polymarket_backtest import _first_valid_per_market
    unique_selected, p_yes = _first_valid_per_market(df, oof_cal_scores)
    oot_windows = _split_chronological_windows(
        unique_selected,
        p_yes,
        quotes,
        strategy_branch="COMBINED",
        min_edge=0.03,
        fee_rate=0.02,
        cost_buffer=0.0,
    )

    trades = backtest.get("trades", [])
    trade_pnls = [t.get("pnl", 0.0) for t in trades]
    max_dd, _ = _compute_drawdown_series(trade_pnls)

    total_pnl = float(backtest.get("total_pnl", 0.0))
    median_win_pnl = oot_windows.get("median_pnl", -1.0)
    non_neg_windows = oot_windows.get("non_negative_windows_count", 0)
    n_trades = int(backtest.get("n_trades", 0))

    is_deployable = (
        total_pnl > 0
        and median_win_pnl > 0
        and non_neg_windows >= 2
        and n_trades >= 30
        and ece <= 0.10
    )

    rejection_reasons = []
    if total_pnl <= 0:
        rejection_reasons.append(f"COMBINED PnL <= 0 ({total_pnl:.2f})")
    if median_win_pnl <= 0:
        rejection_reasons.append(f"Median window PnL <= 0 ({median_win_pnl:.2f})")
    if non_neg_windows < 2:
        rejection_reasons.append(f"Less than 2 non-negative windows ({non_neg_windows}/3)")
    if n_trades < 30:
        rejection_reasons.append(f"Insufficient trades ({n_trades} < 30)")
    if ece > 0.10:
        rejection_reasons.append(f"Excessive ECE ({ece:.4f} > 0.10)")

    return {
        "variant": variant,
        "C": C,
        "class_weight": class_weight,
        "sample_weight_mode": sample_weight_mode,
        "sample_weight_tau": sample_weight_tau,
        "calibration_method": calibration_method,
        "feature_names": feature_names,
        "sequence_coverage": seq_coverage,
        "val_auc": round(val_auc, 4),
        "brier": round(brier, 4),
        "ece": round(ece, 4),
        "total_pnl": round(total_pnl, 4),
        "n_trades": n_trades,
        "win_rate": round(float(backtest.get("win_rate", 0.0)), 4),
        "roi": round(float(backtest.get("roi", 0.0)), 4),
        "max_drawdown": round(max_dd, 4),
        "oot_windows": oot_windows,
        "deployable": is_deployable,
        "rejection_reasons": rejection_reasons,
        "model_blob": pickle.dumps(final_model),
        "oof_scores": oof_cal_scores,
        "raw_oof_scores": oof_raw_scores,
    }


async def retrain_candidates_for_asset(
    session: Any,
    asset_name: str,
    *,
    dry_run: bool = False,
    top_n_save: int = 3,
) -> list[dict[str, Any]]:
    """Train experimental grid for one asset / phase and save top candidates."""
    print(f"\n--- Loading data for {asset_name} ---")
    df = await load_asset_snapshots(session, asset_name)
    if df.empty or len(df) < 50:
        print(f"Skipping {asset_name}: insufficient data ({len(df)} snapshots)")
        return []

    print(f"Loaded {len(df)} snapshots across {df['market_id'].nunique()} markets for {asset_name}.")

    feature_variants = ["BASE", "BASE+DERIVED", "BASE+LAGS"]
    c_values = [0.1, 1.0, 5.0]
    class_weights = [None, "balanced"]
    weight_configs = [
        ("uniform", 0.0),
        ("time_decay", 5.0),
    ]
    calibrations = ["RAW", "PLATT"]

    all_combos = list(itertools.product(
        feature_variants, c_values, class_weights, weight_configs, calibrations
    ))
    total_combos = len(all_combos)
    print(f"Evaluating grid of {total_combos} candidate combinations for {asset_name}...")

    candidates = []
    for idx, (var, c_val, cw, (w_mode, w_tau), cal) in enumerate(all_combos, start=1):
        if idx % 10 == 0 or idx == total_combos:
            print(f"[{asset_name}] Evaluated {idx}/{total_combos} combinations...")
        res = train_and_eval_candidate(
            df,
            variant=var,
            C=c_val,
            class_weight=cw,
            sample_weight_mode=w_mode,
            sample_weight_tau=w_tau,
            calibration_method=cal,
        )
        if res is not None:
            candidates.append(res)

    print(f"Completed evaluation: {len(candidates)} valid candidate runs.")

    # Sort candidates by COMBINED total_pnl descending, then median_pnl descending
    candidates.sort(
        key=lambda c: (c["deployable"], c["total_pnl"], c["oot_windows"].get("median_pnl", -999)),
        reverse=True,
    )

    if not dry_run and candidates:
        # Get next version from DB
        v_stmt = select(func.max(ModelRegistry.version)).where(ModelRegistry.asset == asset_name)
        v_res = await session.execute(v_stmt)
        max_v = v_res.scalar() or 0

        to_save = candidates[:top_n_save]
        for i, cand in enumerate(to_save):
            next_v = max_v + i + 1
            training_params = {
                "experiment_variant": cand["variant"],
                "C": cand["C"],
                "class_weight": cand["class_weight"],
                "sample_weight_mode": cand["sample_weight_mode"],
                "sample_weight_tau": cand["sample_weight_tau"],
                "calibration_method": cand["calibration_method"],
                "validation_scheme": "GROUPED_WALK_FORWARD",
                "target_source": "POLYMARKET_FLIP_VS_FINAL_OUTCOME",
                "sequence_coverage": cand["sequence_coverage"],
                "deployable": cand["deployable"],
                "rejection_reasons": cand["rejection_reasons"],
                "oot_windows": cand["oot_windows"],
                "combined_pnl": cand["total_pnl"],
                "n_trades": cand["n_trades"],
                "win_rate": cand["win_rate"],
                "max_drawdown": cand["max_drawdown"],
            }

            model_rec = ModelRegistry(
                asset=asset_name,
                version=next_v,
                model_type="logreg",
                model_blob=cand["model_blob"],
                accuracy=cand["val_auc"],
                features=",".join(cand["feature_names"]),
                ece=cand["ece"],
                backtest_pnl=cand["total_pnl"],
                backtest_trades=cand["n_trades"],
                is_active=False,  # STRICT INVARIANT: Always False
                decision_threshold=0.55,
                decision_threshold_down=0.45,
                quality_gate_passed=cand["deployable"],
                quality_gate_reasons={"reasons": cand["rejection_reasons"]},
                training_params=training_params,
                trained_at=datetime.now(timezone.utc),
                activation_source=None,
                activated_at=None,
                activated_by=None,
            )
            session.add(model_rec)
            await session.flush()

            # Save OOF artifact
            quotes_df = df[["market_id", "recorded_at", "mid_price", "best_bid", "best_ask", "final_outcome"]].copy()
            artifact_blob = serialize_oof_artifact(
                df,
                cand["oof_scores"],
                quotes_df,
                feature_set=cand["variant"],
                raw_scores=cand["raw_oof_scores"],
            )
            oof_rec = ModelRegistryOOFArtifact(
                model_registry_id=model_rec.id,
                schema_version=OOF_ARTIFACT_SCHEMA_VERSION,
                row_count=len(df),
                artifact_blob=artifact_blob,
                created_at=datetime.now(timezone.utc),
            )
            session.add(oof_rec)
            await session.commit()
            print(f"Saved Candidate v{next_v} (Asset: {asset_name}, Variant: {cand['variant']}, PnL: {cand['total_pnl']:.2f}, Deployable: {cand['deployable']})")

    return candidates


async def run_retraining(
    assets: Sequence[str] = (),
    phases: Sequence[str] = (),
    dry_run: bool = False,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    target_assets = list(assets) if assets else ["BTC", "ETH", "SOL", "XRP", "DOGE"]
    target_phases = list(phases) if phases else ["base", "decided", "leaning", "contested"]

    combinations = []
    for asset in target_assets:
        for phase in target_phases:
            name = asset if phase == "base" else f"{asset}_{phase}"
            combinations.append(name)

    all_results = {}
    async with async_session() as session:
        for combo in combinations:
            cands = await retrain_candidates_for_asset(session, combo, dry_run=dry_run)
            all_results[combo] = [
                {
                    "variant": c["variant"],
                    "C": c["C"],
                    "class_weight": c["class_weight"],
                    "sample_weight_mode": c["sample_weight_mode"],
                    "sample_weight_tau": c["sample_weight_tau"],
                    "calibration_method": c["calibration_method"],
                    "val_auc": c["val_auc"],
                    "brier": c["brier"],
                    "ece": c["ece"],
                    "total_pnl": c["total_pnl"],
                    "n_trades": c["n_trades"],
                    "win_rate": c["win_rate"],
                    "max_drawdown": c["max_drawdown"],
                    "deployable": c["deployable"],
                    "rejection_reasons": c["rejection_reasons"],
                    "oot_windows": c["oot_windows"],
                }
                for c in cands[:10]  # keep top 10 in report
            ]

    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_dir = output_dir or Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"logreg_candidate_comparison_{date_str}.json"
    json_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # Also build summary CSV
    rows = []
    for asset_key, cands in all_results.items():
        for i, c in enumerate(cands):
            rows.append({
                "asset_phase": asset_key,
                "rank": i + 1,
                "variant": c["variant"],
                "C": c["C"],
                "class_weight": c["class_weight"],
                "sample_weight_mode": c["sample_weight_mode"],
                "calibration": c["calibration_method"],
                "auc": c["val_auc"],
                "ece": c["ece"],
                "total_pnl": c["total_pnl"],
                "n_trades": c["n_trades"],
                "win_rate": c["win_rate"],
                "deployable": c["deployable"],
                "t1_pnl": c["oot_windows"].get("T1", {}).get("total_pnl"),
                "t2_pnl": c["oot_windows"].get("T2", {}).get("total_pnl"),
                "t3_pnl": c["oot_windows"].get("T3", {}).get("total_pnl"),
                "median_window_pnl": c["oot_windows"].get("median_pnl"),
                "rejection_reasons": "; ".join(c["rejection_reasons"]),
            })
    df_csv = pd.DataFrame(rows)
    csv_path = out_dir / f"logreg_candidate_comparison_{date_str}.csv"
    df_csv.to_csv(csv_path, index=False)

    print(f"\nRetraining complete! Reports written to {json_path} and {csv_path}")
    return all_results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain LogReg candidate models across experimental grid")
    parser.add_argument("--assets", nargs="*", type=str, default=(), help="Specific base assets (BTC, ETH, etc.)")
    parser.add_argument("--phases", nargs="*", type=str, default=(), help="Specific phases (base, decided, etc.)")
    parser.add_argument("--dry-run", action="store_true", help="Do not save models to database")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Output directory for reports")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    asyncio.run(
        run_retraining(
            assets=args.assets,
            phases=args.phases,
            dry_run=args.dry_run,
            output_dir=args.output_dir,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
