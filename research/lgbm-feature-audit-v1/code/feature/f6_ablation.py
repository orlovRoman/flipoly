"""f6_ablation.py — F6 group ablation (spec lgbm_feature_audit_v1.yaml).

Same LGBM params across ALL variants (documented). Per validation fold (F2..F6,
expanding non-overlapping train chain):
  - ALL             : all 79 catalog features
  - DROP_GROUP      : 79 minus group
  - ONLY_GROUP      : group features only
  - MINIMAL_CONTROL : [ret_1, mid_price, spread, time_left_min]
                      (spec minimal: binance [ret_1] + final_outcome
                      [p_market_yes, time_to_expiry] + flip [mid_price, spread,
                      time_left]; p_market_yes is the market's own probability
                      and is absent from the feature catalog by design)
Metrics vs contract_target: logloss, brier, canonical net pnl (yes/no side vs
yes_ask/no_ask, settlement by contract_target). delta_* vs ALL per fold.

Output: out/feature/GROUP_ABLATION.csv
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MATRIX = os.path.join(ROOT, "out", "feature", "FEATURE_MATRIX.parquet")
SHARED = os.path.join(ROOT, "out", "shared_ds", "shared_dataset.parquet")
OUT = os.path.join(ROOT, "out", "feature", "GROUP_ABLATION.csv")

FOLD_CHAIN = ["trainpool", "F1", "F2", "F3", "F4", "F5", "F6"]
EVAL_FOLDS = ["F2", "F3", "F4", "F5", "F6"]
SEED = 1776
TRAIN_CAP = 120000

LGBM_PARAMS = dict(objective="binary", metric="binary_logloss", learning_rate=0.05,
                   num_leaves=31, min_child_samples=50, subsample=0.9,
                   subsample_freq=1, colsample_bytree=0.9, n_estimators=250,
                   random_state=SEED, verbose=-1, n_jobs=-1)

GROUPS = None
ALL_FEATURES = [
    "alternation_rate_6", "bb_position", "bb_width", "body_to_range",
    "consec_balance", "consec_down", "consec_up", "consecutive_down", "consecutive_up",
    "cvd_1", "cvd_6", "cvd_trend", "day_of_week",
    "dev_sq_x_phase", "deviation_x_time",
    "direction_lag_1", "direction_lag_2", "direction_lag_3",
    "dist_to_high_24", "dist_to_high_96", "dist_to_low_24", "dist_to_low_96",
    "dow", "dow_cos", "dow_sin",
    "ema_ratio_9_21",
    "funding_extreme", "funding_rate", "funding_rate_ma3",
    "high_price_final", "hour_cos", "hour_of_day", "hour_sin", "hour_utc",
    "is_final_phase", "log_moneyness", "log_time_left",
    "mid_price",
    "pm_best_ask", "pm_best_bid", "pm_momentum_5m", "pm_quote_pressure",
    "pm_spread_pct", "pm_volume_5m",
    "price_deviation", "price_deviation_sq", "price_distance_from_max",
    "price_momentum", "price_velocity", "price_velocity_lag1",
    "range_1", "range_avg_24",
    "ret_1", "ret_12", "ret_24", "ret_3", "ret_48", "ret_6",
    "rsi_14",
    "signed_body_pct", "signed_trend_efficiency_6",
    "spread", "spread_pct", "spread_trend",
    "strike_gap_pct", "taker_buy_ratio",
    "time_left_min", "time_phase",
    "up_ratio_4", "velocity_x_phase",
    "vol_24", "vol_48", "vol_6", "vol_ratio", "vol_trend", "vol_z_1", "vol_z_6",
    "volume_5min", "volume_trend",
]
MINIMAL_CONTROL = ["ret_1", "mid_price", "spread", "time_left_min"]


def load_groups():
    cat = pd.read_csv(os.path.join(ROOT, "out", "feature", "FEATURE_CATALOG.csv"),
                      dtype=str, keep_default_na=False)
    g = {}
    for k, v in cat.groupby("group")["feature"].agg(list).items():
        g[str(k)] = [f for f in v if f in ALL_FEATURES]
    return g


FEE_RATE = 0.07      # canonical weighted_fee_rate (matches e5 CFG.fee_rate)
SLIPPAGE_RATE = 0.005  # L1 + 0.5% (matches e5 CFG.slippage_rate)


def exec_cost(ask):
    """Canonical all-in cost for 1 share at quoted ask: ask + fee + slippage.
    fee = FEE_RATE*ask*(1-ask); slippage = SLIPPAGE_RATE*ask (same quadratic
    model as e5_real.diagnose: a=0.07, b=-(1+s+fee))."""
    ask = np.asarray(ask, dtype=np.float64)
    return ask + FEE_RATE * ask * (1.0 - ask) + SLIPPAGE_RATE * ask


def canonical_side_net(p, y, yes_ask, no_ask):
    """Single-side selection on a fixed $1 budget; realized net per opportunity.

    cost_i = ask_i + fee(ask_i) + slippage(ask_i); buy q = 1/cost_i shares.
    BUY_YES: win (y=1) net = q-1, loss net = -1.
    BUY_NO:  win (y=0) net = q-1, loss net = -1.
    SKIP: no position, net = 0.
    Selection: enter on positive edge; BUY_YES if e_yes>0 and e_yes>=e_no,
    else BUY_NO if e_no>0. No side is ever overwritten.
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    c_yes = exec_cost(yes_ask)
    c_no = exec_cost(no_ask)
    e_yes = p - c_yes
    e_no = (1.0 - p) - c_no
    side = np.full(len(p), "SKIP", dtype=object)
    choose_yes = (e_yes > 0.0) & (e_yes >= e_no)
    choose_no = (e_no > 0.0) & (e_no > e_yes)
    side[choose_yes] = "BUY_YES"
    side[choose_no] = "BUY_NO"
    traded = choose_yes | choose_no
    qty = np.ones(len(p))
    qty[choose_yes] = 1.0 / np.maximum(c_yes[choose_yes], 1e-9)
    qty[choose_no] = 1.0 / np.maximum(c_no[choose_no], 1e-9)
    win = np.zeros(len(p), dtype=bool)
    w_y = y[choose_yes] == 1.0
    w_n = y[choose_no] == 0.0
    win[choose_yes] = w_y
    win[choose_no] = w_n
    net = np.zeros(len(p))
    net[traded] = np.where(win[traded], qty[traded] - 1.0, -1.0)
    return side, net


def canonical_pnl(p, y, yes_ask, no_ask):
    """Mean realized net on a $1 budget per opportunity (canonical economics)."""
    _, net = canonical_side_net(p, y, yes_ask, no_ask)
    return float(net.mean())


def main():
    global GROUPS
    GROUPS = load_groups()
    m = pd.read_parquet(MATRIX, columns=["fold", "decision_event_id", "decision_at"] + ALL_FEATURES + ["contract_target", "flip_native"])
    sd = pd.read_parquet(SHARED, columns=["decision_event_id", "yes_ask", "no_ask"])
    m = m.merge(sd, on="decision_event_id", how="left")
    for c in ("yes_ask", "no_ask"):
        m[c] = m[c].astype(float)
    # training label: canonical target, falling back to native where canonical
    # is unavailable (canonical+native only exist from F2 onward)
    m["y_tr"] = m["contract_target"].fillna(m["flip_native"])

    variants = [("ALL", ALL_FEATURES)]
    for grp, feats in GROUPS.items():
        feats = [f for f in feats if f in ALL_FEATURES]
        variants.append((f"DROP_GROUP_{grp}", [f for f in ALL_FEATURES if f not in feats]))
        variants.append((f"ONLY_GROUP_{grp}", feats))
    variants.append(("MINIMAL_CONTROL", MINIMAL_CONTROL))
    print("num variants:", len(variants))

    rows = []
    preds = {}
    for fold in EVAL_FOLDS:
        tr_mask = m["fold"].isin(FOLD_CHAIN[: FOLD_CHAIN.index(fold)])
        va_mask = m["fold"] == fold
        y_va = m.loc[va_mask, "contract_target"].to_numpy(float)
        okv = np.isfinite(y_va)
        y_va_ok = y_va[okv]
        ask_va = m.loc[va_mask, "yes_ask"].to_numpy(float)[okv]
        noa_va = m.loc[va_mask, "no_ask"].to_numpy(float)[okv]
        ys = m.loc[tr_mask, "y_tr"].to_numpy(float)
        oks = np.isfinite(ys)
        base = pd.DataFrame({c: m.loc[tr_mask, c].to_numpy(float) for c in ALL_FEATURES})
        val = pd.DataFrame({c: m.loc[va_mask, c].to_numpy(float) for c in ALL_FEATURES}).loc[okv]
        y_tr_ok = ys[oks]
        tr_ok = base.loc[oks]

        if len(tr_ok) == 0:
            print(f"fold {fold}: no training labels in train chain, skipped", flush=True)
            continue
        preds[fold] = {}

        if len(tr_ok) > TRAIN_CAP:
            rng = np.random.default_rng(SEED + FOLD_CHAIN.index(fold))
            ix = rng.choice(len(tr_ok), TRAIN_CAP, replace=False)
            tr_ok = tr_ok.iloc[ix]
            y_tr_ok = y_tr_ok[ix]

        # chronological split of train for early stopping (last 15k rows)
        stop_sz = min(15000, int(len(tr_ok) * 0.15))
        tr_fit = tr_ok.iloc[:-stop_sz]
        y_fit = y_tr_ok[:-stop_sz]
        tr_stop = tr_ok.iloc[-stop_sz:].reset_index(drop=True)
        y_stop = y_tr_ok[-stop_sz:]

        per_fold = {"fold": fold, "n_train": len(tr_fit), "n_val": int(okv.sum())}
        for name, feats in variants:
            fset = [f for f in feats if f in ALL_FEATURES]
            Xtr = tr_fit[fset]
            Xstop = tr_stop[fset]
            Xva = val[fset]
            mdl = lgb.LGBMClassifier(**LGBM_PARAMS)
            mdl.fit(Xtr, y_fit, eval_set=[(Xstop, y_stop)],
                    callbacks=[lgb.early_stopping(50, verbose=False)])
            p = np.clip(mdl.predict_proba(Xva)[:, 1], 1e-6, 1 - 1e-6)
            preds[fold][name] = p.astype(np.float16)
            ll = float(-(y_va_ok * np.log(p) + (1 - y_va_ok) * np.log(1 - p)).mean())
            br = float(((p - y_va_ok) ** 2).mean())
            pnl = float(canonical_pnl(p, y_va_ok, ask_va, noa_va))
            rows.append({"variant": name, "group": name.split("_", 2)[-1] if name != "ALL"
                         and name != "MINIMAL_CONTROL" else "",
                         **per_fold, "logloss": ll, "brier": br, "canonical_pnl": pnl,
                         "best_iter": int(mdl.best_iteration_ if hasattr(mdl, "best_iteration_") else -1)})
            print(f"{fold} {name:32s} ll={ll:.5f} br={br:.5f} pnl={pnl:.5f}", flush=True)

    out = pd.DataFrame(rows)
    # deltas vs ALL within fold
    all_base = out[out["variant"] == "ALL"][["fold", "logloss", "brier", "canonical_pnl"]]
    out = out.merge(all_base, on="fold", suffixes=("", "_base"))
    out["delta_logloss"] = out["logloss"] - out["logloss_base"]
    out["delta_brier"] = out["brier"] - out["brier_base"]
    out["delta_canonical_pnl"] = out["canonical_pnl"] - out["canonical_pnl_base"]
    out = out.drop(columns=["logloss_base", "brier_base", "canonical_pnl_base"])
    out.to_csv(OUT, index=False)
    print("wrote", OUT)

    # group-level summary
    drop = out[out["variant"].str.startswith("DROP_GROUP")]
    agg = drop.groupby("group").agg(
        n_folds=("fold", "nunique"),
        delta_logloss_mean=("delta_logloss", "mean"),
        delta_brier_mean=("delta_brier", "mean"),
        pnl_impact_mean=("delta_canonical_pnl", "mean"),
        positive_folds_brier=("delta_brier", lambda s: int((s < 0).sum())),
        positive_folds_ll=("delta_logloss", lambda s: int((s < 0).sum())),
    ).reset_index().sort_values("delta_brier_mean")
    print("\nDROP_GROUP summary (negative delta = group removal helps? (delta>0 = group helps):")
    print(agg.to_string(index=False))
    agg.to_csv(os.path.join(ROOT, "out", "feature", "GROUP_ABLATION_SUMMARY.csv"), index=False)

    # ---------------- F7: policy-impact chain ----------------

    imp_rows = []
    for fold in EVAL_FOLDS:
        if fold not in preds:
            continue
        va_mask = m["fold"] == fold
        y_va = m.loc[va_mask, "contract_target"].to_numpy(float)
        okv = np.isfinite(y_va)
        ask_va = m.loc[va_mask, "yes_ask"].to_numpy(float)[okv]
        noa_va = m.loc[va_mask, "no_ask"].to_numpy(float)[okv]
        p_all = np.clip(preds[fold]["ALL"].astype(np.float32), 1e-6, 1 - 1e-6)
        s_all, net_all = canonical_side_net(p_all, y_va[okv], ask_va, noa_va)
        n = len(p_all)
        for name in preds[fold]:
            if name in ("ALL", "MINIMAL_CONTROL"):
                continue
            p_v = np.clip(preds[fold][name].astype(np.float32), 1e-6, 1 - 1e-6)
            s_v, net_v = canonical_side_net(p_v, y_va[okv], ask_va, noa_va)
            flip = (s_v != s_all)
            ll_v = float(-(y_va[okv] * np.log(p_v) + (1 - y_va[okv]) * np.log(1 - p_v)).mean())
            ll_a = float(-(y_va[okv] * np.log(p_all) + (1 - y_va[okv]) * np.log(1 - p_all)).mean())
            imp_rows.append({
                "group": name.split("_", 2)[-1], "fold": fold, "n_val": n,
                "pred_flip_rate": float(flip.mean()),
                "side_flip_rate": float(flip.mean()),
                "trades_all": int((s_all != "SKIP").sum()),
                "trades_variant": int((s_v != "SKIP").sum()),
                "delta_volume_trades": int((s_v != "SKIP").sum() - (s_all != "SKIP").sum()),
                "net_all_mean": float(net_all.mean()),
                "net_variant_mean": float(net_v.mean()),
                "delta_pnl": float(net_v.mean() - net_all.mean()),
                "delta_logloss": float(ll_v - ll_a),
                "delta_brier": float(((p_v - y_va[okv]) ** 2).mean() - ((p_all - y_va[okv]) ** 2).mean()),
            })
            print(f"F7 {fold} {name:32s} dnet={net_v.mean()-net_all.mean():+.6f} "
                  f"dll={ll_v-ll_a:+.6f} trades={int((s_v!='SKIP').sum())}", flush=True)
    imp = pd.DataFrame(imp_rows)
    imp.to_csv(os.path.join(ROOT, "out", "feature", "FEATURE_POLICY_IMPACT.csv"), index=False)
    print("\nF7 FEATURE_POLICY_IMPACT rows:", len(imp))

    # persist per-fold predictions for day-block CI analysis (verdicts stage)
    pdir = os.path.join(ROOT, "out", "feature", "_abl_preds")
    os.makedirs(pdir, exist_ok=True)
    for fold in preds:
        va_mask = m["fold"] == fold
        y_va_f = m.loc[va_mask, "contract_target"].to_numpy(float)
        okv_f = np.isfinite(y_va_f)
        dfp = pd.DataFrame({
            "decision_at": m.loc[va_mask, "decision_at"].to_numpy()[okv_f],
            "contract_target": y_va_f[okv_f],
            "yes_ask": m.loc[va_mask, "yes_ask"].to_numpy(float)[okv_f],
            "no_ask": m.loc[va_mask, "no_ask"].to_numpy(float)[okv_f],
            **{k: v.astype(np.float16) for k, v in preds[fold].items()},
        })
        dfp = dfp[dfp["contract_target"].notna()].reset_index(drop=True)
        dfp.to_parquet(os.path.join(pdir, f"{fold}.parquet"), index=False)
        print("saved _abl_preds", fold, len(dfp))

    if len(imp):
        g = imp.groupby("group").agg(
            n_folds=("fold", "nunique"),
            pred_flip_rate_mean=("pred_flip_rate", "mean"),
            delta_pnl_mean=("delta_pnl", "mean"),
            delta_volume_mean=("delta_volume_trades", "mean"),
            delta_logloss_mean=("delta_logloss", "mean"),
        ).reset_index().sort_values("delta_pnl_mean")
        print(g.to_string(index=False))


if __name__ == "__main__":
    main()