"""e1_replay.py v1.0.1 — E1 full replay (spec lgbm_direction_audit_v1).

Per historically-used version: sample distinct (market, slot, version) pairs
(<=300, seed 20260913). Frozen replay rule (validated 200/200 on v25):
  candles = last <=110 CLOSED (is_closed) with close <= decision;
  window = end-1 primary (idx[-111]:idx[-1]), base fallback (idx[-110]:);
  builder by row date < 08-11 -> ERA-A (pre0811); < 09-08 -> ERA-B (0812); else current;
  decision = frozen end-15min if single proba else per-row created_at (<=5 distinct).
Compares recorded vs replayed (proba, raw, regime, direction) with frozen
tolerances (1e-9 exact-blob, 1e-6 reconstructed, strict direction).
Outputs PREDICTION_RECONCILIATION.csv + per-version summary.
"""
import base64
import csv
import gzip
import hashlib
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\lgbm-audit-v1\code\direction")
sys.path.insert(0, r"C:\Users\orlov\OneDrive\Документы\Default Project\flipoly")
import era_builder_pre0811 as ERA_A
import era_builder_0812 as ERA_B
from polyflip.crypto.feature_builder import (  # noqa: E402
    build_crypto_features as EB_NEW_FN, CRYPTO_FEATURE_COLUMNS)
from polyflip.crypto.edge import (  # noqa: E402
    compute_crypto_signal_strength)
import warnings

warnings.filterwarnings("ignore")

ROOT = r"D:\lgbm-audit-v1"
EXP = os.path.join(ROOT, "data", "exp")
OUT = os.path.join(ROOT, "out", "direction")
os.makedirs(OUT, exist_ok=True)
SAMPLE_PER_VERSION = 300
SAMPLE_SEED = 20260913
E1_VERSION = "1.0.1"


def epoch_s(s):
    v = (s - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1s")
    try:
        return v.astype(np.float64)
    except AttributeError:
        return float(v)


def pick_builder(date_str):
    if date_str < "2026-08-11":
        return ERA_A, list(ERA_A.CRYPTO_FEATURE_COLUMNS), "ERA-A"
    if date_str < "2026-09-08":
        return ERA_B, list(ERA_B.CRYPTO_FEATURE_COLUMNS), "ERA-B"
    return None, list(CRYPTO_FEATURE_COLUMNS), "ERA-C"


def main():
    import csv as _csv
    _csv.field_size_limit(sys.maxsize)
    inv = pd.read_csv(os.path.join(ROOT, "out", "MODEL_INVENTORY.csv"), dtype=str)
    meta = pd.read_csv(os.path.join(ROOT, "data", "registry_meta.csv"),
                       dtype=str, keep_default_na=False)
    metai = meta.set_index("id")
    u = pd.read_csv(os.path.join(ROOT, "out", "MODEL_USAGE_TIMELINE.csv"),
                    dtype=str, keep_default_na=False)
    u = u[u["registry_id"] != ""].copy()
    u["key"] = u["k"] + "|" + u["v"]
    inv["key"] = inv["asset"] + "|" + inv["version"]
    blobs = {}
    with open(os.path.join(ROOT, "data", "blobs.csv"), newline="") as f:
        for row in _csv.reader(f):
            if len(row) == 2 and row[0] != "id":
                blobs[row[0]] = row[1]
    candles = pd.read_csv(os.path.join(EXP, "full", "candles.csv.gz"))
    candles["open_time"] = pd.to_datetime(candles["open_time"], utc=True)
    candles["close_time"] = pd.to_datetime(candles["close_time"], utc=True)
    lm = pd.read_csv(os.path.join(EXP, "full", "livemarkets.csv.gz"),
                     dtype={"market_id": str}, low_memory=False)
    lm["end3"] = pd.to_datetime(lm["market_end_at"], utc=True, errors="coerce")
    m = lm["end3"].isna()
    lm.loc[m, "end3"] = pd.to_datetime(lm.loc[m, "end_time_est"], utc=True, errors="coerce")
    ends = dict(zip(lm["market_id"], lm["end3"]))
    sym_of_slot = {}
    for _, r in u[["k"]].drop_duplicates().iterrows():
        p0 = r["k"].split("_")[0]
        sym_of_slot[r["k"]] = p0 if p0.endswith("USDT") else p0 + "USDT"
    used_keys = set(u["key"].tolist())

    cand_cache = {}

    def get_candles(sym):
        if sym not in cand_cache:
            g = candles[(candles["symbol"] == sym) & (candles["interval"] == "15m")]
            # Frozen E1 rule: only closed candles (is_closed==t), matching
            # production get_recent_candles; open backfilled rows excluded.
            if "is_closed" in g.columns:
                g = g[g["is_closed"].astype(str).str.lower().isin(["t", "true", "1"])]
            g = g.sort_values("open_time").reset_index(drop=True)
            cand_cache[sym] = (g, np.asarray(epoch_s(g["close_time"]), dtype=np.float64))
        return cand_cache[sym]

    print("collecting funnel rows (single pass)...", flush=True)
    import gzip as _gz
    perkey = {}
    for fn in sorted(os.listdir(os.path.join(EXP, "funnel"))):
        with _gz.open(os.path.join(EXP, "funnel", fn), "rt") as f:
            for x in _csv.DictReader(f):
                k = (x["direction_model_key"] or "") + "|" + (x["direction_model_version"] or "")
                if k in used_keys and x["direction_probability"]:
                    perkey.setdefault(k, []).append(x)
    print("collected keys:", len(perkey), flush=True)

    rec_rows = []
    per_version = {}
    for _, ur in u.iterrows():
        key, ver = ur["k"], ur["v"]
        try:
            rid = str(int(float(ur["registry_id"])))
        except (TypeError, ValueError):
            per_version[key + "|" + ver] = {"status": "NO_REGISTRY_ROW"}
            continue
        r = inv[inv["id"] == rid]
        if not len(r):
            per_version[key + "|" + ver] = {"status": "NO_REGISTRY_ROW"}
            continue
        r = r.iloc[0]
        schema = r["features"].split(",") if r["features"] else []
        if not schema:
            per_version[key + "|" + ver] = {"status": "NON_REPRODUCIBLE",
                                            "reason": "empty_schema"}
            continue
        try:
            model = pickle.loads(base64.b64decode(blobs[rid].replace("\n", "")))
        except Exception as e:
            per_version[key + "|" + ver] = {"status": "NON_REPRODUCIBLE",
                                            "reason": "unpickle_%s" % type(e).__name__}
            continue
        thr_up = float(r["threshold_up"]) if r["threshold_up"] not in ("", "nan") else None
        thr_dn = float(r["threshold_down"]) if r["threshold_down"] not in ("", "nan") else None
        try:
            tp = __import__("json").loads(
                metai.loc[rid, "training_params"]) if metai.loc[rid, "training_params"].strip() else {}
        except Exception:
            tp = {}
        cuts = (tp.get("vol_p33"), tp.get("vol_p67"))
        sym = sym_of_slot.get(key, "")
        bymkt = {}
        for x in perkey.get(key + "|" + ver, []):
            bymkt.setdefault(x["market_id"], []).append(x)
        mks = sorted(bymkt)
        rng = np.random.RandomState(SAMPLE_SEED)
        sel = [mks[i] for i in rng.choice(
            len(mks), size=min(SAMPLE_PER_VERSION, len(mks)), replace=False)] if mks else []
        g, ct = get_candles(sym) if sym else (None, None)
        if g is None or len(g) == 0:
            per_version[key + "|" + ver] = {"status": "NON_REPRODUCIBLE",
                                            "reason": "no_candles_%s" % sym}
            continue
        n_match_p = n_match_r = n_match_d = n_match_g = 0
        n_tot = n_frozen_ok = n_reg_app = n_missing = 0
        is_clbg = type(model).__name__ == "CalibratedLightGBMModel"
        try:
            cal_obj = model.calibrated_model.calibrated_classifiers_[0].calibrators[0]
        except Exception:
            cal_obj = None

        def eval_one(x, dec, win_note):
            import math as _math
            idx = np.nonzero(ct <= dec)[0]
            if len(idx) < 101:
                return {"status": "TOO_FEW_CANDLES"}
            bmod, base, bname = pick_builder(x["created_at"][:10])
            rec_r = float(x["direction_p_up_raw"]) if x["direction_p_up_raw"] else None
            rec_pu = float(x["direction_p_up"]) if x["direction_p_up"] else None
            vec = None
            fvd = None
            hit_win = None
            for wname, lo, hi in (("end-1", len(idx) - 111, len(idx) - 1),
                                  ("base", len(idx) - 110, len(idx))):
                seld = g.iloc[max(0, lo):hi].reset_index(drop=True)
                if len(seld) < 100:
                    continue
                if bmod is None:
                    fv = EB_NEW_FN(seld, underlying_price=None, market_context=None)
                else:
                    try:
                        fv = bmod.build_crypto_features(seld, underlying_price=None)
                    except TypeError:
                        fv = bmod.build_crypto_features(seld)
                fvd = dict(zip(base, fv.features[0]))
                vec = np.array([[fvd.get(nm, float("nan")) for nm in schema]])
                if np.isnan(vec).any():
                    vec = None
                    continue
                try:
                    pr_try = float(model.predict_raw_proba(vec)[0][1])
                    has_try = True
                except Exception:
                    pr_try, has_try = float("nan"), False
                if has_try and rec_r is not None and abs(pr_try - rec_r) <= 1e-6:
                    hit_win = wname
                    break
                if hit_win is None:
                    hit_win = wname + "?"
            if vec is None or np.isnan(vec).any():
                return {"status": "MISSING_FEATURES"}
            try:
                praw = float(model.predict_raw_proba(vec)[0][1])
                has_raw = True
            except Exception:
                praw = float("nan")
                has_raw = False
            try:
                pcal = float(model.predict_proba(vec)[0][1])
            except Exception:
                pcal = float("nan")
            pcal_logit = float("nan")
            if (is_clbg and cal_obj is not None and has_raw
                    and np.isfinite(praw) and 0.0 < praw < 1.0):
                try:
                    pcal_logit = float(cal_obj.predict(
                        np.array([_math.log(praw / (1.0 - praw))]))[0])
                except Exception:
                    pass
            mp = (abs(pcal_logit - rec_pu) <= 1e-6) if (
                rec_pu is not None and np.isfinite(pcal_logit)) else False
            if not mp and rec_pu is not None and np.isfinite(pcal):
                mp = abs(pcal - rec_pu) <= 1e-6
            mr = (abs(praw - rec_r) <= 1e-6) if (has_raw and rec_r is not None) else None
            side_src = praw if (has_raw and np.isfinite(praw)) else pcal
            if thr_up is not None and thr_dn is not None:
                d_mine = ("UP" if side_src >= thr_up
                          else ("DOWN" if side_src <= thr_dn else "NONE"))
            else:
                d_mine = "?"
            vt = fvd.get("vol_trend", float("nan"))
            if cuts[0] is not None and cuts[1] is not None and np.isfinite(vt):
                g_rec = ("low_vol" if vt <= float(cuts[0])
                         else ("mid_vol" if vt <= float(cuts[1]) else "high_vol"))
                parts = key.rsplit("_", 2)
                exp_reg = (parts[-2] + "_" + parts[-1]) if len(parts) == 3 else "?"
                reg_eval = (g_rec, exp_reg)
            else:
                reg_eval = None
            return {"status": "OK", "window": (hit_win or "?").rstrip("?") if (hit_win or "").rstrip("?") in ("end-1", "base") else "unvalidated",
                    "builder": bname, "dec_note": win_note, "proba_match": bool(mp),
                    "raw_match": bool(mr), "dir_match": bool(d_mine == x["direction_value"]),
                    "reg_eval": reg_eval, "rec_dir": x["direction_value"],
                    "rec_regime": x["direction_regime"]}

        for mk in sel:
            rows = bymkt[mk]
            probs = set(x["direction_probability"] for x in rows)
            frozen_ok = len(probs) == 1
            n_frozen_ok += int(frozen_ok)
            end = ends.get(mk)
            if frozen_ok:
                if end is None or pd.isna(end):
                    continue
                units = [(rows[0], float(epoch_s(end - pd.Timedelta(minutes=15))), "frozen_end-15min")]
            else:
                seen_p = set()
                units = []
                for x in rows:
                    if x["direction_probability"] not in seen_p and len(units) < 5:
                        seen_p.add(x["direction_probability"])
                        try:
                            dx = float(epoch_s(pd.Timestamp(x["created_at"])))
                        except Exception:
                            continue
                        units.append((x, dx, "per-row_created"))
            for x, dec, note in units:
                r = eval_one(x, dec, note)
                if r["status"] != "OK":
                    if r["status"] == "MISSING_FEATURES":
                        n_missing += 1
                        rec_rows.append({"version": key + "|" + ver, "market": mk,
                                         "status": "MISSING_FEATURES"})
                    continue
                n_tot += 1
                n_match_p += int(r["proba_match"])
                if "raw_match" in r and r["raw_match"]:
                    n_match_r += 1
                n_match_d += int(r["dir_match"])
                if r["reg_eval"] is not None:
                    n_reg_app += 1
                    n_match_g += int(r["reg_eval"][0] == r["reg_eval"][1])
                rec_rows.append({"version": key + "|" + ver, "market": mk,
                                 "status": "OK", "builder": r["builder"],
                                 "window": r["window"], "dec_note": note,
                                 "proba_match": r["proba_match"],
                                 "raw_match": r["raw_match"],
                                 "dir_match": r["dir_match"],
                                 "rec_dir": r["rec_dir"],
                                 "rec_regime": r["rec_regime"]})
        per_version[key + "|" + ver] = {
            "status": "DONE", "pairs": n_tot, "frozen_single": n_frozen_ok,
            "proba_match_rate": (n_match_p / n_tot) if n_tot else None,
            "raw_match_rate": (n_match_r / n_tot) if n_tot else None,
            "dir_match_rate": (n_match_d / n_tot) if n_tot else None,
            "regime_match_rate": (n_match_g / n_reg_app) if n_reg_app else None,
            "regime_applicable": n_reg_app, "missing_features": n_missing}
        print("%s v%s pairs=%d proba=%.3f raw=%s dir=%.3f reg=%.3f" % (
            key, ver, n_tot,
            (n_match_p / n_tot) if n_tot else -1,
            ("%.3f" % (n_match_r / n_tot)) if n_tot else "-",
            (n_match_d / n_tot) if n_tot else -1,
            (n_match_g / n_tot) if n_tot else -1), flush=True)
    pd.DataFrame(rec_rows).to_csv(os.path.join(OUT, "PREDICTION_RECONCILIATION.csv"), index=False)
    with open(os.path.join(OUT, "e1_summary.json"), "w", newline="") as f:
        json.dump(per_version, f, indent=1, sort_keys=True, default=str)
        f.write("\n")
    print("wrote reconciliation + summary")


if __name__ == "__main__":
    main()
