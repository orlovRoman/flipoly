"""decisions.py v1.0.0 — decision events (spec lgbm_shared_v1 steps 27-30).

decision_event_id = funnel.id. decision_at = created_at (funnel.timestamp is
empty everywhere: every row is LOG_WRITE_TIME_PROXY).
model_slot: direction_model_key present -> key; elif any direction
version/probability present -> UNKNOWN_MODEL; else NO_MODEL.
"""
import numpy as np
import pandas as pd

PROXY_NOTE = "LOG_WRITE_TIME_PROXY"


def build_events(f):
    e = pd.DataFrame()
    e["decision_event_id"] = f["id"].to_numpy()
    e["decision_at"] = f["created_at"]
    e["market_id"] = f["market_id"].astype(str).to_numpy()
    e["asset"] = f["asset"].to_numpy()
    e["decision_time_proof"] = PROXY_NOTE
    has_key = f["direction_model_key"].notna() & (f["direction_model_key"] != "")
    has_ver = f["direction_model_version"].notna()
    has_prob = f["direction_probability"].notna()
    slot = np.full(len(f), "NO_MODEL", dtype=object)
    slot[has_key.to_numpy()] = f.loc[has_key, "direction_model_key"].astype(str).to_numpy()
    unk = ~has_key.to_numpy() & (has_ver.to_numpy() | has_prob.to_numpy())
    slot[unk] = "UNKNOWN_MODEL"
    e["model_slot"] = slot
    e["slot_version"] = f["direction_model_version"].to_numpy()
    e["slot_probability"] = f["direction_probability"].to_numpy()
    e["slot_value"] = f["direction_value"].to_numpy()
    e["slot_regime"] = f["direction_regime"].to_numpy()
    e["final_action"] = f["final_action"].to_numpy()
    e["skip_reason"] = f["skip_reason"].to_numpy()
    e["trading_mode"] = f["trading_mode"].to_numpy()
    e["execution_mode"] = f["execution_mode"].to_numpy()
    e["decision_run_id"] = f["decision_run_id"].astype(str).to_numpy()
    for c in ("min_edge_used", "max_acceptable_price", "mrf_mode",
              "threshold_lower", "threshold_upper"):
        e[c] = f[c].to_numpy() if c in f.columns else np.nan
    return e
