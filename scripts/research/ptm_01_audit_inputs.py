"""PTM step 4: audit input composition. Read-only. Writes coverage JSON + manifest."""
import collections
import hashlib
import json
import os
import sys

ART = os.path.join("artifacts", "weighted_policy", "observations_30d.json")
EXP = os.path.join("artifacts", "research", "market_expirations.json")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    obs_doc = json.load(open(ART, encoding="utf-8"))
    obs = obs_doc["observations"]
    exp = json.load(open(EXP, encoding="utf-8"))
    print("rows:", len(obs))
    print("assets:", dict(collections.Counter(o.get("asset") for o in obs)))
    print("time_left_sec nonnull:", sum(1 for o in obs if o.get("time_left_sec") is not None))
    for col in ("best_bid", "best_ask", "yes_ask", "no_ask", "yes_bid", "no_bid"):
        print(col, "nonnull:", sum(1 for o in obs if o.get(col) is not None))
    print("outcome_yes:", dict(collections.Counter(o.get("outcome_yes") for o in obs)))
    print("p_logreg_yes nonnull:", sum(1 for o in obs if o.get("p_logreg_yes") is not None))
    print("p_lgbm_yes nonnull:", sum(1 for o in obs if o.get("p_lgbm_yes") is not None))
    print("market_role:", dict(collections.Counter(o.get("market_role") for o in obs)))
    print("phase:", dict(collections.Counter(o.get("phase") for o in obs)))
    print("expirations:", len(exp))
    mids = {str(o.get("market_id")) for o in obs}
    print("markets total:", len(mids), "with expiry:", sum(1 for m in mids if m in exp))
    ts = sorted(o["timestamp"] for o in obs if o.get("timestamp"))
    print("period:", ts[0], "->", ts[-1])
    print("all keys:", sorted({k for o in obs for k in o.keys()}))
    out = {
        "rows": len(obs),
        "markets": len(mids),
        "markets_with_expiry": sum(1 for m in mids if m in exp),
        "input_hashes": {"observations_30d.json": sha256_file(ART), "market_expirations.json": sha256_file(EXP)},
    }
    os.makedirs(os.path.join("artifacts", "research", "price_time_map", "run_audit"), exist_ok=True)
    json.dump(out, open(os.path.join("artifacts", "research", "price_time_map", "run_audit", "coverage.json"), "w"), indent=2)
    print("wrote coverage.json")


if __name__ == "__main__":
    sys.exit(main())
