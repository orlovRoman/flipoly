"""Cross-universe comparison funnel (11v2) vs independent (indep) + provenance.

Writes cross_universe.json and provenance.json into BOTH run dirs.
"""
import csv
import hashlib
import json
import os
import subprocess
from collections import Counter

PTM = "artifacts/research/price_time_map"


def run_summary(run):
    base = os.path.join(PTM, run)
    ct = {}
    with open(os.path.join(base, "ct_states.csv")) as f:
        for r in csv.DictReader(f):
            ct[r["opportunity_id"]] = r["ct_state"]
    led = list(csv.DictReader(open(os.path.join(base, "opportunity_ledger.csv"))))
    out = {"run": run, "ledger_rows": len(led),
           "ok": sum(1 for r in led if r["selection_status"] == "OK")}
    for pol in ("GRID", "FUNNEL"):
        rs = [r for r in led if r["selection_status"] == "OK" and r.get("entry_policy") == pol
              and r.get("entry_variant") == "YES_OUTSIDER"
              and ct.get(r["opportunity_id"]) == "REVERSION"]
        out[pol.lower() + "_ct"] = {"n": len(rs),
                                    "gross": round(sum(float(r["gross_pnl"]) for r in rs), 2)}
    t5 = sorted((float(r["gross_pnl"]) for r in led
                 if r["selection_status"] == "OK" and r.get("entry_policy") == "GRID"
                 and r.get("entry_rule") == "T-5" and r.get("entry_variant") == "YES_OUTSIDER"
                 and ct.get(r["opportunity_id"]) == "REVERSION"), reverse=True)
    out["t5_ct"] = {"n": len(t5), "gross": round(sum(t5), 2),
                    "top5": round(sum(t5[:5]), 2),
                    "ex_top5": round(sum(t5) - sum(t5[:5]), 2)}
    af = json.load(open(os.path.join(base, "auto_findings.json")))
    surv = [p for p in af["positive_cells"]
            if p.get("holm_adj_p") is not None and p["holm_adj_p"] < 0.05]
    out["holm_survivors"] = len(surv)
    out["holm_best"] = min((p["holm_adj_p"] for p in af["positive_cells"]
                            if p.get("holm_adj_p") is not None), default=None)
    s2 = json.load(open(os.path.join(base, "ct_stage2.json")))
    out["pairs"] = [{k: p.get(k) for k in ("pair", "markets", "mean_diff", "ci95", "perm_p")}
                    for p in s2["pairs"]]
    return out


f = run_summary("ptm_20260909_11v2")
i = run_summary("ptm_20260910_indep")
cross = {
    "funnel": f,
    "independent": i,
    "delta_indep_minus_funnel": {
        "grid_ct_n": i["grid_ct"]["n"] - f["grid_ct"]["n"],
        "grid_ct_gross": round(i["grid_ct"]["gross"] - f["grid_ct"]["gross"], 2),
        "t5_ct_n": i["t5_ct"]["n"] - f["t5_ct"]["n"],
        "t5_ct_gross": round(i["t5_ct"]["gross"] - f["t5_ct"]["gross"], 2),
    },
    "reading": ("Non-funnel markets add 567 CT rows at -171.78 gross (~-0.30/entry). "
                "T-5 keeps the same top5 (+195) but ex-top5 falls to -154.44. "
                "Funnel screening selects markets where the policy loses less; "
                "the T-5 nominal edge does NOT generalize to the unscreened universe. "
                "0 Holm survivors on BOTH universes."),
}
with open(os.path.join(PTM, "cross_universe.json"), "w") as fh:
    json.dump(cross, fh, indent=2)


def git(*a):
    return subprocess.run(["git"] + list(a), capture_output=True, text=True).stdout.strip()


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


scripts = ["ptm_lib2.py", "ptm_build2.py", "ptm_analyze2.py", "ptm_ct2.py",
           "ptm_stage2.py", "ptm_stage2b.py", "ptm_verify2.py", "ptm_bias.py", "ptm_freeze.py"]
prov = {
    "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
    "commit": git("rev-parse", "HEAD"),
    "scripts": {n: {"sha256": sha(os.path.join("scripts", "research", n)),
                    "last_commit": git("log", "-1", "--format=%H", "--", "scripts/research/" + n)}
                for n in scripts},
    "freeze_chunks": {},
    "note": ("Manifests record script hashes AT BUILD TIME (dirty_worktree=true means "
             "the worktree had uncommitted changes then). This file records hashes AT "
             "COMMIT TIME. Ledger-affecting equivalence for the build2 streaming refactor "
             "was proven byte-identical on the funnel freeze (ptm_tmp_equiv run). "
             "Freeze chunks live on server disk (not git); per-chunk sha256 in "
             "_freeze/*/meta.json. Re-freeze from mutable DB does NOT reproduce inputs."),
}
for fr in ("freeze_funnel", "freeze_independent"):
    m = json.load(open(os.path.join(PTM, "_freeze", fr, "meta.json")))
    prov["freeze_chunks"][fr] = {"files": len(m["files"]), "rows": m["rows"],
                                 "meta_sha256": sha(os.path.join(PTM, "_freeze", fr, "meta.json"))}
with open(os.path.join(PTM, "provenance.json"), "w") as fh:
    json.dump(prov, fh, indent=2)
print("cross + provenance written; commit to fill:", prov["commit"][:7])
