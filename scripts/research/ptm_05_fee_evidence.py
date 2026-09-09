"""Dump fee-related evidence from prior research artifacts (read-only)."""
import json

for name in ("data_inventory.json", "experiment_protocol.json", "stage2_empirical_results.json"):
    try:
        d = json.load(open("artifacts/research/" + name, encoding="utf-8"))
    except FileNotFoundError:
        print(name, "MISSING")
        continue
    blob = json.dumps(d)
    print("=" * 20, name, len(blob), "chars")
    for key in ("fee", "commission", "taker", "spread_cost", "cost"):
        if key.lower() in blob.lower():
            print("  mentions:", key)
