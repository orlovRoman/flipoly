"""PTM probe: rows-per-market, quote completeness, fee fields, p_market_yes."""
import collections
import json

obs = json.load(open("artifacts/weighted_policy/observations_30d.json", encoding="utf-8"))["observations"]
bym = collections.defaultdict(list)
for o in obs:
    bym[str(o["market_id"])].append(o)
counts = collections.Counter(len(v) for v in bym.values())
print("rows-per-market distribution:", dict(counts))
print("p_market_yes nonnull:", sum(1 for o in obs if o.get("p_market_yes") is not None))
print("yes_ask&no_ask:", sum(1 for o in obs if o.get("yes_ask") is not None and o.get("no_ask") is not None))
print("spread nonnull:", sum(1 for o in obs if o.get("spread") is not None))
print("fee_rate nonnull:", sum(1 for o in obs if o.get("fee_rate") is not None))
print("fee_source:", dict(collections.Counter(o.get("fee_source") for o in obs)))
print("observed_cost nonnull:", sum(1 for o in obs if o.get("observed_cost_per_share") is not None))
print("candidate_side:", dict(collections.Counter(o.get("candidate_side") for o in obs)))
print("legacy_action:", dict(collections.Counter(o.get("legacy_action") for o in obs)))
print("legacy_ask nonnull:", sum(1 for o in obs if o.get("legacy_ask") is not None))
print("execution_role:", dict(collections.Counter(o.get("execution_role") for o in obs)))
# ask vs 1-no consistency where both present
import statistics
diffs = [abs(o["yes_ask"] + o["no_ask"] - 1.0) for o in obs if o.get("yes_ask") is not None and o.get("no_ask") is not None]
print("both-quotes n:", len(diffs), "mean|yes+no-1|:", round(statistics.mean(diffs), 4) if diffs else None)
