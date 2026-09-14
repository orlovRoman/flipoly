"""Slice test: 08-03 load -> decisions -> quotes. Must pass before full build."""
import sys
import time

sys.path.insert(0, r"D:\lgbm-audit-v1\code\shared")
import load
import decisions
import quotes

t0 = time.time()
f = load.read_funnel_day("2026-08-03")
s = load.read_snaps_day("2026-08-03")
print("funnel rows=%d snaps rows=%d (%.1fs)" % (len(f), len(s), time.time() - t0))
assert f["id"].is_unique, "funnel id not unique"

t0 = time.time()
ev = decisions.build_events(f)
print("events=%d slots=%s (%.1fs)" % (
    len(ev), ev["model_slot"].value_counts().head(4).to_dict(), time.time() - t0))
assert len(ev) == len(f)
assert ev["decision_event_id"].is_unique

t0 = time.time()
out, stats = quotes.attach_quotes(ev, s)
print("quote stats:", stats, "(%.1fs)" % (time.time() - t0))
assert len(out) == len(ev)
assert stats["matched"] > 0
print("SLICE-OK")
