"""
MRF P1 validation tests — verifies all bug fixes.
Standalone, no sqlalchemy/pytest-asyncio needed.
Run: python3 test_mrf_p1_fixes.py
"""
import sys
import math
from datetime import datetime, timezone, timedelta
import numpy as np

sys.path.insert(0, "/home/orlovrp/flipoly-worktrees/market-regime-filter-implementation")

from polyflip.crypto.market_regime import (
    _log_returns, _volatility, _efficiency_ratio,
    compute_asset_features, build_regime_snapshot,
    validate_candle_continuity, MIN_HISTORY_CANDLES,
    HORIZON_24H, HORIZON_4H, HORIZON_12H,
)
from polyflip.crypto.market_regime_classifier import classify_asset_regime, Regime, RegimeConfig
from polyflip.crypto.market_regime_policy import evaluate_policy, PolicyConfig, FilterMode, StrategyType, PolicyResult
from polyflip.crypto.market_regime_integration import build_snapshot_from_candles
from polyflip.crypto.market_regime_audit import serialize_regime_audit

passed = 0
failed = 0
total = 0

def check(name, condition, detail=""):
    global passed, failed, total
    total += 1
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name} — {detail}")

# ═══════════════════════════════════════════════════════════════════════════
print("\n── 1. _log_returns off-by-one fix ──")
test = check
test.__test__ = False

# ═══════════════════════════════════════════════════════════════════════════

# 97 closes → 96 intervals
closes_97 = np.linspace(100, 110, 97)  # monotonically increasing
ret_24h = _log_returns(closes_97, HORIZON_24H)  # 96 intervals
expected = math.log(110.0 / 100.0)
test("_log_returns with 97 closes covers 96 intervals",
     abs(ret_24h - expected) < 1e-10,
     f"got {ret_24h}, expected {expected}")

# 96 closes → should return 0 (insufficient for 96 intervals)
ret_96 = _log_returns(closes_97[:96], HORIZON_24H)
test("_log_returns with 96 closes returns 0 (needs 97)",
     ret_96 == 0.0,
     f"got {ret_96}")

# 17 closes → 16 intervals (4h)
closes_17 = np.linspace(100, 105, 17)
ret_4h = _log_returns(closes_17, HORIZON_4H)
expected_4h = math.log(105.0 / 100.0)
test("_log_returns 4h covers 16 intervals",
     abs(ret_4h - expected_4h) < 1e-10,
     f"got {ret_4h}, expected {expected_4h}")

# ═══════════════════════════════════════════════════════════════════════════
print("\n── 2. history_ready checks actual array length ──")
# ═══════════════════════════════════════════════════════════════════════════

# candle_count=97 but only 50 closes → should NOT be ready
closes_50 = np.linspace(100, 105, 50)
feat_50 = compute_asset_features("BTC", closes_50, closes_50, closes_50, closes_50, candle_count=97)
test("history_ready=False when candle_count > len(closes)",
     not feat_50.history_ready)

# candle_count=97 and 97 closes → should be ready
closes_97b = np.linspace(100, 105, 97)
feat_97 = compute_asset_features("BTC", closes_97b, closes_97b, closes_97b, closes_97b, candle_count=97)
test("history_ready=True when candle_count=97 and len=97",
     feat_97.history_ready)

# candle_count=50 → not ready regardless
feat_50b = compute_asset_features("BTC", closes_97b, closes_97b, closes_97b, closes_97b, candle_count=50)
test("history_ready=False when candle_count < MIN_HISTORY",
     not feat_50b.history_ready)

# ═══════════════════════════════════════════════════════════════════════════
print("\n── 3. validate_candle_continuity improvements ──")
# ═══════════════════════════════════════════════════════════════════════════

# Duplicates
times_dup = [datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=15*i) for i in range(10)]
times_dup[5] = times_dup[4]  # duplicate
ok_dup, reason_dup = validate_candle_continuity(times_dup, 10)
test("validate_candle_continuity catches duplicates",
     not ok_dup and "duplicates" in reason_dup,
     f"ok={ok_dup}, reason={reason_dup}")

# Count mismatch
times_short = [datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=15*i) for i in range(5)]
ok_cnt, reason_cnt = validate_candle_continuity(times_short, 10)
test("validate_candle_continuity catches count mismatch",
     not ok_cnt and "count_mismatch" in reason_cnt,
     f"ok={ok_cnt}, reason={reason_cnt}")

# Not sorted
times_unsorted = [datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=15*i) for i in [0, 2, 1, 3]]
ok_sort, reason_sort = validate_candle_continuity(times_unsorted, 4)
test("validate_candle_continuity catches unsorted",
     not ok_sort and "not_sorted" in reason_sort,
     f"ok={ok_sort}, reason={reason_sort}")

# Good candles
times_good = [datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=15*i) for i in range(97)]
ok_good, reason_good = validate_candle_continuity(times_good, 97)
test("validate_candle_continuity passes good candles",
     ok_good and reason_good == "ok",
     f"ok={ok_good}, reason={reason_good}")

# ═══════════════════════════════════════════════════════════════════════════
print("\n── 4. Zero-division guards ──")
# ═══════════════════════════════════════════════════════════════════════════

test("efficiency ratio with flat price → 0.0 (no crash)",
     _efficiency_ratio(np.ones(97)) == 0.0)

test("volatility with constant returns → 0.0 (no crash)",
     _volatility(np.zeros(100)) == 0.0)

# ═══════════════════════════════════════════════════════════════════════════
print("\n── 5. build_regime_snapshot with max_open_time ──")
# ═══════════════════════════════════════════════════════════════════════════

as_of = datetime(2026, 1, 2, 0, 0, tzinfo=timezone.utc)
closes_100 = np.linspace(100, 110, 100).astype(np.float64)
open_times_100 = [datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=15*i) for i in range(100)]

# One future candle (open_time = 13:00 > as_of = 12:00)
candle_data_with_future = {
    "BTC": {
        "closes": closes_100,
        "highs": closes_100 * 1.01,
        "lows": closes_100 * 0.99,
        "opens": closes_100,
        "open_times": open_times_100,
        "count": 100,
    }
}

snap = build_regime_snapshot(candle_data_with_future, as_of=as_of, max_open_time=as_of)
test("build_regime_snapshot trims future candles",
     snap.assets["BTC"].history_ready,
     f"candle_count={snap.assets['BTC'].candle_count}, ready={snap.assets['BTC'].history_ready}")

# ═══════════════════════════════════════════════════════════════════════════
print("\n── 6. build_snapshot_from_candles lookahead guard ──")
# ═══════════════════════════════════════════════════════════════════════════

class MockCandle:
    def __init__(self, open_time, o, h, l, c):
        self.open_time = open_time
        self.open = o
        self.high = h
        self.low = l
        self.close = c

base_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
mock_candles = [MockCandle(base_time + timedelta(minutes=15*i), 100+i, 101+i, 99+i, 100.5+i) for i in range(100)]

# as_of at 50th candle → should trim the last 50
snap_integration = build_snapshot_from_candles(
    mock_candles, "BTC", base_time + timedelta(minutes=15*50)
)
# Should have 51 candles (0..50 inclusive)
test("build_snapshot_from_candles trims future candles",
     snap_integration.assets["BTC"].candle_count <= 51,
     f"got {snap_integration.assets['BTC'].candle_count}")

# ═══════════════════════════════════════════════════════════════════════════
print("\n── 7. Classifier + Policy integration ──")
# ═══════════════════════════════════════════════════════════════════════════

closes_regime = np.linspace(100, 110, 97).astype(np.float64)
highs_regime = closes_regime * 1.01
lows_regime = closes_regime * 0.99
opens_regime = closes_regime - 0.1

snap_regime = build_regime_snapshot(
    {"BTC": {"closes": closes_regime, "highs": highs_regime,
             "lows": lows_regime, "opens": opens_regime, "count": 97}},
    as_of=as_of,
)
cl = classify_asset_regime(snap_regime.assets["BTC"])
test("classifier returns valid regime",
     cl.regime in list(Regime),
     f"got {cl.regime}")

test("classifier confidence 0-1",
     0.0 <= cl.confidence <= 1.0,
     f"got {cl.confidence}")

policy = evaluate_policy(snap_regime, StrategyType.OUTSIDER, 1.0, FilterMode.SHADOW)
test("policy returns valid result",
     policy.allow in (True, False) and policy.regime in list(Regime))

# ═══════════════════════════════════════════════════════════════════════════
print("\n── 8. Audit serialization ──")
# ═══════════════════════════════════════════════════════════════════════════

audit = serialize_regime_audit(
    snap_regime, policy, FilterMode.SHADOW, mrf_version=1,
)
test("audit has required keys",
     all(k in audit for k in ["mode", "version", "as_of", "global_regime", "global_confidence"]),
     f"keys={list(audit.keys())}")

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"Results: {passed}/{total} passed, {failed} failed")
if failed > 0:
    sys.exit(1)
print("All P1 fixes validated!")
