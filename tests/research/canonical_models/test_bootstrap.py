from datetime import datetime, timedelta, timezone

from polyflip.research.canonical_models.bootstrap import paired_day_bootstrap

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_paired_same_markets_and_days(tmp_path):
    rows = []
    for d in range(4):
        for i in range(3):
            rows.append({"market_id": f"d{d}m{i}",
                         "decision_at": T0 + timedelta(days=d),
                         "target_up": (i + d) % 2,
                         "A": 0.6, "B": 0.55})
    out = paired_day_bootstrap(rows, "A", "B", n_boot=50, workdir=str(tmp_path))
    assert out["paired"] is True and out["n_days"] == 4
    assert len(out["ci95"]) == 2
