from polyflip.research.canonical_models.challenger import pick_challenger
from polyflip.research.canonical_models.report import STATUSES, final_status


def test_lifecycle_statuses():
    assert "NOT_STARTED" in STATUSES  # final_start=null, pre-freeze marker
    assert "INCONCLUSIVE" in STATUSES  # thin data / wide CI verdict


def test_challenger_margin_and_no_clear():
    oof = {"M2": [(0.6, 1), (0.4, 0)], "M3": [(0.61, 1), (0.39, 0)],
           "M4": [(0.9, 0), (0.1, 1)]}
    d = pick_challenger(oof, min_margin=10.0)
    assert d.challenger == "NO_CLEAR_CHALLENGER"
    d2 = pick_challenger(oof, min_margin=0.0)
    assert d2.challenger in ("M2", "M3", "M4")


def test_status_computed_not_fitted():
    assert final_status({"n_canonical": 0}) == "DATA_BLOCKED"
    assert final_status({"n_canonical": 10, "final_has_new_data": False}) == "PENDING_NEW_PERIOD"
    assert final_status({"n_canonical": 10, "challenger_margin_ok": False,
                         "forecast_better": True}) == "NO_INCREMENTAL_VALUE"
    assert final_status({"n_canonical": 10, "challenger_margin_ok": True,
                         "forecast_better": True, "net_pnl": -1}) == "FORECAST_ONLY"
    assert final_status({"n_canonical": 10, "challenger_margin_ok": True,
                         "forecast_better": True, "net_pnl": 5,
                         "ci_lo": 1, "ci_hi": 9,
                         "max_drawdown_ok": True,
                         "concentrated_ok": True}) == "PAPER_CANDIDATE"
    assert final_status({"n_canonical": 10, "challenger_margin_ok": True,
                         "forecast_better": True, "net_pnl": 5,
                         "ci_lo": -1, "ci_hi": 9}) == "PROMISING_UNCERTAIN"
