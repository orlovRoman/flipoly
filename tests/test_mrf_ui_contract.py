from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_trading_template_contains_mrf_v3_controls():
    html = (ROOT / "polyflip" / "templates" / "trading.html").read_text(
        encoding="utf-8"
    )

    assert 'id="MARKET_REGIME_VETO_THRESHOLD"' in html
    assert 'id="MARKET_REGIME_EDGE_OVERRIDE_MARGIN"' in html
    assert 'id="MARKET_REGIME_ASSET_WEIGHT"' in html
    assert 'id="MARKET_REGIME_GLOBAL_WEIGHT"' in html
    assert 'id="mrf-v3-settings"' in html
    assert 'id="MARKET_REGIME_MIN_HISTORY" value="97"' in html
    assert 'min="97"' in html


def test_trading_js_loads_and_saves_mrf_v3_settings():
    js = (ROOT / "polyflip" / "static" / "js" / "trading.js").read_text(
        encoding="utf-8"
    )

    assert "MARKET_REGIME_VETO_THRESHOLD" in js
    assert "MARKET_REGIME_EDGE_OVERRIDE_MARGIN" in js
    assert "MARKET_REGIME_ASSET_WEIGHT" in js
    assert "MARKET_REGIME_GLOBAL_WEIGHT" in js
    assert "avg_regime_evidence" in js
    assert "mrfStatEvidence" in js
    assert "Legacy multiplier" in js
    assert "updateMrfVersionUI" in js
