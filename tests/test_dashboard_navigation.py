from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_references_existing_controller():
    template = (ROOT / "polyflip" / "templates" / "index.html").read_text(encoding="utf-8")
    assert "path='/js/app.js'" in template
    assert not "path='/js/dashboard.js'" in template
    assert (ROOT / "polyflip" / "static" / "js" / "app.js").is_file()


def test_dashboard_navigation_targets_tab_panes():
    template = (ROOT / "polyflip" / "templates" / "index.html").read_text(encoding="utf-8")
    source = (ROOT / "polyflip" / "static" / "js" / "app.js").read_text(encoding="utf-8")

    targets = set(re.findall(r'<li[^>]*class="nav-item[^"]*"[^>]*data-tab="([^"]+)"', template))
    pane_ids = set(re.findall(r'<section[^>]*id="([^"]+)"[^>]*class="tab-pane', template))

    assert targets == {"analytics", "models", "status"}
    assert {f"{target}-tab" for target in targets} <= pane_ids
    assert 'querySelectorAll(".tab-pane")' in source
    assert 'document.getElementById(`${targetId}-tab`)' in source


def test_dashboard_models_render_all_registry_columns_and_pnl():
    template = (ROOT / "polyflip" / "templates" / "index.html").read_text(encoding="utf-8")
    source = (ROOT / "polyflip" / "static" / "js" / "app.js").read_text(encoding="utf-8")
    assert "<th>Тип модели</th>" in template
    assert "<th>Действия</th>" in template
    assert "loadModelsPnLData" in source
    assert "kpi-total-models" in source
    assert "decision_threshold_down" in source



def test_dashboard_css_shows_active_tab_panes():
    css = (ROOT / "polyflip" / "static" / "css" / "style.css").read_text(encoding="utf-8")
    assert ".tab-pane {" in css
    assert ".tab-pane.active {" in css
    assert ".tab-content.active" not in css

