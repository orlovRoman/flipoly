"""Static guardrails for the Stage 9 optimizer UI (no database/server calls)."""

from pathlib import Path
import re


ROOT = Path(__file__).parents[1]
HTML = ROOT / "polyflip" / "templates" / "optimizer.html"
JS = ROOT / "polyflip" / "static" / "js" / "optimizer.js"


def test_stage9_banner_and_read_only_sections_are_present():
    html = HTML.read_text(encoding="utf-8")
    assert "RESEARCH / PAPER" in html
    assert "RESEARCH/PAPER: LIVE-активация отключена" in html
    for section in ("runs", "timeline", "candidates", "shadow", "deployments", "permissions", "errors", "audit"):
        assert f'id="tab-{section}"' in html


def test_optimizer_requests_use_api_base_and_auth_headers():
    js = JS.read_text(encoding="utf-8")
    assert "function getAuthHeaders" in js
    assert all("window.API_BASE" in line for line in js.splitlines() if "fetch(" in line)
    assert js.count("getAuthHeaders()") >= js.count("fetch(")


def test_stage9_has_no_visible_live_mutation_controls():
    html = HTML.read_text(encoding="utf-8")
    assert "LIVE-активация отключена" in html
    assert "Аварийный откат (Rollback)" not in html
    assert "Утвердить и Активировать в LIVE" not in html
