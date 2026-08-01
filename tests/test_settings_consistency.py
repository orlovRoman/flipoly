"""
Проверяет что ключи Redis, читаемые в backtest.js, совпадают
с ключами, сохраняемыми в trading.js (нет дрейфа имён).

Это статический анализ файлов — не требует запуска браузера.
"""
import re
from pathlib import Path
import pytest

BACKTEST_JS  = Path("polyflip/static/backtest.js")
TRADING_JS   = Path("polyflip/static/js/trading.js")


def _extract_redis_keys_read(js_text: str) -> set[str]:
    """Ключи вида s.KEY_NAME в applyLiveSettings / loadSettings."""
    return set(re.findall(r's\.([A-Z_]{3,})', js_text))


def _extract_redis_keys_written(js_text: str) -> set[str]:
    """Ключи вида settingsToSave.KEY_NAME = ..."""
    return set(re.findall(r'settingsToSave\.([A-Z_]{3,})\s*=', js_text))


class TestSettingsConsistency:
    def test_backtest_js_exists(self):
        assert BACKTEST_JS.exists(), f"{BACKTEST_JS} not found"

    def test_trading_js_exists(self):
        assert TRADING_JS.exists(), f"{TRADING_JS} not found"

    def test_no_division_by_100_for_threshold_fields(self):
        """
        После фикса c71ae984: TRADE_NO_FLIP_THRESHOLD, FLIP_THRESHOLD,
        MIN_EDGE, MAX_BET_EDGE должны читаться as-is (без / 100).
        """
        text = BACKTEST_JS.read_text(encoding="utf-8")

        # Проверяем что для этих полей нет / 100 в одной строке
        threshold_fields = [
            "TRADE_NO_FLIP_THRESHOLD",
            "FLIP_THRESHOLD",
            "MIN_EDGE",
            "MAX_BET_EDGE",
        ]
        for field in threshold_fields:
            # Находим строки где используется это поле
            lines = [l for l in text.splitlines() if field in l]
            for line in lines:
                assert "/ 100" not in line and "/100" not in line, (
                    f"Field {field} should NOT be divided by 100 after fix, "
                    f"but found in line: {line.strip()}"
                )

    def test_falsy_zero_safe_for_threshold_fields(self):
        """
        TRADE_NO_FLIP_THRESHOLD, MIN_EDGE должны использовать !== undefined
        или != null, а не ||, чтобы не терять значение 0.
        """
        text = BACKTEST_JS.read_text(encoding="utf-8")
        risky_fields = ["TRADE_NO_FLIP_THRESHOLD", "MIN_EDGE", "MAX_BET_EDGE"]

        for field in risky_fields:
            lines = [l for l in text.splitlines() if field in l and "||" in l]
            assert len(lines) == 0, (
                f"Field {field} uses || (falsy check) — risky for value=0. "
                f"Use !== undefined or != null instead. Lines: {lines}"
            )

    def test_favorite_threshold_not_using_or_operator(self):
        """cfg-fav-thresh должен использовать != null, а не ||."""
        text = BACKTEST_JS.read_text(encoding="utf-8")
        lines = [l for l in text.splitlines()
                 if "cfg-fav-thresh" in l and "||" in l]
        assert len(lines) == 0, (
            "cfg-fav-thresh still uses || operator — "
            "replace with != null check. Lines: " + str(lines)
        )

    def test_trading_js_normalization_comment_present(self):
        """Комментарий нормализации должен быть в trading.js."""
        text = TRADING_JS.read_text(encoding="utf-8")
        lines_with_field = [l for l in text.splitlines() if "FAVORITE_MIN_EDGE" in l]
        assert any("/ 100" in l for l in lines_with_field), (
            "FAVORITE_MIN_EDGE normalization (/ 100) not found in trading.js. "
            f"Lines with field: {lines_with_field}"
        )
