from unittest.mock import patch
import pytest

from polyflip.research.lp_rewards.watchdog import SystemWatchdog


def test_watchdog_disk_guard_status_levels():
    watchdog = SystemWatchdog(
        storage_path=r"D:\flipoly-research\lp-rewards",
        warning_threshold_gb=30.0,
        halt_threshold_gb=20.0,
    )

    # 1. Healthy disk: 50 GB free
    with patch("shutil.disk_usage") as mock_usage:
        mock_usage.return_value = type("Usage", (), {"free": 50 * (1024 ** 3)})()
        free, status = watchdog.check_disk_space()
        assert status == "OK"
        assert watchdog.is_halted is False

    # 2. Warning level: 25 GB free
    with patch("shutil.disk_usage") as mock_usage:
        mock_usage.return_value = type("Usage", (), {"free": 25 * (1024 ** 3)})()
        free, status = watchdog.check_disk_space()
        assert status == "WARNING"
        assert watchdog.is_halted is False

    # 3. Emergency halt: 15 GB free (< 20 GB)
    with patch("shutil.disk_usage") as mock_usage:
        mock_usage.return_value = type("Usage", (), {"free": 15 * (1024 ** 3)})()
        free, status = watchdog.check_disk_space()
        assert status == "EMERGENCY_HALT"
        assert watchdog.is_halted is True
