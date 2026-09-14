import logging
import os
from pathlib import Path
import shutil
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class SystemWatchdog:
    """Monitors disk space on the target storage drive and manages emergency stops."""

    def __init__(
        self,
        storage_path: str = r"D:\flipoly-research\lp-rewards",
        warning_threshold_gb: float = 30.0,
        halt_threshold_gb: float = 20.0,
    ):
        self.storage_path = Path(storage_path)
        self.warning_threshold_gb = warning_threshold_gb
        self.halt_threshold_gb = halt_threshold_gb
        self.is_halted = False

    def check_disk_space(self) -> Tuple[float, str]:
        """Check free disk space in GB.

        Returns (free_gb, status) where status is 'OK', 'WARNING', or 'EMERGENCY_HALT'.
        """
        try:
            target = self.storage_path if self.storage_path.exists() else self.storage_path.anchor
            usage = shutil.disk_usage(target)
            free_gb = usage.free / (1024 ** 3)
        except Exception as e:
            logger.error(f"Error checking disk usage for {self.storage_path}: {e}")
            return 0.0, "ERROR"

        if free_gb < self.halt_threshold_gb:
            self.is_halted = True
            logger.critical(
                f"EMERGENCY DISK HALT: Only {free_gb:.2f} GB free on {target} "
                f"(limit: {self.halt_threshold_gb} GB). Flushing and halting."
            )
            return free_gb, "EMERGENCY_HALT"
        elif free_gb < self.warning_threshold_gb:
            logger.warning(
                f"DISK SPACE WARNING: {free_gb:.2f} GB free on {target} "
                f"(warning threshold: {self.warning_threshold_gb} GB)."
            )
            return free_gb, "WARNING"

        return free_gb, "OK"
