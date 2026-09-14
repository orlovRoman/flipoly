from decimal import Decimal, getcontext
import hashlib
import logging
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Ensure global precision is set to 28 digits minimum
getcontext().prec = 28


class HypothesisConfig(BaseModel):
    claim: str
    target_r100_calendar: Decimal
    confidence_level: Decimal
    currency: str


class CapitalAllocationConfig(BaseModel):
    allocated_working_capital: Decimal
    max_unhedged_per_market: Decimal
    max_unhedged_total: Decimal
    reserve_ratio_exit_buffer: Decimal
    live_wallet_buffer: Decimal


class UniverseConfig(BaseModel):
    rewards_endpoint: str
    page_size: int
    pagination_terminal_marker: str
    exclude_neg_risk: bool
    min_order_age_source: str
    ranking_refresh_interval_sec: int
    max_rotation_rate_per_15m: int
    active_markets_count: int
    reserve_markets_count: int


class ScoringConfig(BaseModel):
    formula_type: str
    payout_period: str
    epoch_samples_documented: int
    daily_projection_samples: int
    min_payout_dust_usdc: Decimal
    midpoint_range: Tuple[Decimal, Decimal]
    midpoint_fallback_allowed: bool
    uncertain_status_on_missing_side: str


class WSCollectorConfig(BaseModel):
    ws_endpoint: str
    ping_interval_sec: float
    pong_timeout_sec: float
    initial_dump: bool
    rest_reconciliation_interval_sec: float
    mismatch_status: str


class QuotingFSMConfig(BaseModel):
    order_type: str
    time_in_force: str
    quote_pair: List[str]
    max_combined_fill_cost: Decimal
    hedging_timeout_sec: float
    expiry_guard_buffer_sec: float
    forced_exit_order_type: str


class DataStorageConfig(BaseModel):
    root_path: str
    disk_warning_threshold_gb: float
    disk_emergency_halt_gb: float
    parquet_compression: str


class GateAConfig(BaseModel):
    min_calendar_days: int
    min_quote_hours: int
    min_active_markets: int
    min_market_coverage_ratio: Decimal
    max_single_market_pnl_share: Decimal
    bootstrap_samples: int
    verdicts: List[str]


class GateBConfig(BaseModel):
    min_live_days: int
    target_confirmed_threshold: Decimal
    max_drawdown_limit: Decimal
    max_reward_prediction_error: Decimal


class GatesConfig(BaseModel):
    gate_a: GateAConfig
    gate_b: GateBConfig


class LPProtocol(BaseModel):
    version: str
    protocol_id: str
    created_at: str
    hypothesis: HypothesisConfig
    capital_allocation: CapitalAllocationConfig
    universe: UniverseConfig
    scoring: ScoringConfig
    ws_collector: WSCollectorConfig
    quoting_fsm: QuotingFSMConfig
    data_storage: DataStorageConfig
    gates: GatesConfig
    sha256_hash: Optional[str] = None


WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:")
DEFAULT_LINUX_STORAGE_FALLBACK = "~/flipoly-research/lp-rewards"


def resolve_storage_root(
    configured_path: str,
    override: Optional[Union[str, Path]] = None,
    platform: Optional[str] = None,
) -> str:
    """Resolve the storage root path with cross-platform validation and environment overrides.

    Order of precedence:
    1. Explicit override argument (e.g. from CLI --storage-root)
    2. Environment variable LP_STORAGE_ROOT (or LP_STORAGE_PATH)
    3. Configured path from protocol YAML. If running on a non-Windows platform and the path
       contains Windows drive syntax (e.g. 'D:\\...'), it falls back to a safe Unix path
       (defaulting to '~/flipoly-research/lp-rewards') to avoid creating malformed directories.
    """
    if override is not None and str(override).strip():
        return str(override).strip()

    env_root = os.environ.get("LP_STORAGE_ROOT") or os.environ.get("LP_STORAGE_PATH")
    if env_root is not None and env_root.strip():
        return env_root.strip()

    target_platform = platform if platform is not None else sys.platform
    is_windows = (target_platform == "win32" or (platform is None and os.name == "nt"))

    if not is_windows and WINDOWS_DRIVE_PATTERN.match(configured_path):
        fallback = os.path.expanduser(DEFAULT_LINUX_STORAGE_FALLBACK)
        logger.warning(
            f"Windows storage path '{configured_path}' detected on non-Windows platform '{target_platform}'. "
            f"Falling back to safe Linux path '{fallback}' to prevent creating invalid directories. "
            f"Set LP_STORAGE_ROOT or use --storage-root to override."
        )
        return fallback

    return configured_path


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_protocol(
    protocol_path: Optional[Path] = None,
    storage_root: Optional[Union[str, Path]] = None,
) -> LPProtocol:
    if protocol_path is None:
        # Default path relative to flipoly repository root
        base = Path(__file__).resolve().parents[3]
        protocol_path = base / "artifacts" / "research" / "lp_rewards" / "protocol_v0.1.yaml"

    if not protocol_path.exists():
        raise FileNotFoundError(f"Protocol file not found at {protocol_path}")

    content_hash = compute_file_sha256(protocol_path)
    with open(protocol_path, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f)

    protocol = LPProtocol(**raw_data)
    protocol.sha256_hash = content_hash

    # Cross-platform validation and storage root override
    protocol.data_storage.root_path = resolve_storage_root(
        configured_path=protocol.data_storage.root_path,
        override=storage_root,
    )
    return protocol
