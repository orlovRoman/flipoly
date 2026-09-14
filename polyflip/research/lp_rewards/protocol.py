from decimal import Decimal, getcontext
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml
from pydantic import BaseModel, Field

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


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_protocol(protocol_path: Optional[Path] = None) -> LPProtocol:
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
    return protocol
