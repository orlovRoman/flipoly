import uuid
from sqlalchemy import Column, Integer, String, Float, DateTime, Index, Text, Numeric, ForeignKey, JSON, text, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from polyflip.db.models import Base

class ExecutionRequest(Base):
    __tablename__ = "execution_requests"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    idempotency_key = Column(String(128), unique=True, nullable=True)
    requested_mode = Column(String(32), nullable=False, default="PAPER")
    trade_history_id = Column(Integer, ForeignKey("trade_history.id", ondelete="RESTRICT"), nullable=True)
    intent = Column(String(32), nullable=False) # 'OPEN', 'CLOSE'
    trigger_reason = Column(String(32), nullable=True) # 'STRATEGY', 'STOP_LOSS', 'TAKE_PROFIT', 'MANUAL', 'RECOVERY'
    market_id = Column(String(128), nullable=False)
    asset = Column(String(32), nullable=False)
    
    # Order parameters
    outcome_to_buy = Column(String(16), nullable=False)
    requested_shares = Column(Numeric(38, 18), nullable=True)
    target_amount_usdc = Column(Numeric(38, 18), nullable=False)
    max_slippage_pct = Column(Float, nullable=False)
    ttl_seconds = Column(Integer, nullable=False, default=60)
    
    # State tracking
    state = Column(String(32), nullable=False, default='READY')
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)
    error_reason = Column(Text, nullable=True)
    
    # Outcome tracking
    filled_shares = Column(Numeric(38, 18), nullable=False, default=0)
    filled_cost_usdc = Column(Numeric(38, 18), nullable=False, default=0)

    __table_args__ = (
        Index(
            "uq_active_open_request",
            "market_id",
            unique=True,
            postgresql_where=text("intent = 'OPEN' AND state IN ('READY', 'CLAIMED', 'SUBMITTING', 'ACCEPTED', 'UNKNOWN', 'PARTIALLY_FILLED', 'RECONCILING')"),
            sqlite_where=text("intent = 'OPEN' AND state IN ('READY', 'CLAIMED', 'SUBMITTING', 'ACCEPTED', 'UNKNOWN', 'PARTIALLY_FILLED', 'RECONCILING')")
        ),
        Index(
            "uq_active_close_request",
            "trade_history_id",
            unique=True,
            postgresql_where=text("intent = 'CLOSE' AND state IN ('READY', 'CLAIMED', 'SUBMITTING', 'ACCEPTED', 'UNKNOWN', 'PARTIALLY_FILLED', 'RECONCILING')"),
            sqlite_where=text("intent = 'CLOSE' AND state IN ('READY', 'CLAIMED', 'SUBMITTING', 'ACCEPTED', 'UNKNOWN', 'PARTIALLY_FILLED', 'RECONCILING')")
        ),
        CheckConstraint(
            "(intent = 'OPEN' AND trade_history_id IS NULL) OR (intent = 'CLOSE' AND trade_history_id IS NOT NULL)",
            name="ck_execution_request_trade_reference",
        ),
        CheckConstraint(
            "requested_mode IN ('PAPER', 'SHADOW', 'LIVE')",
            name="ck_execution_request_mode",
        ),
        CheckConstraint(
            "requested_shares IS NULL OR requested_shares > 0",
            name="ck_execution_request_positive_shares",
        ),
    )

class ExecutionAttempt(Base):
    __tablename__ = "execution_attempts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id = Column(UUID(as_uuid=True), ForeignKey("execution_requests.id"), nullable=False)
    gateway = Column(String(32), nullable=False) # 'POLYMARKET', 'SHADOW', 'FAKE'
    started_at = Column(DateTime(timezone=True), nullable=False)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(32), nullable=False, default='IN_PROGRESS') # 'SUCCESS', 'FAILED', 'UNKNOWN'
    tx_hash = Column(String(128), nullable=True)
    error_msg = Column(Text, nullable=True)
    raw_response = Column(JSON().with_variant(JSONB, "postgresql"), nullable=True)

class ExecutionFill(Base):
    __tablename__ = "execution_fills"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    attempt_id = Column(UUID(as_uuid=True), ForeignKey("execution_attempts.id"), nullable=False)
    price = Column(Numeric(38, 18), nullable=False)
    shares = Column(Numeric(38, 18), nullable=False)
    fee_usdc = Column(Numeric(38, 18), nullable=False, default=0)
    timestamp = Column(DateTime(timezone=True), nullable=False)

class ExecutionApproval(Base):
    __tablename__ = "execution_approvals"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id = Column(UUID(as_uuid=True), ForeignKey("execution_requests.id"), nullable=False)
    approval_hash = Column(String(128), nullable=False, unique=True)
    status = Column(String(32), nullable=False, default='PENDING') # 'PENDING', 'CONSUMED', 'EXPIRED'
    created_at = Column(DateTime(timezone=True), nullable=False)
    consumed_at = Column(DateTime(timezone=True), nullable=True)
    consumed_by_ip = Column(String(64), nullable=True)

class ExecutionEvent(Base):
    __tablename__ = "execution_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(UUID(as_uuid=True), ForeignKey("execution_requests.id"), nullable=False)
    event_type = Column(String(64), nullable=False)
    payload = Column(JSON().with_variant(JSONB, "postgresql"), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False)

class ExposureReservation(Base):
    __tablename__ = "exposure_reservations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    asset = Column(String(32), nullable=False)
    amount_usdc = Column(Numeric(38, 18), nullable=False)
    reserved_at = Column(DateTime(timezone=True), nullable=False)
    released_at = Column(DateTime(timezone=True), nullable=True)
    request_id = Column(UUID(as_uuid=True), ForeignKey("execution_requests.id"), nullable=True)

class ChainTransaction(Base):
    __tablename__ = "chain_transactions"
    
    tx_hash = Column(String(128), primary_key=True)
    attempt_id = Column(UUID(as_uuid=True), ForeignKey("execution_attempts.id"), nullable=True)
    operation = Column(String(32), nullable=False)  # APPROVE, SPLIT, MERGE, REDEEM, SETTLEMENT
    network = Column(String(32), nullable=False)
    gas_paid_native = Column(Numeric(38, 18), nullable=True)
    gas_paid_usdc = Column(Numeric(38, 18), nullable=True)
    paid_by = Column(String(16), nullable=False)  # USER, RELAYER, UNKNOWN
    confirmed_at = Column(DateTime(timezone=True), nullable=True)
