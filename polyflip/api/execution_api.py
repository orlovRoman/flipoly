from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from typing import Literal, Optional
from sqlalchemy import select, func
from datetime import datetime, timezone, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
import uuid
import structlog

from polyflip.db.connection import get_db_session
from polyflip.api.auth import verify_api_key
from polyflip.db.models import RuntimeSettings, TradeHistory
from polyflip.db.execution_models import (
    ExecutionWorkerStatus,
    ExecutionRequest,
    ExecutionEvent,
    ExecutionAttempt,
    LiveMirrorCandidate,
)
from polyflip.execution.config import ExecutionSettings

logger = structlog.get_logger(__name__)

router = APIRouter(
    prefix="/api/execution", tags=["Execution"], dependencies=[Depends(verify_api_key)]
)


class KillSwitchRequest(BaseModel):
    enabled: bool


@router.get("/status")
async def get_live_trading_status(db: AsyncSession = Depends(get_db_session)):
    """
    Returns live trading status, mirror status, and worker status for PAPER, SHADOW, and LIVE modes.
    """
    settings = ExecutionSettings()

    async def get_worker_dict(mode_name: str):
        ws = (
            await db.execute(
                select(ExecutionWorkerStatus)
                .where(ExecutionWorkerStatus.execution_mode == mode_name)
                .order_by(ExecutionWorkerStatus.heartbeat_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if not ws:
            return None
        return {
            "worker_id": ws.worker_id,
            "execution_mode": ws.execution_mode,
            "heartbeat_at": ws.heartbeat_at.isoformat() if ws.heartbeat_at else None,
            "gateway_ready": ws.gateway_ready,
            "credentials_loaded": ws.credentials_loaded,
            "wallet_address": ws.wallet_address,
            "balance_usdc": (
                float(ws.balance_usdc) if ws.balance_usdc is not None else None
            ),
            "collateral_allowance_ready": ws.collateral_allowance_ready,
            "conditional_allowance_ready": ws.conditional_allowance_ready,
            "last_error_code": ws.last_error_code,
            "last_error_message": ws.last_error_message,
        }

    paper_worker = await get_worker_dict("PAPER")
    shadow_worker = await get_worker_dict("SHADOW")
    live_worker = await get_worker_dict("LIVE")

    async def _flag(key: str) -> bool:
        row = (
            await db.execute(select(RuntimeSettings).where(RuntimeSettings.key == key))
        ).scalar_one_or_none()
        return row is not None and row.value.lower() == "true"

    async def _flag_str(key: str, default: str) -> str:
        row = (
            await db.execute(select(RuntimeSettings).where(RuntimeSettings.key == key))
        ).scalar_one_or_none()
        return row.value if row else default

    live_mirror_enabled = await _flag("LIVE_MIRROR_ENABLED")
    live_release_mode = await _flag_str("LIVE_RELEASE_MODE", "DISABLED")
    live_trading_enabled = await _flag("LIVE_TRADING_ENABLED")

    # –ö–æ–ª–∏—á–µ—Å—Ç–≤–æ –∫–∞–Ω–¥–∏–¥–∞—Ç–æ–≤ –ø–æ —Å–æ—Å—Ç–æ—è–Ω–∏—è–º –¥–ª—è —Ä–µ–∂–∏–º–∞ LIVE
    candidate_counts = {}
    for state in ("NEW", "ELIGIBLE", "REJECTED", "RELEASED"):
        cnt = await db.scalar(
            select(func.count())
            .select_from(LiveMirrorCandidate)
            .where(
                LiveMirrorCandidate.state == state,
                LiveMirrorCandidate.target_mode == "LIVE",
            )
        )
        candidate_counts[state] = cnt or 0

    return {
        "live_trading_enabled": live_trading_enabled,
        "execution_mode": settings.execution_mode.value,
        "kill_switch_available": live_worker is not None,
        "paper_worker": paper_worker,
        "shadow_worker": shadow_worker,
        "live_worker": live_worker,
        "worker_status": live_worker,
        "live_mirror_enabled": live_mirror_enabled,
        "live_release_mode": live_release_mode,
        "mirror_candidates": candidate_counts,
    }


@router.put("/kill-switch")
async def toggle_kill_switch(
    payload: KillSwitchRequest, db: AsyncSession = Depends(get_db_session)
):
    """
    –£–ø—Ä–∞–≤–ª—è–µ—Ç —ç–∫—Å—Ç—Ä–µ–Ω–Ω—ã–º –≤—ã–∫–ª—é—á–µ–Ω–∏–µ–º LIVE-—Ç–æ—Ä–≥–æ–≤–ª–∏.
    –í–∫–ª—é—á–µ–Ω–∏–µ LIVE –≤—ã–ø–æ–ª–Ω—è–µ—Ç—Å—è –∏—Å–∫–ª—é—á–∏—Ç–µ–ª—å–Ω–æ —á–µ—Ä–µ–∑ –∞–∫—Ç–∏–≤–∞—Ü–∏—é LIVE-—Å–µ—Å—Å–∏–∏.
    """
    if payload.enabled:
        raise HTTPException(
            status_code=409,
            detail="–í–∫–ª—é—á–µ–Ω–∏–µ LIVE –≤—ã–ø–æ–ª–Ω—è–µ—Ç—Å—è —Ç–æ–ª—å–∫–æ —á–µ—Ä–µ–∑ –∞–∫—Ç–∏–≤–∞—Ü–∏—é LIVE-—Å–µ—Å—Å–∏–∏",
        )

    key = "LIVE_TRADING_ENABLED"

    # –ó–∞—â–∏—â–∞–µ–º —á—Ç–µ–Ω–∏–µ/–∑–∞–ø–∏—Å—å –±–ª–æ–∫–∏—Ä–æ–≤–∫–æ–π FOR UPDATE
    stmt = select(RuntimeSettings).where(RuntimeSettings.key == key)
    bind = db.bind
    if bind and bind.dialect.name != "sqlite":
        stmt = stmt.with_for_update()

    existing = (await db.execute(stmt)).scalar_one_or_none()

    if payload.enabled:
        worker_status = (
            await db.execute(
                select(ExecutionWorkerStatus)
                .where(ExecutionWorkerStatus.execution_mode == "LIVE")
                .order_by(ExecutionWorkerStatus.heartbeat_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()

        if not worker_status:
            raise HTTPException(
                status_code=409,
                detail="Cannot enable LIVE trading: LIVE worker is not running (no status in DB)",
            )

        now = datetime.now(timezone.utc)
        hb_at = worker_status.heartbeat_at
        if hb_at and hb_at.tzinfo is None:
            hb_at = hb_at.replace(tzinfo=timezone.utc)

        if not hb_at or hb_at < now - timedelta(seconds=30):
            raise HTTPException(
                status_code=409,
                detail="Cannot enable LIVE trading: LIVE worker heartbeat is older than 30 seconds",
            )

        if not worker_status.gateway_ready:
            raise HTTPException(
                status_code=409,
                detail=f"Cannot enable LIVE trading: LIVE gateway is not ready ({worker_status.last_error_message})",
            )

        if float(worker_status.balance_usdc or 0) < 5:
            raise HTTPException(
                status_code=409,
                detail=f"Cannot enable LIVE trading: Insufficient USDC balance (Minimum $5, current {float(worker_status.balance_usdc or 0)})",
            )

        if not worker_status.collateral_allowance_ready:
            raise HTTPException(
                status_code=409,
                detail="Cannot enable LIVE trading: Collateral allowance is not ready",
            )

    value = "true" if payload.enabled else "false"

    try:
        now = datetime.now(timezone.utc)
        if existing:
            existing.value = value
            existing.updated_at = now
            existing.updated_by = "api"
        else:
            db.add(
                RuntimeSettings(key=key, value=value, updated_at=now, updated_by="api")
            )

        await db.commit()
        logger.info("kill_switch_toggled", enabled=payload.enabled)
        return {"status": "ok", "live_trading_enabled": payload.enabled}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("kill_switch_error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal Server Error")


from polyflip.db.models import TradeHistory, LiveMarket
from polyflip.execution.states import ACTIVE_POSITION_STATES

# ---------------------------------------------------------------------------
# –≠—Ç–∞–ø 6: —É–ø—Ä–∞–≤–ª–µ–Ω–∏–µ —Ç—Ä–µ–º—è —Ä—É–±–∏–ª—å–Ω–∏–∫–∞–º–∏ LIVE-–∞—Ä—Ö–∏—Ç–µ–∫—Ç—É—Ä—ã
# ---------------------------------------------------------------------------

_BOOL_SWITCH_KEYS = {"LIVE_MIRROR_ENABLED", "LIVE_TRADING_ENABLED"}
_STR_SWITCH_KEYS = {"LIVE_RELEASE_MODE"}
_LIVE_RELEASE_MODE_VALUES = {"DISABLED", "MANUAL", "AUTO"}


class SwitchBoolRequest(BaseModel):
    enabled: bool


class SwitchReleaseModeRequest(BaseModel):
    mode: Literal["DISABLED", "MANUAL", "AUTO"]


async def _get_runtime_flag(db: AsyncSession, key: str, default: str = "false") -> str:
    row = (
        await db.execute(select(RuntimeSettings).where(RuntimeSettings.key == key))
    ).scalar_one_or_none()
    return row.value if row else default


async def _set_runtime_flag(db: AsyncSession, key: str, value: str) -> None:
    now = datetime.now(timezone.utc)
    existing = (
        await db.execute(select(RuntimeSettings).where(RuntimeSettings.key == key))
    ).scalar_one_or_none()
    if existing:
        existing.value = value
        existing.updated_at = now
        existing.updated_by = "api"
    else:
        db.add(RuntimeSettings(key=key, value=value, updated_at=now, updated_by="api"))
    await db.commit()


from polyflip.execution.live_mirror_worker import set_mirror_enabled


@router.put(
    "/mirror-switch", summary="–í–∫–ª—é—á–∏—Ç—å / –≤—ã–∫–ª—é—á–∏—Ç—å LIVE_MIRROR_ENABLED (mirror-–≤–æ—Ä–∫–µ—Ä)"
)
async def toggle_mirror_switch(
    payload: SwitchBoolRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    –†—É–±–∏–ª—å–Ω–∏–∫ 1: —É–ø—Ä–∞–≤–ª—è–µ—Ç LIVE_MIRROR_ENABLED.
    true  ‚Äî mirror-–≤–æ—Ä–∫–µ—Ä —Å–æ–∑–¥–∞—ë—Ç LiveMirrorCandidate –¥–ª—è FILLED PAPER OPEN.
    false ‚Äî –≤–æ—Ä–∫–µ—Ä —Å–ø–∏—Ç, –Ω–∏ –æ–¥–Ω–æ–≥–æ –∫–∞–Ω–¥–∏–¥–∞—Ç–∞ –Ω–µ —Å–æ–∑–¥–∞—ë—Ç.
    """
    try:
        await set_mirror_enabled(db, enabled=payload.enabled, updated_by="api")
        await db.commit()
        logger.info("mirror_switch_toggled", enabled=payload.enabled)
        return {"status": "ok", "LIVE_MIRROR_ENABLED": payload.enabled}
    except Exception:
        await db.rollback()
        raise


@router.put(
    "/release-mode", summary="–£—Å—Ç–∞–Ω–æ–≤–∏—Ç—å LIVE_RELEASE_MODE (DISABLED | MANUAL | AUTO)"
)
async def set_release_mode(
    payload: SwitchReleaseModeRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    –†—É–±–∏–ª—å–Ω–∏–∫ 2: —É–ø—Ä–∞–≤–ª—è–µ—Ç —Å–ø–æ—Å–æ–±–æ–º –≤—ã–ø—É—Å–∫–∞ –∫–∞–Ω–¥–∏–¥–∞—Ç–æ–≤.
    DISABLED ‚Äî release_gate —Å–ø–∏—Ç.
    MANUAL   ‚Äî release_gate –æ–∂–∏–¥–∞–µ—Ç —è–≤–Ω–æ–≥–æ /release-candidate —á–µ—Ä–µ–∑ API.
    AUTO     ‚Äî release_gate –∞–≤—Ç–æ–º–∞—Ç–∏—á–µ—Å–∫–∏ –≤—ã–ø—É—Å–∫–∞–µ—Ç NEW‚ÜíELIGIBLE –∫–∞–Ω–¥–∏–¥–∞—Ç–æ–≤.
    """
    await _set_runtime_flag(db, "LIVE_RELEASE_MODE", payload.mode)
    logger.info("release_mode_set", mode=payload.mode)
    return {"status": "ok", "LIVE_RELEASE_MODE": payload.mode}


@router.put(
    "/ignore-edge-decay", summary="–ò–≥–Ω–æ—Ä–∏—Ä–æ–≤–∞—Ç—å —Ñ–∏–ª—å—Ç—Ä EDGE_DECAYED_BEFORE_RELEASE"
)
async def toggle_ignore_edge_decay(
    payload: SwitchBoolRequest,
    request: Request,
    db: AsyncSession = Depends(get_db_session),
):
    """
    –û—Ç–∫–ª—é—á–∞–µ—Ç –∏–ª–∏ –≤–∫–ª—é—á–∞–µ—Ç –ø—Ä–æ–≤–µ—Ä–∫—É release_net_edge < combined_min_net_edge –≤ release_gate.
    """
    client_ip = request.client.host if request.client else "unknown"
    api_key = request.headers.get("X-API-Key") or "session"
    await _set_runtime_flag(db, "LIVE_IGNORE_EDGE_DECAY", str(payload.enabled).lower())
    logger.info(
        "ignore_edge_decay_toggled",
        enabled=payload.enabled,
        client_ip=client_ip,
        api_key_prefix=api_key[:8] if api_key else "unknown",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    return {"status": "ok", "LIVE_IGNORE_EDGE_DECAY": payload.enabled}


class SwitchOrderModeRequest(BaseModel):
    mode: Literal["MAKER_TTL", "GTC_TTL", "LIMIT_TTL", "FAK", "FAK_RETRY"]
    gtc_ttl_seconds: Optional[float] = Field(None, ge=1.0, le=60.0)
    fak_retry_max_attempts: Optional[int] = Field(None, ge=1, le=10)
    fak_retry_delay_sec: Optional[float] = Field(None, ge=0.1, le=5.0)
    paper_execution_profile: Literal["INSTANT", "LIVE_PARITY"] | None = None
    paper_live_delay_sec: Optional[float] = Field(None, ge=0.0, le=30.0)


@router.get("/order-mode", summary="–ü–æ–ª—É—á–∏—Ç—å —Ä–µ–∂–∏–º –∏ –ø–∞—Ä–∞–º–µ—Ç—Ä—ã –∏—Å–ø–æ–ª–Ω–µ–Ω–∏—è –æ—Ä–¥–µ—Ä–æ–≤")
async def get_order_mode(db: AsyncSession = Depends(get_db_session)):
    mode = await _get_runtime_flag(db, "LIVE_ORDER_MODE", default="MAKER_TTL")
    gtc_ttl = await _get_runtime_flag(db, "LIVE_GTC_TTL_SECONDS", default="10.0")
    retry_attempts = await _get_runtime_flag(db, "LIVE_FAK_RETRY_MAX_ATTEMPTS", default="3")
    retry_delay = await _get_runtime_flag(db, "LIVE_FAK_RETRY_DELAY_SEC", default="0.75")
    paper_profile = await _get_runtime_flag(db, "PAPER_EXECUTION_PROFILE", default="INSTANT")
    paper_delay = await _get_runtime_flag(db, "PAPER_LIVE_DELAY_SEC", default="2.0")
    return {
        "mode": mode.upper(),
        "gtc_ttl_seconds": float(gtc_ttl),
        "fak_retry_max_attempts": int(retry_attempts),
        "fak_retry_delay_sec": float(retry_delay),
        "paper_execution_profile": paper_profile.upper(),
        "paper_live_delay_sec": float(paper_delay),
    }


@router.put("/order-mode", summary="–£—Å—Ç–∞–Ω–æ–≤–∏—Ç—å —Ä–µ–∂–∏–º –∏ –ø–∞—Ä–∞–º–µ—Ç—Ä—ã –∏—Å–ø–æ–ª–Ω–µ–Ω–∏—è –æ—Ä–¥–µ—Ä–æ–≤")
async def set_order_mode(
    payload: SwitchOrderModeRequest,
    request: Request,
    db: AsyncSession = Depends(get_db_session),
):
    client_ip = request.client.host if request.client else "unknown"
    api_key = request.headers.get("X-API-Key") or "session"
    
    await _set_runtime_flag(db, "LIVE_ORDER_MODE", payload.mode.upper())
    if payload.gtc_ttl_seconds is not None:
        await _set_runtime_flag(db, "LIVE_GTC_TTL_SECONDS", str(payload.gtc_ttl_seconds))
    if payload.fak_retry_max_attempts is not None:
        await _set_runtime_flag(db, "LIVE_FAK_RETRY_MAX_ATTEMPTS", str(payload.fak_retry_max_attempts))
    if payload.fak_retry_delay_sec is not None:
        await _set_runtime_flag(db, "LIVE_FAK_RETRY_DELAY_SEC", str(payload.fak_retry_delay_sec))
    if payload.paper_execution_profile is not None:
        await _set_runtime_flag(db, "PAPER_EXECUTION_PROFILE", payload.paper_execution_profile.upper())
    if payload.paper_live_delay_sec is not None:
        await _set_runtime_flag(db, "PAPER_LIVE_DELAY_SEC", str(payload.paper_live_delay_sec))

    logger.info(
        "order_mode_changed",
        mode=payload.mode,
        gtc_ttl_seconds=payload.gtc_ttl_seconds,
        fak_retry_max_attempts=payload.fak_retry_max_attempts,
        fak_retry_delay_sec=payload.fak_retry_delay_sec,
        paper_execution_profile=payload.paper_execution_profile,
        paper_live_delay_sec=payload.paper_live_delay_sec,
        c◊}∑Í⁄$z{-ÆÈ‹j◊ù7F˜'íÊñBíÊ∆&V¬Ç'F˜F≈˜G&FW2"í¿–¢gVÊ2Á7V“Ä–¢66RÇÖ˜Ê≈ˆWá"‚¬í¬V«6UÛ”ê–¢íÊ∆&V¬Ç'vñÊÊñÊu˜G&FW2"í¿–¢gVÊ2Á7V“Ö˜Ê≈ˆWá"íÊ∆&V¬Ç'F˜F≈˜Ê¬"í¿–¢ê–¢ÁvÜW&RÇ¶6ˆÊG2ê–¢Êw&˜Wˆ'íÖG&FTÜó7F˜'íÊ76WBê–¢Ê˜&FW%ˆ'íÜgVÊ2Á7V“Ö˜Ê≈ˆWá"íÊFW62Çíê–¢ê–†–¢&˜w2“ÜvóBF"ÊWÜV7WFRá7F◊BííÊ∆¬Çê–†–¢&WGW&‚∞–¢∞–¢&76WB#¢"Ê76WB¿–¢'F˜F≈˜G&FW2#¢ñÁBá"ÁF˜F≈˜G&FW2˜"í¿–¢'vñÁ&FR#¢&˜VÊBá"ÁvñÊÊñÊu˜G&FW2Ú"ÁF˜F≈˜G&FW2¢¬ê–¢ñb"ÁF˜F≈˜G&FW2ÊB"ÁF˜F≈˜G&FW2‚ –¢V«6R„¿–¢'Ê≈˜W6F2#¢&˜VÊBÜf∆ˆBá"ÁF˜F≈˜Ê¬˜"í¬"í¿–¢––¢f˜""ñ‚&˜w0–¢––†–†–¶7ñÊ2FVbˆvWE˜7G&FVwïˆÊ«óFñ72Ä–¢F#¢7ñÊ56W76ñˆ‚¿–¢÷ˆFS¢7G"“$ƒïdR"¿–¢W&ñˆEˆÜ˜W'3¢ñÁB¬ÊˆÊR“#B¿–¢í”‚∆ó7E∂Fñ7E”†–¢""-	-Ì}-ù]"›Ωç-ç≠2‰¬˝‚--]=ç˝¬‚"" –¢g&ˆ“ˆ«ñf∆óÊF"Ê÷ˆFV«2ñ◊˜'BG&FTÜó7F˜'ê–¢g&ˆ“7∆∆6ÜV◊íñ◊˜'B66P–†–¢7G&FVwïˆ6ˆ¬“gVÊ2Ê6ˆ∆W66RÄ–¢G&FTÜó7F˜'íÁ7G&FVwï˜GóR¿–¢G&FTÜó7F˜'íÊFó&V7FñˆÂˆ÷ˆFV≈ˆ∂Wí¿–¢$4Ù‘$î‰TB"¿–¢íÊ∆&V¬Ç'7G&FVwí"ê–†–¢6ˆÊG2“∞–¢G&FTÜó7F˜'íÊ÷ˆFR”“÷ˆFR¿–¢G&FTÜó7F˜'íÁ˜6óFñˆÂ˜7FGW2ÊñÂÚÖ≤$4ƒı4TB"¬%$U4Ù≈dTEıtÙ‚"¬%$U4Ù≈dTEÙƒı5B"¬%$TDTT‘TB"¬%$U4Ù≈dTEı$TDTT‘$ƒR%“í¿–¢––¢ñbW&ñˆEˆÜ˜W'2ó2Ê˜BÊˆÊS†–¢7WFˆfb“FFWFñ÷RÊÊ˜ráFñ÷W¶ˆÊRÁWF2í“Fñ÷VFV«FÜÜ˜W'3◊W&ñˆEˆÜ˜W'2ê–¢FFUˆ6ˆ¬“gVÊ2Ê6ˆ∆W66RÖG&FTÜó7F˜'íÊ6∆˜6VEˆB¬G&FTÜó7F˜'íÊ7&VFVEˆBê–¢6ˆÊG2ÊVÊBÜFFUˆ6ˆ¬„“7WFˆfbê–†–¢7F◊B“Ä–¢6V∆V7BÄ–¢G&FTÜó7F˜'íÊ76WB¿–¢7G&FVwïˆ6ˆ¬¿–¢gVÊ2Ê6˜VÁBÖG&FTÜó7F˜'íÊñBíÊ∆&V¬Ç'F˜F≈˜G&FW2"í¿–¢gVÊ2Á7V“Ä–¢66RÇÖ˜Ê≈ˆWá"‚¬í¬V«6UÛ”ê–¢íÊ∆&V¬Ç'vñÊÊñÊu˜G&FW2"í¿–¢gVÊ2Á7V“Ö˜Ê≈ˆWá"íÊ∆&V¬Ç'F˜F≈˜Ê¬"í¿–¢ê–¢ÁvÜW&RÇ¶6ˆÊG2ê–¢Êw&˜Wˆ'íÖG&FTÜó7F˜'íÊ76WB¬7G&FVwïˆ6ˆ¬ê–¢Ê˜&FW%ˆ'íÜgVÊ2Á7V“Ö˜Ê≈ˆWá"íÊFW62Çíê–¢ê–†–¢&˜w2“ÜvóBF"ÊWÜV7WFRá7F◊BííÊ∆¬Çê–†–¢&WGW&‚∞–¢∞–¢&76WB#¢"Ê76WB¿–¢'7G&FVwí#¢"Á7G&FVwí˜"$4Ù‘$î‰TB"¿–¢'F˜F≈˜G&FW2#¢ñÁBá"ÁF˜F≈˜G&FW2˜"í¿–¢'vñÁ&FR#¢&˜VÊBá"ÁvñÊÊñÊu˜G&FW2Ú"ÁF˜F≈˜G&FW2¢¬ê–¢ñb"ÁF˜F≈˜G&FW2ÊB"ÁF˜F≈˜G&FW2‚ –¢V«6R„¿–¢'Ê≈˜W6F2#¢&˜VÊBÜf∆ˆBá"ÁF˜F≈˜Ê¬˜"í¬"í¿–¢––¢f˜""ñ‚&˜w0–¢––†–†–§&˜WFW"ÊvWBÇ"ˆ∆ófRˆÊ«óFñ72"¬7V÷÷'ì“-	›Ωç-ç≠ƒïdR›-Ì=Ì-ΩÇ˝‚≠-ç-¬Ç--]=ç˝¬"ê–¶7ñÊ2FVbvWEˆ∆ófUˆÊ«óFñ72Ä–¢W&ñˆC¢˜FñˆÊ≈∑7G%““VW'íÇ&∆¬"¬FW67&óFñˆ„“##FÇ¬vB¬3B¬∆¬"í¿–¢F#¢7ñÊ56W76ñˆ‚“FWVÊG2ÜvWEˆF%˜6W76ñˆ‚í¿–¢ì†–¢W&ñˆEˆ÷“≤##FÇ#¢#B¬#vB#¢cÇ¬#3B#¢s#¬&∆¬#¢ÊˆÊW––¢Ü˜W'2“W&ñˆEˆ÷ÊvWBáW&ñˆB¬ÊˆÊRê–†–¢&WGW&‚∞–¢'W&ñˆB#¢W&ñˆB¿–¢&76WEˆÊ«óFñ72#¢vóBˆvWEˆ76WEˆÊ«óFñ72ÜF"¬$ƒïdR"¬Ü˜W'2í¿–¢'7G&FVwïˆÊ«óFñ72#¢vóBˆvWE˜7G&FVwïˆÊ«óFñ72ÜF"¬$ƒïdR"¬Ü˜W'2í¿–¢––†–†–§&˜WFW"ÊvWBÇ"ˆ∆ófRˆF6Ü&ˆ&B"ê–¶7ñÊ2FVbvWEˆ∆ófUˆF6Ü&ˆ&BÄ–¢Ê«óFñ75˜W&ñˆC¢˜FñˆÊ≈∂ñÁE““VW'íÑÊˆÊR¬FW67&óFñˆ„“-	˝]çÌB›Ωç-ç≠Ç"}R¬ÊˆÊR“--]ÕÚ"í¿–¢F#¢7ñÊ56W76ñˆ‚“FWVÊG2ÜvWEˆF%˜6W76ñˆ‚í¿–¢ì†–¢7FófU˜6W76ñˆ‚“Ä–¢vóBF"ÊWÜV7WFRÄ–¢6V∆V7BÑ∆ófUG&FñÊu6W76ñˆ‚ê–¢ÁvÜW&RÄ–¢∆ófUG&FñÊu6W76ñˆ‚Á7FGW2ÊñÂÚÖ≤$E$eB"¬%$TEí"¬$5DïdR%“ê–¢ê–¢Ê˜&FW%ˆ'íÑ∆ófUG&FñÊu6W76ñˆ‚Ê7&VFVEˆBÊFW62Çíê–¢Ê∆ñ÷óBÉê–¢ê–¢íÁ66∆%ˆˆÊUˆ˜%ˆÊˆÊRÇê–¢ –¢∆7E˜7F˜VE˜6W76ñˆÂˆGFÚ“ÊˆÊP–¢ñbÊ˜B7FófU˜6W76ñˆ„†–¢∆7E˜7F˜VB“Ä–¢vóBF"ÊWÜV7WFRÄ–¢6V∆V7BÑ∆ófUG&FñÊu6W76ñˆ‚ê–¢ÁvÜW&RÑ∆ófUG&FñÊu6W76ñˆ‚Á7FGW2”“%5DıTB"ê–¢Ê˜&FW%ˆ'íÑ∆ófUG&FñÊu6W76ñˆ‚Ê7&VFVEˆBÊFW62Çíê–¢Ê∆ñ÷óBÉê–¢ê–¢íÁ66∆%ˆˆÊUˆ˜%ˆÊˆÊRÇê–¢ñb∆7E˜7F˜VC†–¢'VFvWE˜6Ê“vóBvWE˜6W76ñˆÂˆ'VFvWE˜6Ê6Ü˜BÜF"¬∆7E˜7F˜VBê–¢∆7E˜7F˜VE˜6W76ñˆÂˆGFÚ“6W&ñ∆ó¶Uˆ∆ófU˜6W76ñˆÂˆGFÚÜ∆7E˜7F˜VB¬'VFvWE˜6Êê–¢∆7E˜7F˜VE˜6W76ñˆÂˆGFı≤&ó5˜7F˜VB%““G'VP–†–†–†–¢&WVW7G2“Ä–¢Ä–¢vóBF"ÊWÜV7WFRÄ–¢6V∆V7BÑWÜV7WFñˆÂ&WVW7Bê–¢ÁvÜW&RÑWÜV7WFñˆÂ&WVW7BÁ&WVW7FVEˆ÷ˆFR”“$ƒïdR"ê–¢Ê˜&FW%ˆ'íÑWÜV7WFñˆÂ&WVW7BÊ7&VFVEˆBÊFW62Çíê–¢Ê∆ñ÷óBÉSê–¢ê–¢ê–¢Á66∆'2Çê–¢Ê∆¬Çê–¢ê–†–¢'VFvWE˜6Ê“ÊˆÊP–¢ñb7FófU˜6W76ñˆ„†–¢'VFvWE˜6Ê“vóBvWE˜6W76ñˆÂˆ'VFvWE˜6Ê6Ü˜BÜF"¬7FófU˜6W76ñˆ‚ê–†–¢g&ˆ“ˆ«ñf∆óÊWÜV7WFñˆ‚Á6W&ñ∆ó¶W'2ñ◊˜'B6W&ñ∆ó¶UˆWÜV7WFñˆÂ˜&WVW7G0–†–¢&WVW7EˆGF˜2“vóB6W&ñ∆ó¶UˆWÜV7WFñˆÂ˜&WVW7G2ÜF"¬&WVW7G2ê¢gVÊÊV≈˜7F◊B“Ä¢6V∆V7BÑWÜV7WFñˆÂ&WVW7BÁFW&÷ñÊ≈ˆ6ˆFR¬WÜV7WFñˆÂ&WVW7BÁ7FFR¬gVÊ2Ê6˜VÁBÇíê¢ÁvÜW&RÑWÜV7WFñˆÂ&WVW7BÁ&WVW7FVEˆ÷ˆFR”“$ƒïdR"ê¢Êw&˜Wˆ'íÑWÜV7WFñˆÂ&WVW7BÁFW&÷ñÊ≈ˆ6ˆFR¬WÜV7WFñˆÂ&WVW7BÁ7FFRê¢ê¢ñbÊ«óFñ75˜W&ñˆBó2Ê˜BÊˆÊS†¢gVÊÊV≈˜7F◊B“gVÊÊV≈˜7F◊BÁvÜW&RÄ¢WÜV7WFñˆÂ&WVW7BÊ7&VFVEˆB„“FFWFñ÷RÊÊ˜ráFñ÷W¶ˆÊRÁWF2í“Fñ÷VFV«FÜÜ˜W'3÷Ê«óFñ75˜W&ñˆBê¢ê¢gVÊÊV≈˜&˜w2“ÜvóBF"ÊWÜV7WFRÜgVÊÊV≈˜7F◊BííÊ∆¬Çê¢gVÊÊV≈ˆ6˜VÁG2“∑–¢f˜"FW&÷ñÊ≈ˆ6ˆFR¬&WVW7E˜7FFR¬6˜VÁBñ‚gVÊÊV≈˜&˜w3†¢∂Wí“FW&÷ñÊ≈ˆ6ˆFR˜"&WVW7E˜7FFR˜"%T‰¥‰ıt‚ ¢gVÊÊV≈ˆ6˜VÁG5∂∂Wï““gVÊÊV≈ˆ6˜VÁG2ÊvWBÜ∂Wí¬í≤ñÁBÜ6˜VÁBê¢÷ó'&˜%˜7F◊B“6V∆V7BÜgVÊ2Ê6˜VÁBÇííÁ6V∆V7Eˆg&ˆ“Ñ∆ófT÷ó'&˜$6ÊFñFFRíÁvÜW&RÄ¢∆ófT÷ó'&˜$6ÊFñFFRÁF&vWEˆ÷ˆFR”“$ƒïdR ¢ê¢ñbÊ«óFñ75˜W&ñˆBó2Ê˜BÊˆÊS†¢÷ó'&˜%˜7F◊B“÷ó'&˜%˜7F◊BÁvÜW&RÄ¢∆ófT÷ó'&˜$6ÊFñFFRÊ7&VFVEˆB„“FFWFñ÷RÊÊ˜ráFñ÷W¶ˆÊRÁWF2í“Fñ÷VFV«FÜÜ˜W'3÷Ê«óFñ75˜W&ñˆBê¢ê¢÷ó'&˜%ˆ6˜VÁB“ñÁBÇÜvóBF"ÊWÜV7WFRÜ÷ó'&˜%˜7F◊BííÁ66∆"Çí˜"ê–†–¢v˜&∂W%˜7FGW2“vóBvWEˆ∆FW7Eˆ∆ófU˜v˜&∂W%˜7FGW2ÜF"ê–¢&VFñÊW72“ÊˆÊP–¢ñb7FófU˜6W76ñˆ„†–¢&VFñÊW72“vóBWf«VFUˆ∆ófU˜&VFñÊW72ÜF"¬7FófU˜6W76ñˆ‚¬v˜&∂W%˜7FGW3◊v˜&∂W%˜7FGW2ê–†–¢7FGW2“7FófU˜6W76ñˆ‚Á7FGW2ñb7FófU˜6W76ñˆ‚V«6RÊˆÊP–¢ –¢˜6óFñˆÁ5˜ñ∆ˆB“vóBˆvWE˜˜6óFñˆÁ5ˆFñ7BÜF"¬$ƒïdR"ê–¢˜6óFñˆÁ2“˜6óFñˆÁ5˜ñ∆ˆBÊvWBÇ'˜6óFñˆÁ2"¬≤'G&F&∆R#¢µ“¬'&W6ˆ«fVB#¢µ“¬&&6ÜófR#¢µ◊“ê–¢ –¢7FófU˜˜5ˆ6˜VÁB“7V“Ä–¢f˜"˜6óFñˆ‚ñ‚˜6óFñˆÁ2ÊvWBÇ'G&F&∆R"¬µ“íñb˜6óFñˆ‚ÊvWBÇ'&V÷ñÊñÊu˜6Ü&W2"¬í‚ –¢ê–†–¢&VFñÊW75˜&VGí“&ˆˆ¬á&VFñÊW72ÊB&VFñÊW72Á&VGíê–†–¢ñb7FGW3†–¢fñ∆&∆Uˆ7FñˆÁ2“∞–¢&6ÜV6µ˜&VFñÊW72#¢7FGW2ñ‚≤$E$eB"¬%$TEí"¬%5DıTB'“¿–¢&7FófFR#¢á&VFñÊW75˜&VGíÊB7FGW2ñ‚≤$E$eB"¬%$TEí"¬%5DıTB'“í¿–¢'7F˜#¢7FGW2”“$5DïdR"¿–¢&6∆˜6Uˆ∆¬#¢7FófU˜˜5ˆ6˜VÁB‚¿–¢&fñÊó6Ç#¢7FGW2ñ‚≤$E$eB"¬%$TEí"¬%5DıTB'“¿–¢––¢7FñˆÂ˜&V6ˆÁ2“∞–¢&7FófFR#¢Ä–¢ÊˆÊP–¢ñbfñ∆&∆Uˆ7FñˆÁ5≤&7FófFR%––¢V«6RÄ–¢#≤"Ê¶ˆñ‚á&VFñÊW72ÊW'&˜'2˜"µ“ê–¢ñb&VFñÊW70–¢V«6R-
›}Ω-Ω˝ÌΩ›ç-R˝Ì-]≠2=Ì-Ì-›Ì-Ç –¢ê–¢í¿–¢'7F˜#¢ÊˆÊRñb7FGW2”“$5DïdR"V«6R-
]çÚ›R≠-ç-›"¿–¢&6∆˜6Uˆ∆¬#¢ÑÊˆÊRñb7FófU˜˜5ˆ6˜VÁB‚V«6R-	›]"Ì-≠Ω-ΩR˝Ì}çmçí"í¿–¢&fñÊó6Ç#¢ÊˆÊR¿–¢&6ÜV6µ˜&VFñÊW72#¢ÊˆÊR¿–¢––¢V«6S†–¢fñ∆&∆Uˆ7FñˆÁ2“∞–¢&6ÜV6µ˜&VFñÊW72#¢f«6R¿–¢&7FófFR#¢f«6R¿–¢'7F˜#¢f«6R¿–¢&6∆˜6Uˆ∆¬#¢f«6R¿–¢&fñÊó6Ç#¢f«6R¿–¢––¢7FñˆÂ˜&V6ˆÁ2“∑––†–¢&WGW&‚∞–¢&fñ∆&∆Uˆ7FñˆÁ2#¢fñ∆&∆Uˆ7FñˆÁ2¿–¢&7FñˆÂ˜&V6ˆÁ2#¢7FñˆÂ˜&V6ˆÁ2¿–¢'&VFñÊW72#¢Ä–¢∞–¢'&VGí#¢&VFñÊW72Á&VGí¿–¢&6ÜV6∑2#¢&VFñÊW72Ê6ÜV6∑2¿–¢&W'&˜'2#¢&VFñÊW72ÊW'&˜'2¿–¢'v&ÊñÊw2#¢&VFñÊW72Áv&ÊñÊw2¿–¢––¢ñb&VFñÊW70–¢V«6RÊˆÊP–¢í¿–¢&ñvÊ˜&UˆVFvUˆFV6í#¢ÜvóBˆvWE˜'VÁFñ÷Uˆf∆rÜF"¬$ƒïdUÙît‰ı$UÙTDtUÙDT4í"ííÊ∆˜vW"Çí”“'G'VR"¿–¢&˜&FW%ˆ÷ˆFUˆ6ˆÊfñr#¢∞–¢&÷ˆFR#¢ÜvóBˆvWE˜'VÁFñ÷Uˆf∆rÜF"¬$ƒïdUÙı$DU%Ù‘ÙDR"¬FVfV«C“$‘¥U%ıED¬"ííÁWW"Çí¿–¢&wF5˜GF≈˜6V6ˆÊG2#¢f∆ˆBÜvóBˆvWE˜'VÁFñ÷Uˆf∆rÜF"¬$ƒïdUÙuD5ıED≈ı4T4Ù‰E2"¬FVfV«C“#„"íí¿–¢&fµ˜&WG'ïˆ÷ÖˆGFV◊G2#¢ñÁBÜvóBˆvWE˜'VÁFñ÷Uˆf∆rÜF"¬$ƒïdUÙdµı$UE%ïÙ‘ÖÙEDT’E2"¬FVfV«C“#2"íí¿–¢&fµ˜&WG'ïˆFV∆ï˜6V2#¢f∆ˆBÜvóBˆvWE˜'VÁFñ÷Uˆf∆rÜF"¬$ƒïdUÙdµı$UE%ïÙDTƒïı4T2"¬FVfV«C“#„sR"íí¿–¢'W%ˆWÜV7WFñˆÂ˜&ˆfñ∆R#¢ÜvóBˆvWE˜'VÁFñ÷Uˆf∆rÜF"¬%U%ÙUÑT5UDîÙÂı$ÙdîƒR"¬FVfV«C“$îÂ5DÂB"ííÁWW"Çí¿–¢'W%ˆ∆ófUˆFV∆ï˜6V2#¢f∆ˆBÜvóBˆvWE˜'VÁFñ÷Uˆf∆rÜF"¬%U%ÙƒïdUÙDTƒïı4T2"¬FVfV«C“#"„"íí¿–¢“¿–¢'6W76ñˆ‚#¢Ä–¢6W&ñ∆ó¶Uˆ∆ófU˜6W76ñˆÂˆGFÚÜ7FófU˜6W76ñˆ‚¬'VFvWE˜6Êê–¢ñb7FófU˜6W76ñˆ‡–¢V«6R∆7E˜7F˜VE˜6W76ñˆÂˆGF–¢í¿–¢'v˜&∂W"#¢∞–¢'v˜&∂W%ˆñB#¢v˜&∂W%˜7FGW2Áv˜&∂W%ˆñBñbv˜&∂W%˜7FGW2V«6RÊˆÊR¿–¢&ÜV'F&VEˆB#¢Ä–¢v˜&∂W%˜7FGW2ÊÜV'F&VEˆBÊó6ˆf˜&÷BÇê–¢ñbv˜&∂W%˜7FGW2ÊBv˜&∂W%˜7FGW2ÊÜV'F&VEˆ@–¢V«6RÊˆÊP–¢í¿–¢&vFWvï˜&VGí#¢v˜&∂W%˜7FGW2ÊvFWvï˜&VGíñbv˜&∂W%˜7FGW2V«6Rf«6R¿–¢&7&VFVÁFñ«5ˆ∆ˆFVB#¢Ä–¢v˜&∂W%˜7FGW2Ê7&VFVÁFñ«5ˆ∆ˆFVBñbv˜&∂W%˜7FGW2V«6Rf«6P–¢í¿–¢'v∆∆WEˆFG&W72#¢áv˜&∂W%˜7FGW2Áv∆∆WEˆFG&W72ñbv˜&∂W%˜7FGW2V«6RÊˆÊRí¿–¢&&∆Ê6U˜W6F2#¢Ä–¢f∆ˆBáv˜&∂W%˜7FGW2Ê&∆Ê6U˜W6F2ê–¢ñbv˜&∂W%˜7FGW2ÊBv˜&∂W%˜7FGW2Ê&∆Ê6U˜W6F2ó2Ê˜BÊˆÊP–¢V«6R„ –¢í¿–¢&6ˆ∆∆FW&≈ˆ∆∆˜vÊ6U˜&VGí#¢Ä–¢v˜&∂W%˜7FGW2Ê6ˆ∆∆FW&≈ˆ∆∆˜vÊ6U˜&VGíñbv˜&∂W%˜7FGW2V«6Rf«6P–¢í¿–¢&6ˆÊFóFñˆÊ≈ˆ∆∆˜vÊ6U˜&VGí#¢Ä–¢v˜&∂W%˜7FGW2Ê6ˆÊFóFñˆÊ≈ˆ∆∆˜vÊ6U˜&VGíñbv˜&∂W%˜7FGW2V«6Rf«6P–¢í¿–¢“¿–¢&6ÊFñFFW2#¢µ“¿–¢'˜6óFñˆÁ2#¢˜6óFñˆÁ2¿–¢'&WVW7G2#¢&WVW7EˆGF˜2¿¢&WÜV7WFñˆÂˆgVÊÊV¬#¢≤&÷ó'&˜%ˆ6ÊFñFFW2#¢÷ó'&˜%ˆ6˜VÁB¬'FW&÷ñÊ≈ˆ6˜VÁG2#¢gVÊÊV≈ˆ6˜VÁG7“¿–¢&76WEˆÊ«óFñ72#¢vóBˆvWEˆ76WEˆÊ«óFñ72ÜF"¬÷ˆFS“$ƒïdR"¬W&ñˆEˆÜ˜W'3÷Ê«óFñ75˜W&ñˆBí¿–¢'7G&FVwïˆÊ«óFñ72#¢vóBˆvWE˜7G&FVwïˆÊ«óFñ72ÜF"¬÷ˆFS“$ƒïdR"¬W&ñˆEˆÜ˜W'3”#Bí¿–¢––†–†–§&˜WFW"Á˜7BÇ"˜&WVW7G2˜∑&WVW7EˆñG“˜&V6ˆÊ6ñ∆R"ê–¶7ñÊ2FVb&V6ˆÊ6ñ∆U˜&WVW7BÄ–¢&WVW7EˆñC¢WVñBÂUTîB¿–¢F#¢7ñÊ56W76ñˆ‚“FWVÊG2ÜvWEˆF%˜6W76ñˆ‚í¿–¢ì†–¢&W“vóBF"Á66∆"Ä–¢6V∆V7BÑWÜV7WFñˆÂ&WVW7Bê–¢ÁvÜW&RÑWÜV7WFñˆÂ&WVW7BÊñB”“&WVW7EˆñBê–¢ÁvóFÖˆf˜%˜WFFRÇê–¢ê–†–¢ñb&Wó2ÊˆÊS†–¢&ó6RÖEEWÜ6WFñˆ‚ÉCB¬-	}˝-≠›R›ùM]›"ê–†–¢ñb&WÁ&WVW7FVEˆ÷ˆFR“$ƒïdR#†–¢&ó6RÖEEWÜ6WFñˆ‚ÉCí¬-
-]≠}]ç]›-ÌΩÕ≠‚MΩÚƒïdR›}˝-Ì¢"ê–†–¢ñb&WÁ7FFR”“%$T4Ù‰4îƒî‰r#†–¢&WGW&‚∞–¢'&WVW7EˆñB#¢7G"á&WÊñBí¿–¢'7FFR#¢%$T4Ù‰4îƒî‰r"¿–¢&ñFV◊˜FVÁB#¢G'VR¿–¢––†–¢∆∆˜vVE˜7FFW2“∞–¢%5T$‘ïEDî‰r"¿–¢$44UDTB"¿–¢%T‰¥‰ıt‚"¿–¢%%Dîƒ≈ïÙdîƒƒTB"¿–¢%$T4Ù‰4îƒî‰r"¿–¢$‘ÂT≈ı$UdîUuı$UTï$TB"¿–¢––†–¢ñb&WÁ7FFRÊ˜Bñ‚∆∆˜vVE˜7FFW3†–¢&ó6RÖEEWÜ6WFñˆ‚Ä–¢Cí¿–¢b-	}˝-≠2"--=R∑&WÁ7FFW“›]ΩÕ}Ú˝]]-ÌMç-¬"$T4Ù‰4îƒî‰r"¿–¢ê–†–¢&˜fñFW%ˆWfñFVÊ6R“vóBF"Á66∆"Ä–¢6V∆V7BÜgVÊ2Ê6˜VÁBÑWÜV7WFñˆ‰GFV◊BÊñBííÁvÜW&RÄ–¢WÜV7WFñˆ‰GFV◊BÁ&WVW7EˆñB”“&WVW7EˆñB¿–¢WÜV7WFñˆ‰GFV◊BÁ&˜fñFW%ˆ˜&FW%ˆñBÊó5ˆÊ˜BÑÊˆÊRí¿–¢ê–¢ê–†–¢ñbÊ˜B&˜fñFW%ˆWfñFVÊ6S†–¢&ó6RÖEEWÜ6WFñˆ‚Ä–¢C#"¿–¢-	›]"&˜fñFW%ˆ˜&FW%ˆñB(	B-]≠ˆ«ñ÷&∂WB›]-Ì}ÕÌm›"¿–¢ê–†–¢&WÁ7FFR“%$T4Ù‰4îƒî‰r –¢&WÁWFFVEˆB“FFWFñ÷RÊÊ˜ráFñ÷W¶ˆÊRÁWF2ê–¢vóBF"Ê6ˆ÷÷óBÇê–†–¢&WGW&‚≤'&WVW7EˆñB#¢7G"á&WÊñBí¬'7FFR#¢%$T4Ù‰4îƒî‰r'––†–†–¢2““““““““““““““““““““““““““““““““““““““““““““““““““““““““““““““““““““““““““––¢26WGF∆V÷VÁBb&VFV◊Fñˆ‚VÊGˆñÁG0–¢2““““““““““““““““““““““““““““““““““““““““““““““““““““““““““““““““““““““““““––†–§&˜WFW"Á˜7BÇ"ˆ∆ófR˜˜6óFñˆÁ2˜∑G&FUˆñG“˜&V6ˆÊ6ñ∆R◊&W6ˆ«WFñˆ‚"ê–¶7ñÊ2FVb&V6ˆÊ6ñ∆U˜&W6ˆ«WFñˆÂˆVÊGˆñÁBÄ–¢G&FUˆñC¢ñÁB¿–¢F#¢7ñÊ56W76ñˆ‚“FWVÊG2ÜvWEˆF%˜6W76ñˆ‚í¿–¢ì†–¢g&ˆ“ˆ«ñf∆óÊWÜV7WFñˆ‚Ê∆ófU˜6WGF∆V÷VÁE˜6W'fñ6Rñ◊˜'B&V6ˆÊ6ñ∆Uˆ∆ófU˜&W6ˆ«WFñˆ‚¬∆ófU˜6óFñˆ‰Ê˜Df˜VÊB¬÷&∂WDÊ˜E&W6ˆ«fVB¬v÷÷îW'&˜ –¢G'ì†–¢G&FR“vóB&V6ˆÊ6ñ∆Uˆ∆ófU˜&W6ˆ«WFñˆ‚ÜF"¬G&FUˆñBê–¢vóBF"Ê6ˆ÷÷óBÇê–¢&WGW&‚≤'7FGW2#¢&ˆ≤"¬'˜6óFñˆÂ˜7FGW2#¢G&FRÁ˜6óFñˆÂ˜7FGW7––¢WÜ6WB∆ófU˜6óFñˆ‰Ê˜Df˜VÊC†–¢&ó6RÖEEWÜ6WFñˆ‚ÉCB¬$ƒïdR›˝Ì}çmçÚ›R›ùM]›"ê–¢WÜ6WB÷&∂WDÊ˜E&W6ˆ«fVC†–¢vóBF"Ê6ˆ÷÷óBÇê–¢&ó6RÖEEWÜ6WFñˆ‚ÉCí¬-
Ω›Ì¢]ùR›R}-]ç]“"ê–¢WÜ6WBv÷÷îW'&˜"2S†–¢&ó6RÖEEWÜ6WFñˆ‚ÉS2¬b-	Ìçç≠v÷÷ì¢∑7G"ÜRó“"ê–¢WÜ6WBWÜ6WFñˆ‚2S†–¢∆ˆvvW"ÊWÜ6WFñˆ‚Ç'&V6ˆÊ6ñ∆U˜&W6ˆ«WFñˆÂˆW'&˜""ê–¢&ó6RÖEEWÜ6WFñˆ‚ÉS¬b-	Ìçç≠-]≠É¢∑7G"ÜRó“"ê–†–§&˜WFW"Á˜7BÇ"ˆ∆ófR˜˜6óFñˆÁ2˜∑G&FUˆñG“˜&VFVV“"ê–¶7ñÊ2FVb&VFVV’˜˜6óFñˆÂˆVÊGˆñÁBÄ–¢G&FUˆñC¢ñÁB¿–¢F#¢7ñÊ56W76ñˆ‚“FWVÊG2ÜvWEˆF%˜6W76ñˆ‚í¿–¢ì†–¢&ó6RÖEEWÜ6WFñˆ‚Ä–¢7FGW5ˆ6ˆFS”S¿–¢FWFñ√“-	˝Ì=ç]›çR}]]rMçÌB˝Ì≠›R]Ωç}Ì-›‚"¿–¢ê–†–§&˜WFW"Á˜7BÇ"ˆ∆ófR˜˜6óFñˆÁ2˜∑G&FUˆñG“˜&V6ˆÊ6ñ∆R◊&VFV◊Fñˆ‚"ê–¶7ñÊ2FVb&V6ˆÊ6ñ∆U˜&VFV◊FñˆÂˆVÊGˆñÁBÄ–¢G&FUˆñC¢ñÁB¿–¢F#¢7ñÊ56W76ñˆ‚“FWVÊG2ÜvWEˆF%˜6W76ñˆ‚í¿–¢ì†–¢G&FR“vóBF"Á66∆"Ä–¢6V∆V7BÖG&FTÜó7F˜'íê–¢ÁvÜW&RÖG&FTÜó7F˜'íÊñB”“G&FUˆñBê–¢ÁvóFÖˆf˜%˜WFFRÇê–¢ê–¢ñbÊ˜BG&FR˜"G&FRÊ÷ˆFR“$ƒïdR#†–¢&ó6RÖEEWÜ6WFñˆ‚ÉCB¬$ƒïdR›˝Ì}çmçÚ›R›ùM]›"ê–¢ –¢ñbG&FRÁ˜6óFñˆÂ˜7FGW2Ê˜Bñ‚≤%$TDTT‘î‰r"¬%$TDT’DîÙÂıT‰¥‰ıt‚"¬%$U4Ù≈dTEı$TDTT‘$ƒR'”†–¢&ó6RÖEEWÜ6WFñˆ‚ÉCí¬-	›]MÌ˝=-çÕΩí--=MΩÚ˝Ì-]≠Ç˝Ì=ç]›çÚ"ê–†–¢2	˝‚
-	r›R--ç¬$TDTT‘TB]rWfñFVÊ6R‡–¢&ó6RÖEEWÜ6WFñˆ‚ÉC#"¬$ˆ‚÷6Üñ‚-]≠]ùR›R]Ωç}Ì-›"ê–†