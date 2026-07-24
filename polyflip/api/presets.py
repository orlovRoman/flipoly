import json
import structlog
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from polyflip.db.connection import get_db_session
from polyflip.services.preset_service import PresetService
from polyflip.db.models import ConfigPreset, RuntimeSettings

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/presets", tags=["presets"])

class SavePresetRequest(BaseModel):
    name: str
    description: Optional[str] = None

class PresetResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    preset_type: str
    capital_at_save: Optional[float]
    pnl_at_save: Optional[float]
    created_at: str
    created_by: str
    param_count: int

@router.get("/", response_model=List[PresetResponse])
async def list_presets(db: AsyncSession = Depends(get_db_session)):
    """Возвращает список сохранённых пресетов."""
    rows = (await db.execute(
        select(ConfigPreset)
        .where(ConfigPreset.is_active == True)  # noqa: E712
        .order_by(ConfigPreset.created_at.desc())
    )).scalars().all()

    result = []
    for r in rows:
        try:
            param_count = len(json.loads(r.snapshot))
        except Exception:
            param_count = 0
        result.append(PresetResponse(
            id=r.id,
            name=r.name,
            description=r.description,
            preset_type=r.preset_type,
            capital_at_save=r.capital_at_save,
            pnl_at_save=r.pnl_at_save,
            created_at=r.created_at.isoformat(),
            created_by=r.created_by,
            param_count=param_count,
        ))
    return result

@router.post("/", response_model=PresetResponse)
async def save_preset(req: SavePresetRequest, db: AsyncSession = Depends(get_db_session)):
    """Сохраняет текущие RuntimeSettings как новый ручной пресет."""
    if not req.name or not req.name.strip():
        raise HTTPException(status_code=400, detail="Имя пресета не может быть пустым")

    preset = await PresetService.save_preset(
        db=db,
        name=req.name.strip(),
        description=req.description,
        preset_type="manual",
        created_by="user_ui"
    )
    snap = json.loads(preset.snapshot)
    return PresetResponse(
        id=preset.id,
        name=preset.name,
        description=preset.description,
        preset_type=preset.preset_type,
        capital_at_save=preset.capital_at_save,
        pnl_at_save=preset.pnl_at_save,
        created_at=preset.created_at.isoformat(),
        created_by=preset.created_by,
        param_count=len(snap),
    )

@router.post("/{preset_id}/restore")
async def restore_preset(preset_id: int, db: AsyncSession = Depends(get_db_session)):
    """Применяет параметры из указанного пресета к RuntimeSettings."""
    try:
        changed_count, updated_params = await PresetService.restore_preset(db, preset_id, restored_by="user_ui")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {
        "ok": True,
        "preset_id": preset_id,
        "changed_keys": changed_count,
        "updated_params": updated_params,
    }

@router.get("/{preset_id}/diff")
async def diff_preset(preset_id: int, db: AsyncSession = Depends(get_db_session)):
    """Возвращает разницу между пресетом и текущими RuntimeSettings."""
    preset = await db.get(ConfigPreset, preset_id)
    if not preset or not preset.is_active:
        raise HTTPException(status_code=404, detail="Пресет не найден")

    try:
        preset_snap = json.loads(preset.snapshot)
    except Exception:
        preset_snap = {}

    current_snap = await PresetService.capture_snapshot(db)

    diff = {}
    for k, v in preset_snap.items():
        curr_val = current_snap.get(k)
        if str(v) != str(curr_val or ""):
            diff[k] = {
                "preset": str(v),
                "current": str(curr_val) if curr_val is not None else "N/A"
            }

    return {"preset_id": preset_id, "diff_count": len(diff), "diff": diff}

@router.delete("/{preset_id}")
async def delete_preset(preset_id: int, db: AsyncSession = Depends(get_db_session)):
    """Софт-удаление пресета (is_active = False)."""
    preset = await db.get(ConfigPreset, preset_id)
    if not preset or not preset.is_active:
        raise HTTPException(status_code=404, detail="Пресет не найден")

    preset.is_active = False
    await db.commit()
    return {"ok": True, "preset_id": preset_id}
