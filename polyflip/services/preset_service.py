import json
import structlog
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from polyflip.db.models import ConfigPreset, RuntimeSettings
from polyflip.settings_registry import editable_keys

logger = structlog.get_logger(__name__)

class PresetService:

    @staticmethod
    async def capture_snapshot(db: AsyncSession) -> Dict[str, str]:
        """Читает ВСЕ RuntimeSettings и возвращает dict {key: value}."""
        rows = (await db.execute(select(RuntimeSettings))).scalars().all()
        return {r.key: r.value for r in rows}

    @staticmethod
    async def save_preset(
        db: AsyncSession,
        name: str,
        description: Optional[str] = None,
        preset_type: str = "manual",
        capital_at_save: Optional[float] = None,
        pnl_at_save: Optional[float] = None,
        created_by: str = "user",
    ) -> ConfigPreset:
        """Сохраняет новый слепок конфигурации."""
        snapshot = await PresetService.capture_snapshot(db)
        preset = ConfigPreset(
            name=name,
            description=description,
            preset_type=preset_type,
            snapshot=json.dumps(snapshot, ensure_ascii=False),
            capital_at_save=capital_at_save,
            pnl_at_save=pnl_at_save,
            created_at=datetime.now(timezone.utc),
            created_by=created_by,
            is_active=True,
        )
        db.add(preset)
        await db.commit()
        await db.refresh(preset)
        logger.info("preset_saved", id=preset.id, name=name, preset_type=preset_type)
        return preset

    @staticmethod
    async def restore_preset(
        db: AsyncSession,
        preset_id: int,
        restored_by: str = "user",
    ) -> int:
        """
        Применяет параметры из слепка.
        БЕЗОПАСНОСТЬ: Применяются ТОЛЬКО редактируемые торговые ключи из editable_keys().
        Игнорируются кнопка TRADING_ENABLED и автокалибровки порогов (AUTO_FLIP_THRESHOLD_*).
        """
        preset = await db.get(ConfigPreset, preset_id)
        if not preset or not preset.is_active:
            raise ValueError(f"Пресет {preset_id} не найден или был удалён")

        params = json.loads(preset.snapshot)
        valid_editable_keys = set(editable_keys())
        now = datetime.now(timezone.utc)
        changed = 0

        for key, value in params.items():
            # Безопасность: восстанавливаем только ключи из editable_keys()
            if key not in valid_editable_keys:
                continue

            row = await db.get(RuntimeSettings, key)
            if row:
                if row.value != str(value):
                    row.value = str(value)
                    row.updated_at = now
                    row.updated_by = f"preset_restore:{preset_id}:{restored_by}"
                    changed += 1
            else:
                db.add(RuntimeSettings(
                    key=key,
                    value=str(value),
                    updated_at=now,
                    updated_by=f"preset_restore:{preset_id}:{restored_by}",
                ))
                changed += 1

        await db.commit()
        logger.info("preset_restored", id=preset_id, changed_keys=changed, restored_by=restored_by)
        return changed

    @staticmethod
    async def check_and_save_ath(
        db: AsyncSession,
        current_capital: float,
        current_pnl: float,
        min_pnl_diff: float = 1.0,
        min_interval_hours: int = 1,
    ) -> Optional[ConfigPreset]:
        """
        Сохраняет ATH-слепок при установке нового рекорда.
        Предохранители:
          - Рост рекорда минимум на min_pnl_diff (+1.0 USDC)
          - Минимальный интервал между авто-слепками min_interval_hours (1 час)
        """
        now = datetime.now(timezone.utc)

        # 1. Получаем существующие ATH-пресеты
        q = select(ConfigPreset).where(
            and_(
                ConfigPreset.preset_type.in_(["ath_capital", "ath_pnl"]),
                ConfigPreset.is_active == True,  # noqa: E712
            )
        ).order_by(ConfigPreset.created_at.desc())
        ath_presets = (await db.execute(q)).scalars().all()

        # 2. Проверка 1-часового интервала от последнего ATH
        if ath_presets:
            last_ath_time = ath_presets[0].created_at
            if now - last_ath_time < timedelta(hours=min_interval_hours):
                return None

        # 3. Вычисляем текущие максимумы
        prev_max_capital = max((p.capital_at_save or 0.0 for p in ath_presets), default=0.0)
        prev_max_pnl     = max((p.pnl_at_save     or 0.0 for p in ath_presets), default=0.0)

        is_capital_ath = (current_capital - prev_max_capital) >= min_pnl_diff
        is_pnl_ath     = (current_pnl - prev_max_pnl) >= min_pnl_diff

        if not (is_capital_ath or is_pnl_ath):
            return None

        ptype = "ath_capital" if is_capital_ath else "ath_pnl"
        ts_str = now.strftime("%Y-%m-%d_%H-%M")
        name = f"🏆 ATH_{ptype.upper()}_{ts_str}"
        description = f"Авто-слепок рекорда: Capital=${current_capital:.2f}, PnL=${current_pnl:.2f}"

        return await PresetService.save_preset(
            db=db,
            name=name,
            description=description,
            preset_type=ptype,
            capital_at_save=current_capital,
            pnl_at_save=current_pnl,
            created_by="system_ath",
        )
