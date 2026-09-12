import json
import structlog
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from polyflip.db.models import ConfigPreset, RuntimeSettings

TRADING_PRESET_KEYS = {
    # Мёртвая зона
    "AUTO_DEAD_ZONE",
    "DEAD_ZONE_WIDTH",
    # Режим и активы
    "TRADING_MODE",
    "TRADE_ASSETS",
    "ENTRY_STRATEGY",
    # Пороги стратегии и флип
    "FLIP_THRESHOLD",
    "TRADE_FLIP_THRESHOLD",
    "NO_FLIP_THRESHOLD",
    "FAVORITE_THRESHOLD",
    "TRADE_ON_FAVORITE",
    "TRADE_ON_FLIP",
    "MIN_DIRECTION_PROB",
    "MIN_WIN_PROB",
    "COMBINED_DIR_DISCOUNT_WEIGHT",
    "COMBINED_DIR_STRONG_THRESHOLD",
    "COMBINED_REQUIRE_CONSENSUS",
    "COMBINED_FALLBACK_TO_LOGREG_ON_NONE",
    "COMBINED_LGBM_UNAVAILABLE_POLICY",
    "COMBINED_LOGREG_ABSTAIN_BAND",
    "INVERT_LGBM_SIGNAL",
    "ENABLE_ECE_CORRECTION",
    "LIGHTGBM_DECISION_MODE",
    "TRADING_POLICY_MODE",
    "WEIGHTED_POLICY_ID",
    "WEIGHTED_MARKET_WEIGHT",
    "WEIGHTED_LOGREG_WEIGHT",
    "WEIGHTED_LGBM_WEIGHT",
    "WEIGHTED_FEE_RATE",
    "WEIGHTED_FEE_EXPONENT",
    "WEIGHTED_SLIPPAGE_RATE",
    "WEIGHTED_EXECUTION_ROLE",
    "WEIGHTED_MIN_NET_EV_FAVORITE",
    "WEIGHTED_MIN_NET_EV_OUTSIDER",
    "WEIGHTED_SIZING_MODE",
    "OUTSIDER_PWIN_DISCOUNT",
    # Edge и цены
    "MIN_EDGE",
    "FAVORITE_MIN_EDGE",
    "NO_MIN_EDGE",
    "FAVORITE_MIN_PRICE",
    "FAVORITE_MAX_PRICE",
    "OUTSIDER_MAX_PRICE",
    "TRADE_MIN_PRICE",
    "TRADE_MAX_PRICE",
    "MAX_PRICE_DRIFT",
    "MAX_SPREAD_PCT",
    # Сайзинг и лимиты
    "TRADE_BET_SIZE_USDC",
    "MAX_BET_SIZE_USDC",
    "BET_SIZING_MODE",
    "LIQUIDITY_FRACTION",
    "MAX_BET_EDGE",
    "DAILY_LOSS_LIMIT_USDC",
    "MAX_OPEN_POSITIONS",
    "MAX_TOTAL_EXPOSURE_USDC",
    "MAX_SINGLE_ORDER_USDC",
    "CONFIRM_THRESHOLD_USDC",
    "INITIAL_CAPITAL",
    # Стоп-лосс и тейк-профит
    "STOP_LOSS_ENABLED",
    "STOP_LOSS_PCT_FAVORITE",
    "STOP_LOSS_PCT_OUTSIDER",
    "STOP_LOSS_CHECK_SEC",
    "TAKE_PROFIT_ENABLED",
    "TAKE_PROFIT_MULTIPLIER",
    "TAKE_PROFIT_CHECK_INTERVAL_SEC",
    "TAKE_PROFIT_ORDER_MODE",
    # Крипто
    "USE_CRYPTO_CONFIRM",
    "CRYPTO_STANDALONE",
    # Таймеры / комиссии
    "LIVE_POLL_INTERVAL_SECONDS",
    "TRADE_MIN_TIME_LEFT_SEC",
    "TRADE_MAX_TIME_LEFT_SEC",
    "TRADE_EXECUTION_TIME_SEC",
    "EXECUTION_COOLDOWN_SEC",
    "POLYMARKET_FEE_RATE",
    "PAPER_EXECUTION_PROFILE",
    "PAPER_LIVE_DELAY_SEC",
    "PAPER_SLIPPAGE_PCT",
    "PAPER_FEE_MODEL",
    "PAPER_FEE_RATE",
    "PAPER_MAKER_FEE_RATE",
    "PAPER_FEE_EXPONENT",
    "PAPER_MIN_ORDER_SHARES",
    "LIVE_ORDER_MODE",
    "LIVE_GTC_TTL_SECONDS",
    "LIVE_MAKER_REPRICE_ON_CROSS",
    "LIVE_MAKER_REPRICE_MAX_RETRIES",
    "LIVE_MAKER_TICK_SIZE",
}

OPERATIONAL_KEYS = {
    "EXECUTION_MODE",
    "TRADING_ENABLED",
    "LIVE_TRADING_ENABLED",
    "LIVE_MIRROR_ENABLED",
    "LIVE_RELEASE_MODE",
    "LIVE_MIRROR_STARTED_AT",
    "BYPASS_BET_SIZE_CHECK",
}

logger = structlog.get_logger(__name__)


class PresetService:

    @staticmethod
    def sanitize_snapshot(settings: Dict[str, Any]) -> Dict[str, str]:
        """
        Оставляет только те настройки, которые входят в торговый пресет,
        исключает оперативные рубильники и преобразует значения в строки.
        """
        return {
            key: str(value)
            for key, value in settings.items()
            if key in TRADING_PRESET_KEYS
            and key not in OPERATIONAL_KEYS
            and value is not None
        }

    @staticmethod
    def get_trading_preset_settings(
        settings: Dict[str, Any],
    ) -> Dict[str, str]:
        """Псевдоним sanitize_snapshot для обратной совместимости."""
        return PresetService.sanitize_snapshot(settings)

    @staticmethod
    async def capture_snapshot(db: AsyncSession) -> Dict[str, str]:
        """Читает RuntimeSettings и возвращает только торговые параметры."""
        rows = (await db.execute(select(RuntimeSettings))).scalars().all()
        # Include registry defaults so a snapshot is a complete reproducible
        # configuration even when a key has never been written to the DB.
        from polyflip.settings_registry import registry_defaults

        all_settings = {**registry_defaults(), **{r.key: r.value for r in rows}}
        return PresetService.sanitize_snapshot(all_settings)

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
        logger.info(
            "preset_saved", id=preset.id, name=name, preset_type=preset_type
        )
        return preset

    @staticmethod
    async def restore_preset(
        db: AsyncSession,
        preset_id: int,
        restored_by: str = "user",
    ) -> Tuple[int, Dict[str, str]]:
        """
        Применяет параметры из слепка.
        БЕЗОПАСНОСТЬ: Применяются ТОЛЬКО настройки из белого списка.
        Оперативные переключатели (TRADING_ENABLED и др.) исключены.
        """
        preset = await db.get(ConfigPreset, preset_id)
        if not preset or not preset.is_active:
            raise ValueError(f"Пресет {preset_id} не найден или был удалён")

        params = json.loads(preset.snapshot)
        safe_params = PresetService.sanitize_snapshot(params)

        now = datetime.now(timezone.utc)
        changed = 0
        updated_params = {}

        try:
            for key, value in safe_params.items():
                row = await db.get(RuntimeSettings, key)
                if row:
                    if row.value != str(value):
                        row.value = str(value)
                        row.updated_at = now
                        row.updated_by = (
                            f"preset_restore:{preset_id}:{restored_by}"
                        )
                        changed += 1
                        updated_params[key] = str(value)
                else:
                    db.add(
                        RuntimeSettings(
                            key=key,
                            value=str(value),
                            updated_at=now,
                            updated_by=(
                                f"preset_restore:{preset_id}:{restored_by}"
                            ),
                        )
                    )
                    changed += 1
                    updated_params[key] = str(value)
            await db.commit()
        except Exception:
            await db.rollback()
            raise
        logger.info(
            "preset_restored",
            id=preset_id,
            changed_keys=changed,
            restored_by=restored_by,
        )
        return changed, updated_params

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
          - Минимальный интервал между авто-слепками (1 час)
        """
        now = datetime.now(timezone.utc)

        # 1. Получаем существующие ATH-пресеты
        q = (
            select(ConfigPreset)
            .where(
                and_(
                    ConfigPreset.preset_type.in_(["ath_capital", "ath_pnl"]),
                    ConfigPreset.is_active == True,  # noqa: E712
                )
            )
            .order_by(ConfigPreset.created_at.desc())
        )
        ath_presets = (await db.execute(q)).scalars().all()

        # 2. Проверка интервала от последнего ATH
        if ath_presets:
            last_ath_time = ath_presets[0].created_at
            if now - last_ath_time < timedelta(hours=min_interval_hours):
                return None

        # 3. Вычисляем текущие максимумы
        prev_max_capital = max(
            (p.capital_at_save or 0.0 for p in ath_presets), default=0.0
        )
        prev_max_pnl = max(
            (p.pnl_at_save or 0.0 for p in ath_presets), default=0.0
        )

        is_capital_ath = (current_capital - prev_max_capital) >= min_pnl_diff
        is_pnl_ath = (current_pnl - prev_max_pnl) >= min_pnl_diff

        if not (is_capital_ath or is_pnl_ath):
            return None

        ptype = "ath_capital" if is_capital_ath else "ath_pnl"
        ts_str = now.strftime("%Y-%m-%d_%H-%M")
        name = f"🏆 ATH_{ptype.upper()}_{ts_str}"
        description = (
            f"Авто-слепок: Capital=${current_capital:.2f}, "
            f"PnL=${current_pnl:.2f}"
        )

        return await PresetService.save_preset(
            db=db,
            name=name,
            description=description,
            preset_type=ptype,
            capital_at_save=current_capital,
            pnl_at_save=current_pnl,
            created_by="system_ath",
        )
