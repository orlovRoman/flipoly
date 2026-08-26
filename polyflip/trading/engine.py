import structlog
import os
from dataclasses import replace
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from polyflip.collector.client import PolymarketClient
from polyflip.constants import TRADING_MODE_COMBINED
from polyflip.trading.settings_loader import load_trading_settings
from polyflip.trading.trading_config import parse_trading_settings
from polyflip.trading.market_loader import load_eligible_markets
from polyflip.trading.market_guards import check_market_guards
from polyflip.trading.decision_runners import decide_combined_mode
from polyflip.trading.pre_trade_validator import validate_pre_trade
from polyflip.trading.trade_recorder import (
    execute_and_record,
    save_or_update_skipped_trade,
    EnqueueRejected,
)
from polyflip.crypto.candle_repository import get_recent_candles

logger = structlog.get_logger(__name__)

_crypto_predictor = None
_crypto_predictor_initialized = False


def _get_crypto_predictor():
    global _crypto_predictor, _crypto_predictor_initialized
    if not _crypto_predictor_initialized:
        try:
            from polyflip.crypto.predictor import CryptoPredictor

            _crypto_predictor = CryptoPredictor()
        except Exception as e:
            logger.error("crypto_predictor_init_failed", error=str(e))
            _crypto_predictor = None
        finally:
            _crypto_predictor_initialized = True
    return _crypto_predictor


_ACTIVE_MARKETS = set()


async def _record_skip(
    db_session,
    market,
    reason,
    decision_res,
    start_time,
    existing_skipped,
    asset_mode,
    cfg,
) -> None:
    from polyflip.trading.trade_recorder import _get_trade_active_features

    dec_details = (
        decision_res.decision_obj.decision_details
        if (decision_res and decision_res.decision_obj)
        else None
    )
    await save_or_update_skipped_trade(
        db_session,
        market,
        reason,
        decision_res.p_flip if decision_res else 0.0,
        decision_res.model_ver if decision_res else None,
        start_time,
        existing_skipped,
        edge=decision_res.edge if decision_res else None,
        active_features=_get_trade_active_features(
            asset_mode,
            cfg.active_features_str,
            decision_res.decision_obj if decision_res else None,
            market.asset,
        ),
        lgbm_metadata=decision_res.lgbm_metadata if decision_res else None,
        market_role=dec_details.get("market_role") if dec_details else None,
        model_key=decision_res.used_model_key if decision_res else None,
        confirm_model_key=decision_res.confirm_model_key if decision_res else None,
        confirm_model_version=decision_res.confirm_model_version
        if decision_res
        else None,
        decision_details=dec_details,
        direction_value=decision_res.decision_obj.direction_value
        if (decision_res and decision_res.decision_obj)
        else None,
    )


async def trade_worker_cycle(db_session: AsyncSession, api_client: PolymarketClient):
    """
    Фоновый процесс торгового движка (оркестратор).
    """
    start_time = datetime.now(timezone.utc)
    execution_mode = os.getenv("EXECUTION_MODE", "PAPER")

    try:
        raw_settings = await load_trading_settings(db_session)
        overlay_ids: list[int] = []
        if execution_mode.strip().upper() == "PAPER":
            from polyflip.ai_lab.paper_overlay import resolve_paper_runtime_settings

            raw_settings, overlay_ids = await resolve_paper_runtime_settings(
                db_session, raw_settings, now=start_time
            )
        cfg = parse_trading_settings(raw_settings)

        if not cfg.trading_enabled:
            logger.info("trading_disabled_skipping", mode=cfg.trading_mode)
            return

        markets = await load_eligible_markets(db_session, cfg, start_time)
        if markets is None or not markets:
            return

        from polyflip.trading.ml_inference import populate_models_cache

        await populate_models_cache(db_session)

        for market in markets:
            if market.market_id in _ACTIVE_MARKETS:
                continue
            _ACTIVE_MARKETS.add(market.market_id)
            try:
                asset_mode = cfg.trading_mode.lower() if cfg.trading_mode else ""
                asset_min_edge = cfg.outs_min_edge
                asset_max_price = cfg.trade_max_price

                end_time_utc = market.end_time_est
                if end_time_utc.tzinfo is None:
                    end_time_utc = end_time_utc.replace(tzinfo=timezone.utc)
                time_left_sec = (end_time_utc - start_time).total_seconds()

                guard_res = await check_market_guards(
                    db_session, market, cfg, asset_mode, time_left_sec, start_time
                )

                if not guard_res.passed:
                    if guard_res.skip_reason and guard_res.skip_reason not in (
                        "guard: Time left <= 0",
                        "guard: Trade already exists",
                    ):
                        await save_or_update_skipped_trade(
                            db_session,
                            market,
                            guard_res.skip_reason,
                            p_flip_val=0.0,
                            model_version=None,
                            start_time=start_time,
                            existing_skipped=guard_res.existing_skipped,
                        )
                    continue

                existing_skipped = guard_res.existing_skipped
                decision_res = None

                try:
                    from polyflip.trading.ml_inference import get_models_cache

                    models_cache = get_models_cache()
                    decision_res = await decide_combined_mode(
                        db_session,
                        api_client,
                        market,
                        cfg,
                        raw_settings,
                        models_cache,
                        _get_crypto_predictor(),
                        start_time,
                        time_left_sec,
                        existing_skipped,
                        execution_mode=execution_mode,
                    )
                    if (
                        overlay_ids
                        and decision_res is not None
                        and decision_res.decision_obj is not None
                    ):
                        details = dict(decision_res.decision_obj.decision_details or {})
                        details["ai_lab_overlay_ids"] = list(overlay_ids)
                        details["ai_overlay_ids"] = list(overlay_ids)
                        decision_res.decision_obj = replace(
                            decision_res.decision_obj, decision_details=details
                        )
                except Exception as e:
                    logger.exception(
                        "decision_logic_error", market=market.market_id, error=str(e)
                    )
                    await save_or_update_skipped_trade(
                        db_session,
                        market,
                        f"engine: Error calculating prediction: {e}",
                        0.0,
                        None,
                        start_time,
                        existing_skipped,
                    )
                    continue

                if (
                    not decision_res
                    or not decision_res.decision_obj
                    or decision_res.decision_obj.action == "SKIP"
                ):
                    reason = decision_res.skip_reason if decision_res else "SKIP"
                    await _record_skip(
                        db_session,
                        market,
                        reason,
                        decision_res,
                        start_time,
                        existing_skipped,
                        asset_mode,
                        cfg,
                    )
                    continue

                validation = await validate_pre_trade(
                    db_session,
                    api_client,
                    market,
                    decision_res.decision_obj,
                    cfg,
                    asset_mode,
                    asset_min_edge,
                    asset_max_price,
                    decision_res.p_flip,
                    decision_res.model_ver,
                )

                if not validation.valid:
                    await _record_skip(
                        db_session,
                        market,
                        validation.skip_reason,
                        decision_res,
                        start_time,
                        existing_skipped,
                        asset_mode,
                        cfg,
                    )
                    continue

                try:
                    await execute_and_record(
                        db_session,
                        market,
                        decision_res.decision_obj,
                        validation,
                        asset_mode,
                        cfg.active_features_str,
                        decision_res.p_flip,
                        decision_res.model_ver,
                        cfg,
                        existing_skipped,
                        start_time,
                        lgbm_metadata=decision_res.lgbm_metadata
                        if decision_res
                        else None,
                        model_key=decision_res.used_model_key if decision_res else None,
                        confirm_model_key=decision_res.confirm_model_key
                        if decision_res
                        else None,
                        confirm_model_version=decision_res.confirm_model_version
                        if decision_res
                        else None,
                    )
                except EnqueueRejected as exc:
                    await _record_skip(
                        db_session,
                        market,
                        f"engine: Execution not enqueued: {exc}",
                        decision_res,
                        start_time,
                        existing_skipped,
                        asset_mode,
                        cfg,
                    )
                    continue
            finally:
                _ACTIVE_MARKETS.discard(market.market_id)

    except Exception as e:
        logger.exception("trade_worker_error", error=str(e))
    finally:
        try:
            await db_session.commit()
        except Exception as e_commit:
            logger.error("failed_to_commit_in_finally", error=str(e_commit))
