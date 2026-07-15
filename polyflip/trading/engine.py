import structlog
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from polyflip.collector.client import PolymarketClient
from polyflip.trading.trader import PolyTrader
from polyflip.constants import TRADING_MODE_CRYPTO, TRADING_MODE_ML, TRADING_MODE_FAVORITE

from polyflip.trading.settings_loader import load_trading_settings
from polyflip.trading.trading_config import parse_trading_settings
from polyflip.trading.market_loader import load_eligible_markets
from polyflip.trading.market_guards import check_market_guards
from polyflip.trading.decision_runners import decide_favorite_mode, decide_ml_mode
from polyflip.trading.pre_trade_validator import validate_pre_trade
from polyflip.trading.trade_recorder import execute_and_record, save_or_update_skipped_trade

logger = structlog.get_logger(__name__)

_crypto_predictor = None

def _get_crypto_predictor():
    global _crypto_predictor
    if _crypto_predictor is None:
        try:
            from polyflip.crypto.predictor import CryptoPredictor
            _crypto_predictor = CryptoPredictor()
        except ImportError:
            _crypto_predictor = None
    return _crypto_predictor


async def trade_worker_cycle(db_session: AsyncSession, trader: PolyTrader, api_client: PolymarketClient):
    """
    Фоновый процесс торгового движка (оркестратор).
    """
    start_time = datetime.now(timezone.utc)
    
    try:
        raw_settings = await load_trading_settings(db_session)
        cfg = parse_trading_settings(raw_settings)
        
        if not cfg.trading_enabled:
            logger.info("trading_disabled_skipping", mode=cfg.trading_mode)
            return

        markets = await load_eligible_markets(db_session, cfg, start_time)
        if markets is None or not markets:
            return

        for market in markets:
            asset_mode = raw_settings.get(f"TRADING_MODE_{market.asset.upper()}", cfg.trading_mode)
            asset_min_edge = float(raw_settings.get(f"MIN_EDGE_{market.asset.upper()}", cfg.min_edge))
            asset_max_price = float(raw_settings.get(f"TRADE_MAX_PRICE_{market.asset.upper()}", cfg.trade_max_price))

            end_time_utc = market.end_time_est
            if end_time_utc.tzinfo is None:
                end_time_utc = end_time_utc.replace(tzinfo=timezone.utc)
            time_left_sec = (end_time_utc - start_time).total_seconds()

            guard_res = await check_market_guards(db_session, market, cfg, asset_mode, time_left_sec, start_time)
            
            if not guard_res.passed:
                if guard_res.skip_reason and guard_res.skip_reason != "Time left <= 0":
                    await save_or_update_skipped_trade(
                        db_session, market, guard_res.skip_reason, p_flip_val=0.0,
                        model_version=None, start_time=start_time,
                        existing_skipped=guard_res.existing_skipped
                    )
                continue

            existing_skipped = guard_res.existing_skipped
            decision_res = None
            
            try:
                if asset_mode == TRADING_MODE_ML:
                    from polyflip.trading.ml_inference import get_models_cache
                    models_cache = get_models_cache()
                    decision_res = await decide_ml_mode(
                        db_session, api_client, market, cfg, models_cache, _get_crypto_predictor(),
                        start_time, time_left_sec, existing_skipped
                    )
                elif asset_mode == TRADING_MODE_FAVORITE:
                    decision_res = await decide_favorite_mode(
                        market, cfg, asset_min_edge, asset_max_price, start_time, time_left_sec
                    )
                elif asset_mode == TRADING_MODE_CRYPTO:
                    try:
                        from polyflip.trading.decision_runners import decide_crypto_mode
                        decision_res = await decide_crypto_mode(
                            api_client, market, cfg, raw_settings, _get_crypto_predictor(), start_time, time_left_sec
                        )
                    except ImportError:
                        pass
            except Exception as e:
                logger.exception("decision_logic_error", market=market.market_id, error=str(e))
                await save_or_update_skipped_trade(
                    db_session, market, f"Error calculating prediction: {e}", 0.0, None, start_time, existing_skipped
                )
                continue
                
            if not decision_res or not decision_res.decision_obj or decision_res.decision_obj.action == "SKIP":
                skip_reason = decision_res.skip_reason if decision_res else "SKIP"
                p_flip = decision_res.p_flip if decision_res else 0.0
                edge = decision_res.edge if decision_res else None
                model_ver = decision_res.model_ver if decision_res else None
                await save_or_update_skipped_trade(
                    db_session, market, skip_reason or "SKIP", p_flip, model_ver, start_time, existing_skipped, edge
                )
                continue

            validation = await validate_pre_trade(
                api_client, market, decision_res.decision_obj, cfg, asset_mode,
                asset_min_edge, asset_max_price, decision_res.p_flip, decision_res.model_ver
            )

            await execute_and_record(
                db_session, trader, market, decision_res.decision_obj, validation,
                asset_mode, cfg.active_features_str, decision_res.p_flip, decision_res.model_ver,
                cfg, existing_skipped, start_time
            )

    except Exception as e:
        logger.exception("trade_worker_error", error=str(e))
    finally:
        try:
            await db_session.commit()
        except Exception as e_commit:
            logger.error("failed_to_commit_in_finally", error=str(e_commit))
