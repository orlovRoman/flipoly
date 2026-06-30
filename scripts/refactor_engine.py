import re
import os

engine_path = "polyflip/trading/engine.py"

with open(engine_path, "r", encoding="utf-8") as f:
    code = f.read()

# ADD IMPORTS
import_str = """
from polyflip.trading.feature_builder import MarketSignal
from polyflip.trading.decision_logic import decide_favorite, decide_ml_trend, decide_outsider
"""
if "decide_favorite" not in code:
    code = code.replace("from polyflip.db.models", import_str.strip() + "\nfrom polyflip.db.models")

# REPLACE PURE FAVORITE BLOCK
# lines 254 to 311 (inclusive, up to actual_bet_size = bet_size)
fav_pattern = re.compile(
    r"# Определяем фаворита по цене из БД.*?# Фиксированная ставка, Kelly не применяется\s*actual_bet_size = bet_size\s*num_shares = round\(actual_bet_size / buy_price, 2\)", 
    re.DOTALL
)

fav_replacement = """# REFACTORED: was inline
                signal = MarketSignal(
                    asset=market.asset,
                    mid_price=market.current_yes_price,
                    spread=market.current_spread or 0.01,
                    volume_5min=market.volume_5min or 0.0,
                    price_velocity=market.price_velocity or 0.0,
                    hour_of_day=start_time.hour,
                    time_left_min=time_left_sec / 60.0
                )
                decision_obj = decide_favorite(signal, settings_db)
                if decision_obj.action == "SKIP":
                    logger.info("favorite_mode_skipped", reason=decision_obj.reason)
                    await save_or_update_skipped_trade(
                        db_session, market,
                        decision_obj.reason,
                        0.0, None, start_time,
                        existing_skipped=existing_skipped
                    )
                    continue

                decision = decision_obj.action.replace("BUY_", "")
                buy_price = decision_obj.buy_price
                actual_bet_size = decision_obj.bet_size_usdc
                token_to_buy = market.yes_token_id if decision == "YES" else market.no_token_id
                
                num_shares = round(actual_bet_size / buy_price, 2)"""

code = fav_pattern.sub(fav_replacement, code)

# ML TREND AND OUTSIDER BLOCK
# The ML decision logic starts at line 462: "# Логика принятия решения"
# up to the end of the OUTSIDER block, just before the "except Exception as e:"
ml_pattern = re.compile(
    r"# Логика принятия решения\s*decision = None.*?invalidate_dashboard_cache\(\)\s*else:\s*logger\.info\(\"trade_skipped\".*?kelly_multiplier=None\s*\)",
    re.DOTALL
)

ml_replacement = """# Логика принятия решения
                signal = MarketSignal(
                    asset=market.asset,
                    mid_price=fresh_yes_price,
                    spread=fresh_spread,
                    volume_5min=market.volume_5min,
                    price_velocity=market.price_velocity,
                    hour_of_day=start_time.hour,
                    time_left_min=time_left_sec / 60.0
                )

                decision_obj = decide_ml_trend(signal, p_flip, settings_db)
                
                if decision_obj.action == "SKIP" and trade_on_flip:
                    decision_obj = decide_outsider(signal, p_flip, settings_db)

                if decision_obj.action == "SKIP":
                    logger.info("trade_skipped", reason=decision_obj.reason, p_flip=p_flip)
                    await save_or_update_skipped_trade(
                        db_session, market, decision_obj.reason, p_flip, model_ver, start_time,
                        existing_skipped=existing_skipped,
                        kelly_fraction=None, kelly_multiplier=None
                    )
                    continue

                decision = decision_obj.action.replace("BUY_", "")
                buy_price = decision_obj.buy_price
                actual_bet_size = decision_obj.bet_size_usdc
                token_to_buy = yes_token_id if decision == "YES" else no_token_id
                num_shares = round(actual_bet_size / buy_price, 2)
                edge = decision_obj.edge
                kelly_f = decision_obj.kelly_fraction
                
                trade_res = await trader.execute_trade(
                    market_id=market.market_id,
                    token_id=token_to_buy,
                    side="BUY",
                    price=buy_price,
                    size=num_shares
                )
                
                if existing_skipped:
                    await db_session.delete(existing_skipped)

                history = TradeHistory(
                    market_id=market.market_id,
                    asset=market.asset,
                    outcome_bought=decision,
                    amount_usdc=trade_res.get("executed_usdc", actual_bet_size),
                    executed_price=trade_res.get("executed_price", buy_price),
                    predicted_flip_prob=p_flip,
                    active_features=f"{active_features_str.strip().rstrip(',')},{decision_obj.strategy_type.lower()}" if (active_features_str and active_features_str.strip().rstrip(',')) else decision_obj.strategy_type.lower(),
                    model_version=model_ver,
                    status=trade_res.get("status", "FAILED"),
                    error_msg=trade_res.get("error_msg"),
                    mode=trade_res.get("mode", "PAPER"),
                    kelly_fraction=round(kelly_f, 4) if kelly_f is not None else None,
                    kelly_multiplier=1.0,
                    edge=round(edge, 4) if edge is not None else None,
                    created_at=start_time
                )
                db_session.add(history)
                await db_session.flush()

                if trade_res.get("status") == "SUCCESS":
                    exec_p = trade_res.get("executed_price", buy_price)
                    slip = round(exec_p - buy_price, 6)
                    slip_pct = round(slip / buy_price * 100, 4) if buy_price > 0 else 0.0
                    slip_cost = round(slip * (actual_bet_size / exec_p), 4) if exec_p > 0 else 0.0

                    slippage_record = SlippageLog(
                        trade_id=history.id,
                        market_id=market.market_id,
                        asset=market.asset,
                        outcome_bought=decision,
                        expected_price=buy_price,
                        executed_price=exec_p,
                        slippage=slip,
                        slippage_pct=slip_pct,
                        bet_size_usdc=actual_bet_size,
                        slippage_cost_usdc=slip_cost,
                        mode=trade_res.get("mode", "PAPER"),
                        created_at=start_time,
                    )
                    db_session.add(slippage_record)

                invalidate_stats_cache()
                invalidate_dashboard_cache()"""

code = ml_pattern.sub(ml_replacement, code)

with open(engine_path, "w", encoding="utf-8") as f:
    f.write(code)

print("Refactored engine.py")
