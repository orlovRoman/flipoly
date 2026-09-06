# Code Audit Log

This file tracks the automated Copilot Review audit initiated on 2026-07-24.

## Known issues found during manual review

1. **position_sizing.py** — ZeroDivisionError when `max_edge == min_edge`
2. **pre_trade_validator.py** — stale `p_win` in FAVORITE branch uses `market.current_yes_price` instead of `fresh_ask`
3. **decision_runners.py** — COMBINED bet penalty bypassed by `max(reduced_bet, min_bet)` using full `min_bet` instead of scaled floor
4. **predictor.py** — thundering herd: no `asyncio.Lock()` on model load, multiple coroutines deserialize same model blob simultaneously
5. **trainer.py** — `epsilon_quantile` dashboard default conflicts with code default (0.04 vs 0.70)
