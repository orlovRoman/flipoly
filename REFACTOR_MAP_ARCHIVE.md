## engine.py dependency map

### DB reads
- Lines 93-123: load `RuntimeSettings` (global settings)
- Lines 129-135: load `RuntimeSettings` (per-asset thresholds)
- Lines 167-183: load `LiveMarket` (active markets within time window)
- Lines 185-200: load `TradeHistory` (calculate daily PnL)
- Lines 227-231: load `TradeHistory` (check existing trades for favorite mode)
- Lines 354-356: load `ModelRegistry` (load active models)
- Lines 382-384: load `TradeHistory` (check existing trades for ML mode)

### DB writes
- Lines 53-83: `save_or_update_skipped_trade()` adds/updates `TradeHistory`
- Lines 332-348: inserts `TradeHistory` (PURE_FAVORITE success)
- Lines 587-606: inserts `TradeHistory` (ML_TREND success)
- Lines 613-628: inserts `SlippageLog` (ML_TREND success)
- Lines 693-712: inserts `TradeHistory` (OUTSIDER success)
- Lines 719-734: inserts `SlippageLog` (OUTSIDER success)

### External API calls  
- Lines 262-264: `api_client.get_market_prices(token_to_buy)` (fetch ask for PURE_FAVORITE)
- Lines 321-328: `trader.execute_trade(...)` (execute PURE_FAVORITE)
- Lines 420-421: `api_client.get_market_prices(yes_token_id)` (fetch YES prices for ML_TREND)
- Lines 504-506: `api_client.get_market_prices(no_token_id)` (fetch NO prices if model chose NO)
- Lines 574-581: `trader.execute_trade(...)` (execute ML_TREND)
- Lines 632-634: `api_client.get_market_prices(no_token_id)` (fetch NO prices for OUTSIDER)
- Lines 682-689: `trader.execute_trade(...)` (execute OUTSIDER)

### Pure logic blocks (кандидаты для extraction)
- Lines 17-18: dependencies to move (`compute_kelly_fraction`, `compute_dead_zone`, `add_derived_features`)
- Lines 439-444: feature vector building (using `add_derived_features` on single row dataframe)
- Lines 476-482: dead zone calculation (`compute_dead_zone`)
- Lines 548-554: Kelly fraction calculation (`compute_kelly_fraction`)
- Lines 206-352: PURE_FAVORITE decision logic (threshold checks, edge bounds)
- Lines 483-630: ML_TREND decision logic (threshold checks, model prediction check)
- Lines 631-737: OUTSIDER decision logic (TRADE_ON_FLIP check, limits)
- Lines 530-535: edge calculation (implied prob from buy_price)
- Lines 150-165: settings extraction and default values

### Caches
- Lines 349-350, 629-630, 735-736: `invalidate_stats_cache()`, `invalidate_dashboard_cache()`
