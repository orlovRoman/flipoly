# Source mapping (step 5) — every required field links to a REAL column/API or is marked MISSING

> Rule: field names are never guessed. Fill from the actual schema; keep MISSING rows visible.

| Required field | Real table/column or API | Status | Notes |
|---|---|---|---|
| market_id | `markets.id` (Gamma `/events`) | MAPPED | canonical market key |
| asset | derived from event title/tags (BTC/ETH/SOL/XRP/DOGE) | MAPPED | see `PolymarketClient.get_active_15m_markets` |
| start_at / end_at | `markets.endDate` + 15m window | TODO_VERIFY | confirm start = end-15m for 15m contracts |
| up_token_id / down_token_id | `markets.clobTokenIds[0..1]` | MAPPED | YES=UP leg, NO=DOWN leg |
| strike_value | `_canonical_strike(market, event)` candidates | MAPPED_WITH_GAPS | opening/Chainlink strike; Binance fallback FORBIDDEN in canonical sample |
| strike_source | new registry field | TODO | canonical_confirmed | retrospective | binance_proxy | unknown |
| strike_available_at | TODO | MISSING | must prove availability before decision (step 8) |
| resolution_rule | TODO | MISSING | price source, averaging, time cut, equality rule per family (step 9) |
| resolution_source | TODO | MISSING | Gamma resolution payload |
| actual_outcome | `extract_final_outcome` / settlement | TODO_VERIFY | real market resolution only; own Binance-vs-strike comparison NEVER substitutes |
| outcome_available_at | TODO | MISSING | needed for split causality |
| quotes both sides + depth | CLOB `/book` bids/asks per token | MAPPED | store event_at/received_at/age/token id; NO never via 1-YES |
| underlying price | TODO | MISSING | contract-consistent source; Binance separate proxy column |
| fee (applicable commission) | TODO | MISSING | until known: net_pnl=null + signed scenarios |

## Strike credibility classes (step 8)
- `canonical_confirmed`: confirmed source, provably available at decision.
- `retrospective`: canonically restored, availability separately proven.
- `binance_proxy`: NEVER in the main canonical sample.
- `unknown`: excluded from main sample, counted in coverage.

## Resolution families (step 9)
Document per family: price source, averaging method, time boundary, equality (price==strike) rule.
