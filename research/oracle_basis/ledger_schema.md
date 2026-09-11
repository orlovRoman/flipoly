# Oracle-Basis Ledger Schema v1 (`backtest_ledger.parquet`)

Preliminary (non-execution) ledger for hold-to-settlement C contracts. All money
fields are **scaled integers** — no floats anywhere in the file.

## Units

| Field | Type | Unit |
|---|---|---|
| `p_up_e6` | int64 | probability × 10⁶ |
| `ask_price_e12` | int64 | ask price × 10¹² |
| `quantity` | int64 | contracts (whole units) |
| `fee_micros`, `other_costs_micros`, `pnl_micros` | int64 | USDC × 10⁶ |
| `decision_ms`, `assumed_fill_ms` | int64 | Unix milliseconds UTC |
| `spot_age_ms` | int64, nullable | ms between checkpoint and spot receipt |
| `horizon_sec` | int64 | seconds before market end |

Rounding: half away from zero on the exact Decimal value (fee formula
`C · 0.07 · a · (1−a)` computed in Decimal, then scaled).

## Columns (fixed order)

`market_id: string`, `asset: string`, `horizon_sec: int64`, `side: string[UP|DOWN]`,
`decision_ms: int64`, `assumed_fill_ms: int64`, `p_up_e6: int64`,
`model_version: string`, `protocol_version: string`, `ask_price_e12: int64`,
`quantity: int64`, `fee_micros: int64`, `other_costs_micros: int64`,
`fee_scheme: string`, `outcome_up: bool`, `pnl_micros: int64`,
`price_sources: string`, `spot_status: string`, `spot_age_ms: int64?`,
`gap_bucket: string`, `policy: string`, `post_latency_fill: bool`,
`depth_limited: bool`, `ledger_schema_version: string (= "1")`.

## Rules

- DOWN rows require an observed `ask_down`; `1 − ask_up` is never stored.
- `assumed_fill_ms >= decision_ms`; rows without post-latency arrival data set
  `post_latency_fill = false` and are optimistic estimates, not executability proof.
- `outcome_up` is the real Polymarket outcome; unresolved markets never enter the ledger.
- `price_sources` names every quote used (e.g. `book@decision+arrival@+900ms`).
- `model_version` + `protocol_version` pin the producing configuration.
