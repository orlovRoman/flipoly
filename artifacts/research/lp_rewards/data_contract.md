# Data Contract: Polymarket LP Rewards Research (v0.1)

## 1. Storage Layout & Path Isolation
All high-volume and stateful data MUST reside on `D:\flipoly-research\lp-rewards\`:
- L2 Snapshots: `D:\flipoly-research\lp-rewards\l2_snapshots\{condition_id}\{date}.parquet` (ZSTD)
- Public Trades: `D:\flipoly-research\lp-rewards\public_trades\{condition_id}\{date}.parquet` (ZSTD)
- Virtual Orders & Fills: `D:\flipoly-research\lp-rewards\simulated_ledger\{date}.parquet`
- Daily Evaluations: `D:\flipoly-research\lp-rewards\daily_evaluations\{date}.json`
- SQLite Event Log: `D:\flipoly-research\lp-rewards\events.db`

## 2. Mathematical Precision
- All monetary amounts (prices, sizes, rewards, fees, cashflows) MUST be represented using `decimal.Decimal` with precision >= 28.
- Float representations are strictly prohibited for balance accounting, PnL evaluation, and reward normalization.

## 3. Schema Definitions

### L2 Reward Zone Snapshot
| Column | Type | Description |
|---|---|---|
| `timestamp_ns` | int64 | UTC timestamp in nanoseconds |
| `condition_id` | string | Polymarket market condition identifier |
| `asset_id` | string | Token ID (YES or NO) |
| `side` | string | `bid` or `ask` |
| `price` | decimal(10,4) | Price in USDC ($0.0001 - $1.0000) |
| `size` | decimal(18,4) | Available size at price level |
| `valid_from_ns` | int64 | Timestamp when this state became valid |
| `valid_to_ns` | int64 | Timestamp when next update arrived |
| `order_age_sec` | decimal(10,2) | Age of this level state |

### Public Trade Record
| Column | Type | Description |
|---|---|---|
| `timestamp_ns` | int64 | UTC timestamp in nanoseconds |
| `condition_id` | string | Polymarket market condition identifier |
| `asset_id` | string | Token ID |
| `side` | string | Aggressor side: `BUY` or `SELL` |
| `price` | decimal(10,4) | Executed price |
| `size` | decimal(18,4) | Executed quantity |

### Virtual FSM Fill Record
| Column | Type | Description |
|---|---|---|
| `fill_id` | string | UUID of the simulated fill |
| `order_id` | string | ID of the resting order |
| `timestamp_ns` | int64 | Fill timestamp |
| `condition_id` | string | Market condition ID |
| `asset_id` | string | Token ID (YES / NO) |
| `side` | string | `BUY` or `SELL` |
| `price` | decimal(10,4) | Executed price |
| `size` | decimal(18,4) | Executed size |
| `queue_depletion_ratio` | decimal(6,4) | Pessimistic queue fill ratio |
| `taker_fee_paid` | decimal(18,6) | Fee deducted (if forced exit) |
| `markout_5s` | decimal(10,4) | Midpoint 5 seconds post-fill |
| `markout_60s` | decimal(10,4) | Midpoint 60 seconds post-fill |
| `markout_15m` | decimal(10,4) | Midpoint 15 minutes post-fill |

## 4. Key Metrics Contract
- **$R_{100\_calendar}$**: $\frac{\text{Net PnL}}{\text{Completed Full UTC Days}}$ on strictly $100 allocated capital base.
- **Net PnL**: $\text{LP Rewards} + \text{Maker Rebates} + \text{Realized Trading PnL} + \text{Inventory MTM (executable liquidation)} - \text{Taker Fees} - \text{Merging Gas/Costs}$.
