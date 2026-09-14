# Live Configuration Deep Audit & Fixes

## 1. Signer Schema & Client Legacy Fixes
- `execution.py`: Updated `CTF_EXCHANGE_ADDRESS` to the official V2 address (`0xE11118001712aA868d4aB713A2c8f85fBB9161aB`).
- `execution.py`: Verified V1 signer fields (`taker`, `expiration`, `nonce`, `feeRateBps`) are successfully removed, `timestamp` uses milliseconds, and `metadata`/`builder` correctly use `bytes32`.

## 2. PUSD On-Chain Validation
- `execution.py`: Replaced manual fallback inputs with full on-chain RPC `eth_call` lookups to Polygon for pUSD (`0xC011a7E12a19f7b1f670d46f03b03f3342e82dfb`).
- Implemented `get_pusd_balance_onchain` and `get_pusd_allowance_onchain`.
- Renamed USDC references to pUSD.

## 3. Mock Client Removal
- `06_run_live_calibration.py`: Ensure that actual initialization exceptions in `ClobClient` kill the process (Fail-closed behavior) rather than gracefully falling back to a `MockClobClient`.
- Implemented a continuous `while True` loop that repeatedly evaluates FSM strategies rather than a smoke test with immediate teardown.

## 4. Competitive Logic
- `scoring.py`: Verified that `subtract_orders` actively removes our local positions from the public orderbook before assessing `Q_one` and `Q_two`, preventing double counting of virtual liquidity.
- `03_run_shadow_collector.py`: Changed `quote_hours` to calculate based on actual aggregated quoting intervals (`fsm_ticks` multiplied by `interval_sec`) rather than raw process runtime.

## 5. Reward Reconciliation
- `reward_calibration.py`: Enforced presence of full authenticated L2 headers (`POLY_ADDRESS`, `POLY_SIGNATURE`, `POLY_TIMESTAMP`, `POLY_PASSPHRASE`) before querying `/rewards/user`, throwing `PermissionError` if missing.
- Added pagination (`next_cursor`) looping for `/rewards/user` to avoid silent truncation of payout records.
- `08_reconcile_rewards.py`: Fixed exact date matching by truncating the API ISO datetime to the `YYYY-MM-DD` prefix.
- Aggregated actual rewards across all markets to correctly compare with the daily combined `total_rewards_accrued` from shadow evaluations.
- `09_evaluate_gate_b.py`: Set default `mean_prediction_error = Decimal("1.0")` to guarantee failure when reconciliation logs are missing.
