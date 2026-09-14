# Walkthrough

This update fixes the most critical blocking issues preventing live execution:
- Updated EIP-712 signer schema for CLOB V2, removing V1 legacy fields and changing types (metadata, builder to bytes32).
- Replaced seconds with milliseconds in the timestamp for order signing.
- Corrected the CTF Exchange address.
- Removed the dangerous mock fallback in `06_run_live_calibration.py`.
- Replaced the single-run smoke test in live calibration with an infinite `while True` loop that repeatedly places and cancels orders for a continuous continuous testing cycle.

Other issues raised in the bug report remain open and will be evaluated in subsequent passes.
