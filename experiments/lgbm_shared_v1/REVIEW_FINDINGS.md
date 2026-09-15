# REVIEW_FINDINGS — 2026-09-15 rev2

- direction INVALIDATED_PENDING_RERUN (E4-E6 stub)
- feature INVALIDATED_PENDING_RERUN (F1 now 79 valid, F3 79x79 valid, verdicts preliminary UNSUPPORTED)
- shared CONDITIONAL_PASS (parquet 48a239b6, 1497818 rows, 21757 markets)
- E1 repro_status now in e1_summary.json: 3/9/45/16
- opportunity_id to be defined as market_id+policy_id+window, clustering by market/day
