# MRF v2 Baseline Report
## 2026-08-21

### Commit
- Branch: codex/market-regime-filter-implementation
- HEAD: 485e770 (mrf-v2: critical fixes from user review Steps 1-10)
- Parent: 3fec11e (merge main into MRF deployment branch)

### Git Status
- Working tree: clean (no uncommitted changes)

### Alembic
- Head: mrf_v2_columns_001
- Applied: yes (1 row in alembic_version)
- No duplicate migration files

### Containers (all UP ~3 hours)
| Container | Status |
|-----------|--------|
| polyflip_api | healthy |
| polyflip_scheduler | healthy |
| polyflip_execution_worker_paper | UP |
| polyflip_lgbm_training_worker | UP |
| polyflip_ai_lab_agent | UP |
| polyflip_db | UP 30h |

### MRF Mode
- MARKET_REGIME_FILTER_MODE = SHADOW (in runtime_settings)

### Tests
- 23 passed, 2 skipped (require deployed code with regime_config param)
- Skipped: test_evaluate_policy_with_regime_config, test_audit_regime_config_consistency

### /api/mrf/status
- Returns HTTP 500 (expected: container runs old code with pnl_usdc bug, not yet deployed)

### Trade Status Distribution (DB)
| position_status | status | count | total_pnl |
|-----------------|--------|-------|-----------|
| CANCELLED | SKIPPED | 2533 | 0 |
| CANCELLED | SUCCESS | 1 | 0 |
| CLOSED | SUCCESS | 4919 | -89.33 |
| ENTRY_FAILED | FAILED | 141 | 0 |
| OPEN | SKIPPED | 2917 | 0 |
| OPEN | SUCCESS | 8 | 0 |
| OPENING | FAILED | 21 | 0 |
| RESOLVED_LOST | SUCCESS | 35 | -35.29 |
| RESOLVED_REDEEMABLE | SUCCESS | 25 | 33.27 |

### Terminal statuses for PnL: CLOSED, RESOLVED_LOST, RESOLVED_REDEEMABLE
