# AI Lab Phase 9 — Activation, Deployment Revisions & Rollback

Phase 9 implements secure, human-in-the-loop activation of models from `SHADOW` or `PENDING_APPROVAL` to `ACTIVE`, content-addressed `DeploymentRevision` management, and atomic instant rollbacks.

## Key Principles

1. **Autonomous Boundary**:
   - The laboratory can propose activations (`propose_live_deployment` via `/runs/{id}/approval`), but cannot self-activate.
   - Activation requires an explicit administrative action (`POST /api/ai-lab/approvals/{id}/approve`).
2. **Server-Side Structured Diff**:
   - The server automatically calculates the differences between candidate model/parameters and the current active baseline in `ModelRegistry`.
   - Compares: model family, features, decision thresholds, median PnL, trade counts, and max drawdown.
3. **Immutable Deployment Revisions**:
   - Every proposal creates a `DeploymentRevision` with a deterministic SHA-256 `manifest_hash` and `parent_id` linking to the previous active revision.
4. **Transactional Pointer Switching with Row-Level Locks**:
   - Model activation switches `is_active` flags in `ModelRegistry` using `SELECT ... FOR UPDATE` row locks.
   - Live open positions (`OpenPositions`, orders, exchange limits) are **never modified or liquidated**.
5. **Instant Rollback**:
   - `POST /api/ai-lab/deployments/rollback` reverts active model pointers to the parent revision (or specified revision).
6. **Cryptographic Audit Log**:
   - All events (`CREATED`, `SHADOW_ASSIGNED`, `APPROVED`, `ACTIVATED`, `REJECTED`, `ROLLED_BACK`) are recorded in an append-only hash chain in `deployment_events`.

## API Endpoints

### Request Live Activation
```http
POST /api/ai-lab/runs/{run_id}/approval
Content-Type: application/json
X-API-Key: <api-key>

{
  "requested_action": "ACTIVATE"
}
```

### Approve and Activate
```http
POST /api/ai-lab/approvals/{approval_id}/approve
Content-Type: application/json
X-API-Key: <api-key>

{
  "actor": "admin",
  "reason": "Approved based on +2.5% median Polymarket-OOT PnL"
}
```

### Reject Proposal
```http
POST /api/ai-lab/approvals/{approval_id}/reject
Content-Type: application/json
X-API-Key: <api-key>

{
  "actor": "risk_officer",
  "reason": "Drawdown exceeds acceptable threshold"
}
```

### Rollback Revision
```http
POST /api/ai-lab/deployments/rollback
Content-Type: application/json
X-API-Key: <api-key>

{
  "target_revision_id": 1,
  "actor": "admin",
  "reason": "Market regime change rollback"
}
```

### Inspect Revisions
```http
GET /api/ai-lab/deployments/revisions
GET /api/ai-lab/deployments/revisions/{revision_id}
```
