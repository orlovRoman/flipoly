# Canonical model comparison — runbook (local only)

Six forecast variants on REAL Polymarket outcomes; pick one challenger to M1;
check trading economics on a LATER period. M1 stays the control. CT is checked
separately as an entry filter (never as a forecast variant).

## Where things run (annotation 1)
- LOCAL Windows disk: dataset prep, features, train, bootstrap, economics.
- SERVER: collection + limited export only (chunked by dates/markets).
- NEVER run `train`/`bootstrap`/feature jobs on the server or from `\\sshfs...`.
  `guards.assert_train_allowed()` refuses SSHFS paths and `POLYFLIP_ROLE=server`.

## Layout
- `polyflip/research/canonical_models/` — library (no scheduler/worker imports)
- `scripts/research/canonical_models/` — one CLI per stage; single entry `run_all.py`
- `tests/research/canonical_models/` — self-checks for every protocol step
- `artifacts/research/canonical_models/<run_id>/` — manifests, tables, report;
  heavy locals (parquet/models) indexed by hash in `file_index.json`, plus `latest.json`

## Protocol
`protocol.yaml` is hash-pinned. Every result embeds `protocol_hash`
(see `protocol.attach_hash`). Changing the protocol starts a NEW experiment.

Key pins: assets BTC/ETH/SOL/XRP/DOGE · decision T-5m +15s · one decision per
market/policy · outsider ask [0.01,0.40] · $1 fee-inclusive · 6 variants
(market/M0/M1/M2/M3/M4) · pre-fixed periods · challenger = best OOF logloss
among M2-M4 · single pre-registered EV threshold · computed final status.

## Run (local)
```powershell
$env:POLYFLIP_ROLE="local"
python scripts/research/canonical_models/run_all.py --workdir C:\work\flipoly-research --run-id local-smoke-001 --smoke
pytest tests/research/canonical_models -q
```

Resume: each stage writes its outputs; re-running skips completed stages
(`--resume`). Keep threads capped (`compute.threads_cap: 4`), Parquet + chunks
(`compute.chunk_rows: 200000`).

## Gate (step 10)
If canonical labels are insufficient for temporal splitting: mark the study
`PENDING_CANONICAL_DATA`, ship the tested pipeline, and DO NOT re-run the old
proxy experiment under a new name.
