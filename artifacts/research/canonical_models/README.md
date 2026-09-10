# artifacts/research/canonical_models

Per-run dirs `<run_id>/` hold `manifest.json`, `train_manifest.json`,
`report.json`, tables (parquet/csv) and `file_index.json` (SHA-256 + byte
sizes + storage paths for heavy locals). `latest.json` (Windows) points to
the newest run. Heavy binaries stay local; git keeps manifests + report.
