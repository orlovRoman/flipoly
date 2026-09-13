# Offline LightGBM threshold review: ETH and SOL

Generated from a read-only database export. Threshold calculations ran on the local workstation; no runtime setting or model registry row was changed.

- Generated at (UTC): `2026-09-13T17:41:25.591802+00:00`
- Selected target coverage: `40.0`
- Symbols: `ETHUSDT`, `SOLUSDT`
- Models evaluated: `6`; errors: `0`
- JSON SHA-256: `4c2519284e8ecc86dd6add645376b33da7c80c7c54f195df67441e95f9471611`

## Selected 40% coverage candidates

| Model | Active version | Branch | Lower | Upper | OOF | signals | trades | net PnL | ROI |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| ETHUSDT_high_vol | 33 | OUTSIDER_ONLY | 0.4723 | 0.5572 | 790 | 312 (39.5%) | 29 | -0.6412 | -2.2% |
| ETHUSDT_low_vol | 34 | FAVORITE_ONLY | 0.4611 | 0.5419 | 790 | 313 (39.6%) | 5 | -3.0788 | -61.6% |
| ETHUSDT_mid_vol | 28 | OUTSIDER_ONLY | 0.4719 | 0.5361 | 815 | 349 (42.8%) | 22 | +9.5103 | +43.2% |
| SOLUSDT_high_vol | 26 | FAVORITE_ONLY | 0.4732 | 0.5181 | 770 | 303 (39.4%) | 29 | +1.8486 | +6.4% |
| SOLUSDT_low_vol | 27 | OUTSIDER_ONLY | 0.4629 | 0.5083 | 770 | 451 (58.6%) | 47 | +28.8237 | +61.3% |
| SOLUSDT_mid_vol | 26 | FAVORITE_ONLY | 0.4886 | 0.5752 | 790 | 317 (40.1%) | 5 | +0.6647 | +13.3% |

The selected candidate is an offline sensitivity result at the frozen 40% target. It is not an activation recommendation. The sweep is evaluated on historical OOF/quote artifacts and is therefore subject to selection bias, especially where the trade count is small.

## Best sweep rows (diagnostic only)

| Model | Target | Branch | Lower | Upper | Trades | net PnL | ROI |
|---|---:|---|---:|---:|---:|---:|---:|
| ETHUSDT_high_vol | 20% | FAVORITE_ONLY | 0.4847 | 0.6973 | 3 | +2.6913 | +89.7% |
| ETHUSDT_low_vol | 60% | COMBINED | 0.4661 | 0.4862 | 207 | +15.3973 | +7.4% |
| ETHUSDT_mid_vol | 20% | OUTSIDER_ONLY | 0.3337 | 0.5912 | 1 | +3.1603 | +316.0% |
| SOLUSDT_high_vol | 20% | COMBINED | 0.4642 | 0.6496 | 3 | +1.2585 | +42.0% |
| SOLUSDT_low_vol | 60% | OUTSIDER_ONLY | 0.4629 | 0.5083 | 47 | +28.8237 | +61.3% |
| SOLUSDT_mid_vol | 20% | FAVORITE_ONLY | 0.4725 | 0.6174 | 1 | +0.9980 | +99.8% |

## Self-checks and decision

- `read_only=true`; the audit uses SELECTs and writes only the local JSON/report artifact.
- `errors=0`; all six latest ETH/SOL regime models had an OOF artifact and completed evaluation.
- Thresholds were not written to runtime settings, model registry, or the active model cache.
- The audit script now filters unsupported `policy_mode` metadata before calling the threshold evaluator; this fixes the previous pre-calculation failure without changing the evaluator.
- Do not lower live ETH/SOL thresholds from this report. Any candidate needs a new forward period and the same frozen gate used for activation.

Conclusion: the offline sweep shows heterogeneous, unstable historical candidates (small trade counts for most regimes). It is useful for prioritizing a future holdout, not evidence that ETH or SOL should be changed in production.
