# T32 CLOSEOUT — market_relative_v1
status: CLOSED
stage_reached: HISTORICAL_GATE
verdict: STOP
forward_started: false
verdict_semantics: PASS means checks executed correctly; trading outcome is FAIL/STOP.

## Hypothesis
Binance 5m/15m candles and Polymarket top-of-book microstructure contain
information beyond the market probability sufficient for a positive net edge
after taker costs, via a market-relative correction
logit(p_final) = logit(p_market) + delta(features).

## Protocol
- Spec v1.0.4 sha256 2dce310ce3d4917061109d154520b0eb932dc9894b88b787d00fd32b0898071c
  (chain v1.0.0 b86a3532 -> v1.0.1 94febc06 -> v1.0.2 49967e4b -> v1.0.3 1a64cd54).
- Choice D event-driven: decision = first snapshot in [expiry-330, expiry-270];
  forward pairs keyed by (market_id, event_at).
- 6 expanding folds, 1d embargo, 7d validation; max 4 configs (1 linear + 3 LGBM);
  historical gate as code (gate.py, spec-sha-pinned); STOP -> no forward.

## Models
- REGULARIZED_LINEAR_OFFSET (manual IRLS Newton, L2 lambda=10, intercept free,
  standardized, 2000 iter, tol 1e-8, deterministic).
- LIGHTGBM_OFFSET A/B/C (init_score = market logit, lr 0.03, 2000 est, ES 100,
  deterministic, threads 4). Depths 3/4/5, leaves 7/15/15, min_data 200/200/500.
- Baseline MARKET_ONLY (delta = 0).

## Data
- 26628 rows, dataset sha256 14ce26d3 (two independent rebuilds identical).
- Window 2026-07-06..2026-09-12 (excl); 18 features + 18 missing flags, fixed order.
- Binance 5m/15m closed candles only (grid complete; backfilled 5m close_time
  completed as open+299.999ms, exact on 100k+ rows having both).
- No seconds/perp/strike/depth-history (insufficient history, frozen exclusions).
- end3 frozen at export; live-table drift for recent markets documented.

## Raw edge
- Wilson-tail selection totals: linear +714.94, lgbm_A +546.93, B +549.35,
  C +499.66, baseline +519.76. Pattern identical incl. baseline -> NO
  model-added raw edge. All concentrated in F1 (val 27.07-03.08): +556..+622
  per model; F2/F4/F5 negative; F3/F6 near zero.

## Costs
- Canonical 0.07 price-dependent fee per side; historical top-of-book ask +
  0.5% adverse; forward would use depth VWAP (never reached).
- Fee/trade ~3.7c; spread captured via executable book.

## Canonical net edge
- linear +101.31 (ROI +1.58c/trade, n=6421), lgbm_A +50.43, lgbm_B +45.40,
  lgbm_C +0.27, baseline +14.70. Positive only pooled; per-fold mostly negative.

## Statistical stability
- fold_pos_share 0.83-0.96 (one fold holds everything) -> FAIL (<=0.50).
- Day-block bootstrap CI95 lower bounds -0.11..-0.15 -> FAIL (>-0.01).
- maxDD/staked -9.2..-12.3% -> FAIL (<=7.5%).
- Logloss improvement overall: linear -0.0018, lgbm ~-0.015 -> FAIL (>=+0.002).
- Logloss positive folds: linear 1/6, lgbm 0/6 -> FAIL (>=4).
- Canon positive folds: 2/6 each -> FAIL (>=4).
- lgbm ECE 0.063 (mid-range overconfidence) -> FAIL (<=0.03).
- True slopes: baseline 1.02, linear 0.97, lgbm 1.02 (all pass; an it/sl swap
  bug briefly reported them as intercepts - caught by decile check, fixed,
  regression-tested).
- Gate control tests: bad-model FAIL, good-model PASS, order-invariant.

## Bounds of conclusion (exact safe formulations)
- На этой выборке, с этими признаками и проверенными моделями дополнительное
  качество вне обучения не обнаружено. Это граница вывода, а не повод
  повторять подбор (подбор на той же истории запрещён).
- Агрегированные показатели калибровки близки к целевым (slopes ~1, децили
  совпадают). Это не подтверждает калибровку каждого актива, стороны и
  ценового диапазона.
- Устойчивого улучшения по временным блокам нет (logloss лучше market-only
  в 1/6 folds у linear и 0/6 у LGBM; экономия положительна в 2/6).
- Pooled-OOF Wilson-бины рассчитаны с использованием исходов той же
  оцениваемой выборки, поэтому не являются независимой проверкой выбора
  сделок. Для отрицательного решения это не повод перезапускать
  исследование; ограничение зафиксировано здесь.

## Verdict
STOP. No candidate passes (7-8 failed conditions each). Forward not started.
No model registered, activated, or deactivated. No further tuning on this
history. Production untouched:
no migrations ran, no orders, DB read-only, compute local only.

## Allowed next action
No more training in this setup (STOP). Return to model search only with a
concrete new information source or a different testable hypothesis (new
experiment_id, new frozen spec). Branch retained for audit.
A future v2 (seconds flow, perpetual, strike, depth history) needs its own
experiment_id and frozen spec; this closeout does not authorize it.
Dashboard-settings-to-execution propagation is NOT covered by this experiment
and must be verified separately (open item).

## Provenance
- Commits: 0c0d68f4 (T08 dataset freeze), 410c6aae (T09-T18, this closeout follows).
- Local tests: 50 passed (dataset 11, costs 18, models 8, ev 9, gate 4).
- Repo suite at T03 stage-gate: 1718 passed, 6 pre-existing failures + 4
  collection errors (env, proven unrelated by isolation run).
- Artifacts (local, not in git): dataset.csv, oof.csv, signals.csv, eval/ full dir.
- In git: spec+sha, schema, manifest, dataset.sha, code+tests,
  metrics/gate/bins/run_manifest/medians/manual20.
