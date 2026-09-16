# Отчёт о выполненной работе — lgbm-feature-audit-v1

**Статус:** завершено (кроме git-push, ожидает путь к репозиторию)
**Спецификация:** `spec/lgbm_feature_audit_v1.yaml` (frozen, sha `ca4dd2fdb7ea6b65e084cb4c59941bcb65741519b385b1fc36e4c5a971147295` v1.0.0)
**Итог:** `STOP_NO_SUPPORTED_FEATURES` — 0/79 фич поддерживаются для продвижения.

Замечания ревью П.Р. («неверный spec-SHA», «canonical-PnL баг», «F2 без проверки timestamp/leakage», «F3 не воспроизводим», «verify_final слаб», «группы vs MINIMAL», «missingness-proxy не считался», «leakage-проверка выдавалась за полную», «F3 статистики считались не на общем маске») — все отработаны: повторный независимый прогон, исправленная экономика, честная SAMPLED-label leakage-проверка, реальная missingness-proxy из FEATURE_MATRIX, F3 на общем маске пары и жёсткие 71 проверки.

---

## 1. Контекст и постановка

Аудит проверяет 79 фич переобученных кандидатов на кохорте решений `shared_ds`
(1 497 818 решений). Вопросы исследования:

1. Какие фичи пригодны для использования (есть поддерживаемый сигнал)?
2. Есть ли индивидуальный значимый вклад отдельных фич?
3. Помогают ли группы фич на OOF?
4. Влияют ли фичи на policy/экономику торговли?

Пайплайн: F2 (качество данных) → F2b (leakage/provenance) → F3 (корреляции) →
F4 (важность: split/gain/SHAP/permutation) → F5 (унивариантный скрининг) →
F6 (групповая абляция) → F7 (policy-impact) → F8 (вердикты + гейты).

## 2. Данные и матрица фич

- **Fold-цепочка:** trainpool (276 912) < F1 (454 476) < F2 (240 293) < F3 (112 110) <
  F4 (114 251) < F5 (106 398) < F6 (94 140); `gap` (99 238) — буфер лейблов, не train-fold.
- **Лейблы есть только со F2** (`contract_target`, fallback `flip_native`); eval-кохорта = F2–F6.
- **Матрица:** 79 фич, as-of через searchsorted, снапшот-фичи ~51% покрытия, lag 29 сек.
  Фолд F1 предшествует появлению снапшотов (покрытие ~0%).
- **Модели:** 37 активных (17 lgbm + 20 logreg) из frozen blobs;
  8 eval-lgbm с deployment-рядами: ids 804, 805, 812, 815, 816, 817, 1019, 1024
  (22 fold-юнита, 34 142 строки).

## 3. Что сделано по шагам

### F1 — фичевая матрица
Построена `FEATURE_MATRIX.parquet` (1 497 818 × 87, из них 79 фич). Кэши
`_candle_features_*.parquet` и `_snapshot_features.parquet` переиспользованы.

### F2 — качество данных → `FEATURE_DATA_QUALITY.csv`
- **5 фич = 100% missing во всех фолдах**: funding_rate, funding_rate_ma3,
  funding_extreme, strike_gap_pct, log_moneyness.
- **26 снапшот-производных** получили жёсткие `BLOCK`/`MISSINGNESS_DRIFT`-вердикты:
  артефакт «train без снапшотов ≈100% missing vs val ≈49%». Итог: **31 фича
  `DATA_QUALITY_BLOCKED`** в F8.
- Кеш-фичи (candle) — OK, без PSI/outlier-аномалий.

### F2b — проверка утечек → `SAMPLED_LEAKAGE_CHECK*` (честная выборочная проверка)
- `f2b_leakage.py`: **60 000 строк детерминированной выборки** (12k/fold F2–F6) из 1 497 818 строк матрицы (4.0% покрытия). 220 строк candle (44 BASE-фичи × 5 активов, пересчитаны из сырых candles.csv.gz), 26 snapshot-фич, 4 time-фичи, 5 структурно всегда-NaN.
- Результат: `any_future_timestamp_row_in_sample = false`, `leakage_risk_block_in_sample = false`; все реальные фичи `value_match ≥ 0.999`, `origin_le_decision = 1.0`. 5 флагов — только always-NaN фичи.
- **Честная маркировка**: проверка — **60k строк, а не полная матрица**. Полноматричная причинность заявлена как свойство алгоритма as-of в `f1_build_matrix` (searchsorted + 29s lag), а не как выполненный построчный scan. Артефакты: `SAMPLED_LEAKAGE_CHECK.csv`, `SAMPLED_LEAKAGE_CHECK_SUMMARY.json`.

### F3 — корреляции → `FEATURE_CORRELATIONS*` (новый воспроизводимый этап)
- 15 405 попарных метрик (pearson+spearman+Jaccard по missing, train_part_each_fold);
  192 near-dup пары в ≥1 фолде, **19 стабильных пар ≥4 фолдов** → 21 фича-участник,
  из них 13 уже блокированы по качеству; 8 неблокированных членов → все `REDUNDANT`
  (cvd_1, day_of_week, dow, hour_of_day, hour_utc, ret_1, signed_body_pct, taker_buy_ratio).

### F4 — важность фич → MODEL_IMPORTANCE/SHAP/PERMUTATION
- **SHAP:** лидер range_1 (weighted_abs 0.0247, знак +), далее range_avg_24,
  dist_to_low_24 (знак-устойчивость 1.0), ema_ratio_9_21, ret_1.
- **Split/gain:** лидеры ret_1, ret_3, range_1, dist_to_low_24, vol_z_1,
  taker_buy_ratio, ema_ratio_9_21, range_avg_24, vol_24, rsi_14, cvd_6, ret_6.
- **Permutation (стратификация day/asset/regime, fold-unit bootstrap):**
  только **cvd_6** и **vol_6** имеют CI ≥ 0. Остальные — статистический ноль.

### F5 — унивариантный скрининг → `UNIVARIATE_OOF.csv`, `F5_VERDICT.json`
- L2-LogReg (C=1, StandardScaler), mapping train_k→validation_k, Holm α=0.05.
- Топ по AUC: mid_price 0.8107, pm_best_ask/bid/quote_pressure ≈0.807,
  price_distance_from_max 0.74.
- **INCONCLUSIVE**: min p_holm = 0.1741 при разрешении 200 пермутаций; ни одна фича
  не выживает Holm (0/79). Лимит задокументирован в `F5_VERDICT.json` и в поле
  gate `f5_resolution`.

### F6 — групповая абляция → GROUP_ABLATION + GROUP_DAYBLOCK_CI
- 20 вариантов (ALL / DROP×9 / ONLY×9 / MINIMAL_CONTROL=[ret_1, mid_price, spread,
  time_left_min]) на fold'ах F3–F6 (F2 пропущен — нет train-лейблов).
- **Day-block paired CI (канонический main_ci):** при удалении любой группы логлосс
  растёт; значимо помогают **cross_asset_breadth** (ΔLL 0.01338, CI [0.00070, 0.03362])
  и **volume_taker_flow** (0.00641, CI [0.00009, 0.01579]).
- regime — наибольший точечный эффект (0.01340), но CI пересекает 0.
- **MINIMAL_VS_ALL:** ΔLL 0.00912, CI пересекает 0, 2/4 фолдов положительны →
  полный набор из 79 фич НЕ надёжно лучше 4-фичевого контроля.

### F7 — policy-impact → `FEATURE_POLICY_IMPACT.csv` (повторный прогон по ревью)
- **Каноническая single-side экономика с бюджетом $1** (исправлен старый select_side):
  `exec_cost(ask) = ask + 0.07·ask·(1−ask) + 0.005·ask`; выбор единственной стороны по
  максимуму положительной edge: `e_yes = p − cost_yes`, `e_no = (1−p) − cost_no`;
  win net = `q−1` (q = 1/cost), loss = −1, skip = 0. Юнит-проверки:
  cost(0.5)=0.52; BUY_NO при y=1 → −1 (регрессионная защита в verify_final).
- **Все DROP-дельта по каноническому net теперь отрицательны** (удаление не помогает
  полной модели); **удаление sequence / time / polymarket_state надёжно УХУДШАЕТ
  экономику** (day-block net CI целиком > 0).
- Предсказания некалиброваны → результат диагностический, не торговая рекомендация.
- `GROUP_TRADING_DAYBLOCK_CI.csv` — per-day `d_net = net_ALL − net_DROP` CI.

### F8 — вердикты → FEATURE_VERDICTS.json(+EVIDENCE.csv), GATE_FEATURE.json
- **SUPPORTED: 0/79. DATA_QUALITY_BLOCKED: 31. REDUNDANT: 8. UNSTABLE: 5.
  NO_POLICY_IMPACT: 20. UNSUPPORTED: 15.**
- UNSTABLE: cvd_6, dist_to_high_24/96, dist_to_low_24/96 — «группа-надёжные, фича-нет».
- Все 8 условий считаются из реальных артефактов; `no_leakage_no_missingness_proxy`
  = все-Y (f2b чистый; missingness-proxy |r|>0.2 ни для одной фичи).
- GATE_FEATURE: `STOP_NO_SUPPORTED_FEATURES`, status **COMPLETED**,
  gate_version **v1.1.0**, spec_sha256 **ca4dd2fd…**.

### Gates и манифест
- RUN_MANIFEST.json обновлён: feature_matrix, f2_quality, f2b_leakage, f3_correlations,
  f4_importance, f5_univariate, f6_ablation, f7_policy_impact, f8_verdicts, verify,
  gates; spec_sha = ca4dd2fd… (c36f0de5… = direction-spec, НЕ привязано).
- **gate_21 = STOP_NO_SUPPORTED_FEATURES (status COMPLETED);
  gate_22 = STOP_NO_SUPPORTED_FEATURES.**

### Верификация → `verify_final.py` (переписан как независимый verifier)
**71 независимая проверка, результат PASS:**
- пересчёт матрицы (1 497 818×79), глобально-missing (5);
- day-block LL CI пересчитан с нуля из `_abl_preds` (10 групп, tol 5e-4);
- канонический net-trading day-block CI пересчитан НЕЗАВИСИМО (своя реализация
  экономики: бюджет $1, fee 0.07·ask·(1−ask), sl 0.005·ask, BUY_NO win/loss-гарант,
  сверка построчно с `canonical_side_net`);
- spec-SHA привязка (= ca4dd2fd…, ≠ c36f0de5…), отсутствие константных gate-флагов;
- SAMPLED leakage: `checked_rows == 60000`, `coverage_fraction == checked/total`,
  per-fold 12k, `full_matrix_provenance` раскрыта (as-of по построению, не scan);
- missingness-proxy из РЕАЛЬНОЙ матрицы: 79 фич × F2–F6 = 395 строк, r конечны,
  gate раскрывает 395; отсутствуют фичи с |r| > 0.2;
- near-dup↔REDUNDANT согласованность (≥19 стабильных пар), robust-perm {cvd_6, vol_6},
  Holm пуст + INCONCLUSIVE, счётчики вердиктов + status-priority, согласованность
  gate-счётчиков, все обязательные входы гейта непустые.

## 4. Ответы на вопросы исследования

| Вопрос | Ответ |
|---|---|
| Какие фичи пригодны? | Ни одной. 31 заблокирована (качество), остальные ниже разрешения теста |
| Есть индивидуальный сигнал? | Статистически нет; максимум — permutation CI>0 у cvd_6, vol_6, и те не проходят полный грид |
| Помогают ли группы? | Да, на уровне групп: cross_asset_breadth и volume_taker_flow значимы (LL); sequence/time/polymarket_state (net) |
| Влияние на policy? | Удаление групп ухудшает канонический net; ни одна группа не «чинится» удалением |

## 5. Выводы и рекомендации

1. **Не продвигать ни одну из 79 фич** для дообучения/форвардинга.
2. Живые сигналы — только на уровне **групп** (cross-asset breadth, taker-поток;
   для торговли sequence/time/polymarket_state), не отдельных фич. Это **гипотезы**,
   их активация требует отдельного исследования.
3. При будущей ре-архитектуре: исключить 5 always-missing фич; снапшот-кохорту
   не использовать до ~10.08; сравнивать через control-vs-ALL, а не single-feature.
4. Чувствительность к единице агрегации (per-fold vs day-block) задокументирована —
   primary остаётся day-block CI.
5. **Порог missingness-proxy |r| > 0.2 — поздняя операционализация** (не зафиксирован
   численно в frozen spec). Максимальный фактический |r_missing_target| ≈ 0.058, что
   значительно ниже порога. Вердикт STOP не зависит от этого условия: ближайшие
   кандидаты проваливают ещё минимум два других условия гейта.
6. **Leakage проверен построчно только на 4% данных** (60 000 строк из 1 497 818).
   Формулировки честно помечены как SAMPLED; нельзя впоследствии сокращать их до
   «утечек во всей матрице нет».
7. Артефакты заморожены; остаётся git-push в `research/lgbm-feature-audit-v1`.

## 5b. Граница решения (что этот аудит НЕ разрешает)

| Результат | Действие |
|---|---|
| 0 отдельных SUPPORTED-фич | Ничего не активировать; forward-кандидат не запускать |
| 31 DATA_QUALITY_BLOCKED | Улучшить данные, затем пересчитать; это не «доказанная бесполезность» |
| 8 REDUNDANT | Не удалять из production без отдельной модели-замены |
| 5 UNSTABLE | Не использовать как отдельные признаки |
| Групповые сигналы (group-level) | Отдельное исследование; текущих данных недостаточно для активации |
| F5 INCONCLUSIVE | Не считать доказательством отсутствия сигнала |

Аудит не разрешает: создавать/активировать новые LGBM на проверенных 79 фичах,
менять действующие модели по результатам этого аудита, запускать forward-кандидата.
Аудит **не запрещает и не деактивирует** уже работающие production-модели, CT,
LogReg и текущие торговые режимы.

## 6. Артефакты (out/feature)

FEATURE_MATRIX.parquet (тяжёлый, вне Git — SHA в манифесте), FEATURE_CATALOG.csv,
FEATURE_DATA_QUALITY.csv,
SAMPLED_LEAKAGE_CHECK.csv, SAMPLED_LEAKAGE_CHECK_SUMMARY.json,
MISSINGNESS_PROXY.csv,
FEATURE_CORRELATIONS(.parquet/_FOLD.parquet/_AGGREGATE.csv),
MODEL_IMPORTANCE(.csv/_DETAIL/_AGGREGATE),
SHAP_STABILITY(.csv/_AGGREGATE), PERMUTATION_IMPORTANCE(.csv/_AGGREGATE),
UNIVARIATE_OOF.csv, F5_VERDICT.json,
GROUP_ABLATION(.csv/_SUMMARY), GROUP_DAYBLOCK_LL.csv, GROUP_DAYBLOCK_CI.csv,
GROUP_TRADING_DAYBLOCK_CI.csv, _abl_preds/ (тяжёлое, вне Git), FEATURE_POLICY_IMPACT.csv,
FEATURE_VERDICTS.json, FEATURE_VERDICTS_EVIDENCE.csv, GATE_FEATURE.json,
CLOSEOUT.md, RUN_MANIFEST.json (обновлён).