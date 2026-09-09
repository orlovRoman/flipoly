# Отчёт об устранении дефектов аудита c0e97ff и повторной оценке ценового фильтра

## 1. Резюме исправлений аудита

На основании замечаний к коммиту `c0e97ff` устранены 7 критических дефектов каузальности, статистической сопоставимости и приёмочного контура:

| № | Уровень | Проблема из аудита | Что исправлено в коде | Статус / Тест |
|---|---|---|---|---|
| **1** | **P0** | `label_available_at` в `temporal_validation.py` принимался, но не фильтровал `train_index` | В `grouped_walk_forward_folds` добавлена строгая каузальная фильтрация: наблюдения, у которых `label_available_at > validation_start`, полностью исключаются из обучающего фолда (`train_index`). Если на момент сплита нет доступных меток — возвращается 0 предсказаний со статусом `NO_LABELS_AVAILABLE`. | `tests/trading/test_meta_leakage.py::test_meta_model_label_availability_prevents_future_leakage` — **PASS** |
| **2** | **P0** | Отсутствие проверки контракта происхождения (provenance) базовых прогнозов | В `polyflip/trading/combined_voting.py` внедрён строгий provenance contract: отклоняются базовые прогнозы с `is_out_of_sample=False`, `is_in_sample=True`, `prediction_type="IN_SAMPLE"`, а также с `training_cutoff_at > decision_at` (ошибка `FUTURE_PREDICTIONS_REJECTED`). | `tests/trading/test_meta_leakage.py::test_meta_model_rejects_in_sample_and_future_predictions` — **PASS** |
| **3** | **P1** | Синтетические даты `pd.date_range` и скрытый fallback-сплит | Из `build_meta_model_dataset` полностью удалена генерация дат из воздуха (`pd.date_range`) и скрытая сортировка по порядку групп. При отсутствии `decision_at` или `market_id` пайплайн немедленно возвращает ошибку `MISSING_TEMPORAL_FIELDS`. | `tests/trading/test_meta_leakage.py::test_meta_model_dataset_requires_explicit_temporal_fields` — **PASS** |
| **4** | **P1** | Подавление исключений `except Exception: pass` в обучении meta-модели | Удалены пустые блоки `except Exception: pass`. Ошибки обучения теперь логируются структурированно, а статусы фолдов явно фиксируют причину (`INSUFFICIENT_CLASSES`, `TRAINING_FAILED`, `NO_LABELS_AVAILABLE`). Окончательная модель обучается только на доступных метках (`label_available_at <= max(decision_at)`). | `polyflip/trading/combined_voting.py`, `scripts/research/verify_stage2.py` — **PASS** |
| **5** | **P1** | Несопоставимые выборки при оценке вклада ML поверх наивного фильтра | В `run_stage2_price_filter_evaluation.py` добавлено строгое парное сопоставление на общей OOF-когорте: на полной выборке Dev наивное правило ($\le 0.40$) совершает 177 сделок (+3.73 USDC), но **на той же когорте, где доступны OOF-прогнозы Model A1 (158 строк), наивное правило даёт 158 сделок и +8.16 USDC**. Model A1 ($\le 0.40$) даёт 82 сделки и **+6.33 USDC**. Вклад ML на сопоставимой когорте отрицателен: **-1.83 USDC**. На test holdout вклад ML равен ровно **0.00 USDC** (93 сделки, +5.08 USDC у обеих стратегий). | `scripts/research/run_stage2_price_filter_evaluation.py`, `artifacts/research/stage2_price_filter_verdict.json` — **PASS** |
| **6** | **P1** | 4-кратное расхождение единиц PnL в `evaluate_lgbm_outsider_interaction` | Формула PnL во всех ветках взаимодействия (`B_ONLY`, `VETO`, `INPUT`) унифицирована со спецификацией `OutsiderReplayEngine` на фиксированную ставку 1 USDC: `shares = 1.0 / ask`, `pnl = shares * (outcome - ask) - fee`, где `fee = fee_rate`. | `polyflip/trading/combined_voting.py`, `scripts/research/verify_stage2.py` (Item 2.9) — **PASS** |
| **7** | **P1** | Фиктивные имена колонок в тестах утечки и формальные проверки `verify_stage2.py` | В `tests/trading/test_meta_leakage.py` синтетические колонки `time`/`market` заменены на боевые `decision_at`/`market_id`. В `verify_stage2.py` тест 2.9 дополнен проверкой темпоральной инвариантности (изменение будущих меток не меняет прошлые OOF-прогнозы), а тест 2.10 верифицирует календарную суточную кластеризацию блочного бутстрапа. | `verify_stage2.py` (10/10 PASS), `test_meta_leakage.py` (4/4 PASS) — **PASS** |

---

## 2. Результаты сопоставимого сравнения стратегий

### 2.1. Development выборка (1220 наблюдений)
* **Model A1 Base (без фильтра цен)**: 198 сделок, PnL = **-9.70 USDC**, Expectancy = -0.0490, Max DD = 26.30 USDC.
* **Model A1 с фильтром ($\le 0.40$)**: 82 сделки, PnL = **+6.33 USDC**, Expectancy = +0.0772, Max DD = 10.02 USDC.
* **Naive Price Rule ($\le 0.40$, полный Dev)**: 177 сделок, PnL = **+3.73 USDC**, Expectancy = +0.0211, Max DD = 18.99 USDC.
* **Naive Price Rule ($\le 0.40$, сопоставимая OOF-когорта Model A1)**: 158 сделок, PnL = **+8.16 USDC**, Expectancy = +0.0516, Max DD = 13.87 USDC.

> [!IMPORTANT]
> **Честный вывод о роли ML**:
> На сопоставимой когорте Model A1 отсекла 76 сделок, но суммарный PnL снизился с +8.16 до +6.33 USDC ($\Delta PnL_{ML} = -1.83$ USDC).
> Следовательно, положительный финансовый результат обусловлен исключительно **ценовым порогом входа ($\le 0.40$)**, а не отбором со стороны ML-модели.

### 2.2. Out-of-Sample Test Holdout (524 наблюдения)
* **Base Model A1**: 322 сделки, PnL = **-8.75 USDC**, Expectancy = -0.0272 `[-0.102, +0.041]`, Max DD = 25.69 USDC.
* **Filtered Model A1 ($\le 0.40$)**: 93 сделки, PnL = **+5.08 USDC**, Expectancy = +0.0546 `[-0.223, +0.269]`, Max DD = 13.59 USDC.
* **Naive Price Rule ($\le 0.40$)**: 93 сделки, PnL = **+5.08 USDC**, Expectancy = +0.0546 `[-0.223, +0.269]`, Max DD = 13.59 USDC.
* **Парная дельта (Cap 0.40 vs Base)**: $\Delta PnL = +13.82$ USDC, 95% CI = `[-5.677, +31.266]`.
* **Парная дельта (Cap 0.40 vs Naive)**: $\Delta PnL = 0.00$ USDC, 95% CI = `[+0.000, +0.000]`.

---

## 3. Статистический вердикт и эпистемологический статус

1. **Вердикт**: **`INCONCLUSIVE`** (Не подтверждено статистически).
   - Точечный PnL положителен (+5.08 USDC), однако 95% блочный суточный доверительный интервал пересекает ноль (`[-5.677, +31.266]`).
   - На тестовом отрезке фильтрованная модель и наивное ценовое правило совершают абсолютно идентичные 93 сделки. Модель A1 не даёт никакого статистического преимущества перед простым фильтром `ask <= 0.40`.
2. **Эпистемологический статус**:
   - Данный пересчёт на исторических данных BTC является **исследовательским анализом (exploratory)**, а не «независимым подтверждением».
   - Статус независимой валидации может быть присвоен только после накопления новых рынков через `polyflip_spot_collector` в режиме Paper Trading.

---

## 4. Верификационные прогоны
- `pytest tests/trading/test_meta_leakage.py tests/models/test_outsider_repairs_regression.py` -> **29/29 PASSED**.
- `scripts/research/verify_stage2.py` -> **10/10 PASSED**.
- `scripts/research/verify_model_repairs.py` -> **34/34 PASSED**.
- `scripts/research/run_stage2_price_filter_evaluation.py` -> **Успешно пересчитан и зафиксирован**.
