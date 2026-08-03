# План анализа и исправления PAPER-режима PolyFlip

## 1. Цель и границы работ
**Цель этапа:** сделать PAPER-режим достоверным испытательным контуром, в котором можно проверить качество торгового решения, симуляции исполнения и расчёта результата до дальнейшего использования LIVE.

В этот этап входят:
- получение и проверка рыночных данных;
- выбор LightGBM-модели для текущего режима волатильности;
- прогноз направления базового актива через LightGBM;
- оценка вероятности выигрыша контракта и возможности входа через LogReg;
- вычисление edge по исполнимой цене Polymarket;
- создание одной PAPER-заявки на одно решение;
- реалистичная симуляция покупки;
- разрешение завершившихся рынков;
- расчёт realized PnL;
- однозначная атрибуция результата моделям;
- диагностика причин SKIP, отказов и ошибок.

В этот этап не входят:
- stop loss;
- take profit;
- автоматическое закрытие позиции до завершения рынка;
- настройка логики CLOSE;
- погашение LIVE-токенов;
- изменение LIVE Release Gate;
- увеличение LIVE-бюджета или размера LIVE-заявки.

*Поля и существующий код SL/TP пока не удалять и не рефакторить. Они просто не должны участвовать в новых вычислениях, интерфейсах и тестах этого этапа.*

## 2. Целевая архитектура решения
Старая схема «LogReg принимает решение, LightGBM может только наложить VETO» больше не используется как целевая.
Новая схема **PAPER COMBINED**:
1. LightGBM прогнозирует направление цены базового актива на горизонте рынка: `UP` или `DOWN`.
2. Фазовая LogReg оценивает вероятность события Polymarket с учётом состояния рынка: `contested`, `leaning` или `decided`.
3. Система выбирает сторону контракта только при согласованном и экономически выгодном результате.
4. Решение принимается по абсолютному net edge относительно свежего `best_ask` выбранной стороны.
5. Если обязательный компонент недоступен или данные устарели, результатом становится `SKIP`, а не незаметный fallback к непроверенной логике.

Канонический поток:
```text
Market snapshot
  -> свежая котировка YES/NO
  -> режим волатильности
  -> LightGBM direction probability
  -> фаза рынка
  -> фазовая или базовая LogReg
  -> p_candidate_win
  -> best_ask выбранной стороны
  -> gross_edge и net_edge
  -> PAPER execution request
  -> simulated fill
  -> final_outcome
  -> realized PnL
```

## 3. Сначала зафиксировать исходное состояние
До изменения торговых формул нужен read-only снимок минимум за 30 дней. Он станет контрольной точкой, с которой сравниваются результаты после исправления.

### P0.1. Снимок конфигурации
Сохранить:
- все настройки, влияющие на COMBINED, LogReg, LightGBM, edge и PAPER fill;
- список активных моделей и их версий;
- доступные LightGBM-модели по активам и режимам `low_vol`, `mid_vol`, `high_vol`;
- фазовые LogReg-модели;
- количество рынков, решений, заявок, исполнений и разрешённых сделок.

Настройки во время контрольного PAPER-прогона не менять. Иначе сравнение периодов будет недостоверным.

### P0.2. Проверка задвоения решений
Ввести и использовать единый `decision_run_id`. Для одного цикла оценки одного рынка допустима ровно одна итоговая строка COMBINED.
Проверить исторические данные:
```sql
SELECT decision_run_id, count(*)
FROM decision_funnel_log
WHERE execution_mode = 'PAPER'
  AND created_at >= now() - interval '30 days'
GROUP BY decision_run_id
HAVING count(*) > 1;
```
Если `decision_run_id` отсутствует в старых строках, временно использовать составной диагностический ключ:
```text
market_id + evaluation_window + execution_mode + strategy_config_version
```
Не удалять старые строки. Для истории создать отдельный отчёт о дублях, а исправление применять к новым решениям.

### P0.3. Проверка связей решения и исполнения
Для каждого решения проверить цепочку:
```text
DecisionFunnelLog -> TradeHistory -> ExecutionRequest -> ExecutionAttempt
```
Инварианты:
- у одного прошедшего решения не более одного PAPER `OPEN` request;
- у одного request не более одной активной попытки отправки;
- `SUCCESS/OPEN` невозможен при `filled_shares <= 0`;
- `FILLED` невозможен при `filled_cost_usdc <= 0`;
- `ENTRY_FAILED` не считается позицией и не участвует в PnL;
- `SKIP` не создаёт TradeHistory и ExecutionRequest.

## 4. Исправить доступность и выбор моделей

### P0.4. LightGBM как Direction Model
LightGBM должен возвращать структурированный результат:
```python
DirectionResult(
    status="READY",
    model_key="BTCUSDT_mid_vol",
    model_version=42,
    volatility_regime="mid_vol",
    p_up=0.61,
    p_down=0.39,
    predicted_direction="UP",
    feature_timestamp=feature_timestamp,
)
```
Допустимые неуспешные статусы должны быть явными:
- `MODEL_NOT_LOADED`
- `REGIME_UNAVAILABLE`
- `FEATURES_STALE`
- `FEATURES_INVALID`
- `DEGENERATE_PROBABILITY`
- `INFERENCE_FAILED`

Любой такой статус в PAPER COMBINED приводит к `SKIP`. Недоступность модели не считается VETO и не должна выглядеть как отрицательный прогноз.
Не использовать модель другого режима волатильности. Например, отсутствие `BTCUSDT_high_vol` нельзя компенсировать `BTCUSDT_mid_vol`.

### P0.5. Проверка полноты реестра LightGBM
На странице моделей и в диагностическом API показывать матрицу:
| Актив | low_vol | mid_vol | high_vol | Текущий режим | Статус |
|---|---|---|---|---|---|
| BTC | v... | нет | v... | mid_vol | REGIME_UNAVAILABLE |

Перед запуском PAPER-прогона не требуется обучать все 15 моделей любой ценой. Но статистика должна разделять:
- модель была доступна и дала прогноз;
- нужная режимная модель отсутствовала;
- признаки были непригодны;
- модель упала на inference.

### P0.6. Fallback для Entry LogReg
Для PAPER разрешить только один мягкий fallback:
```text
Phase LogReg -> Base LogReg -> SKIP
```
Глобальный или случайный fallback не использовать.
Каждое решение должно сохранять источник:
- `entry_model_source=PHASE`
- `entry_model_source=BASE_FALLBACK`
- `entry_model_source=UNAVAILABLE`

Результаты `PHASE` и `BASE_FALLBACK` нельзя смешивать в одной строке PnL аналитики. Это разные стратегии.

## 5. Привести вероятности и edge к одной формуле

### P0.7. Не смешивать разные вероятности
Хранить отдельно:
- `direction_p_up` и `direction_p_down` от LightGBM;
- `entry_p_yes` от LogReg;
- `candidate_side`;
- `p_candidate_win`;
- `best_ask` выбранной стороны;
- `gross_edge`;
- `cost_buffer`;
- `net_edge`.

Нельзя записывать `p_flip_effective`, вероятность направления и вероятность выигрыша выбранного контракта в одно поле.

### P0.8. Единое отображение вероятности выбранной стороны
Если LogReg возвращает вероятность YES:
```python
p_candidate_win = p_yes if candidate_side == "YES" else 1.0 - p_yes
```
LightGBM используется для проверки экономической согласованности выбранной стороны с прогнозом направления базового актива. Точная таблица соответствия должна быть покрыта тестами для четырёх случаев:
1. прогноз `UP`, покупка `YES`;
2. прогноз `UP`, покупка `NO`;
3. прогноз `DOWN`, покупка `YES`;
4. прогноз `DOWN`, покупка `NO`.

### P0.9. Edge считать по исполнимой цене
Для PAPER-входа использовать свежий `best_ask`, а не mid, last trade или цену старого snapshot.
```python
gross_edge = p_candidate_win - best_ask
net_edge = gross_edge - cost_buffer
```
Где `cost_buffer` включает только реально моделируемые затраты PAPER-входа:
```python
cost_buffer = slippage_buffer + fee_buffer
```
Порог входа применять к `net_edge`:
```python
if direction_confidence < min_direction_prob:
    return SKIP("DIRECTION_CONFIDENCE_TOO_LOW")

if p_candidate_win < min_win_prob:
    return SKIP("WIN_PROBABILITY_TOO_LOW")

if net_edge < min_net_edge:
    return SKIP("NET_EDGE_TOO_LOW")
```
`expected_roi = net_edge / best_ask` можно показывать в аналитике, но не использовать вместо абсолютного edge без отдельного решения.

## 6. Сделать PAPER-исполнение реалистичным

### P0.10. Отделить решение от симуляции исполнения
Decision engine отвечает на вопрос «нужно ли войти». PAPER gateway отвечает на вопрос «могла ли заявка исполниться по доступной цене».
Decision engine не должен заранее выставлять `SUCCESS`, `OPEN` или итоговый PnL.

### P0.11. Использовать top-of-book для PAPER BUY
Минимальная первая версия симулятора:
1. Повторно получить свежий `best_ask` перед fill.
2. Проверить возраст котировки.
3. Проверить допустимое изменение цены относительно решения.
4. Рассчитать shares по фактической simulated fill price.
5. Записать fill только после успешной проверки.
```python
fill_price = fresh_best_ask
filled_shares = spend_usdc / fill_price
filled_cost_usdc = filled_shares * fill_price
```
Если котировки нет или она устарела:
```text
request.state = REJECTED
reason_code = PAPER_QUOTE_UNAVAILABLE
```
Нельзя создавать нулевой fill со статусом успеха.

### P0.12. Моделировать биржевые ограничения
PAPER должен применять те же базовые ограничения, которые уже обнаружились в LIVE:
- минимальная сумма заявки;
- цена в допустимом диапазоне;
- достаточная точность amount/shares;
- отсутствие устаревшей котировки;
- ограничение price drift;
- при наличии объёма top-of-book сумма не должна превышать доступную ликвидность.

Если depth недоступен, это фиксируется отдельным полем `liquidity_check_status=NOT_AVAILABLE`. Не нужно придумывать полное исполнение на неограниченном объёме.
Для текущей конфигурации размер PAPER-заявки можно оставить фиксированным, но фактический размер и источник размера нужно сохранять:
- `requested_spend_usdc`;
- `simulated_fill_cost_usdc`;
- `sizing_source=FIXED|AUTO`.

### P0.13. Состояния PAPER-заявки
Использовать однозначный переход:
```text
READY -> CLAIMED -> SUBMITTING -> FILLED
                           \-> REJECTED
                           \-> RECONCILING
```
Для локального PAPER gateway `RECONCILING` обычно не нужен. Если результат симуляции неизвестен, это ошибка кода или данных, а не неизвестное состояние внешней биржи.
TradeHistory получает `status=SUCCESS` и `position_status=OPEN` только при `filled_shares > 0` и `filled_cost_usdc > 0`.

## 7. Исправить завершение PAPER-сделок и PnL

### P0.14. Условия расчёта realized PnL
Сделку можно разрешить только если одновременно выполнено:
- рынок завершён;
- `final_outcome` непустой и нормализован;
- есть фактический PAPER fill;
- результат ещё не был применён;
- связанный request относится к PAPER.

Если `final_outcome` пустой, сделку не считать ни выигрышной, ни проигрышной. Оставить её в состоянии `AWAITING_OUTCOME` и вывести в отдельный диагностический список.

### P0.15. Каноническая формула PnL
Для покупки контракта:
```python
won = normalized_final_outcome == normalized_bought_outcome
gross_payout_usdc = filled_shares if won else 0
realized_pnl_usdc = (
    gross_payout_usdc
    - filled_cost_usdc
    - simulated_fees_usdc
)
```
Не использовать для PnL:
- requested amount вместо fill cost;
- исходный размер сигнала;
- текущую цену после завершения рынка;
- строку decision log как самостоятельную сделку.

### P0.16. Идемпотентность resolver
Добавить уникальную защиту или атомарную проверку, чтобы один рынок не начислял PnL повторно.
Минимальные поля аудита:
- `resolved_at`;
- `resolution_source`;
- `resolution_version`;
- `final_outcome_raw`;
- `final_outcome_normalized`;
- `pnl_calculation_version`.

Повторный запуск resolver с той же версией не должен менять баланс и PnL.

### P0.17. Проверить исторический PnL без разрушительной перезаписи
Сначала построить read-only сверку:
```text
stored_pnl vs recomputed_pnl
```
Исправление истории выполнять отдельным идемпотентным скриптом только после отчёта:
- число несовпадений;
- суммарное расхождение;
- сделки с пустым final_outcome;
- сделки с zero fill;
- повторно разрешённые сделки;
- сделки без связанного request.

Старое значение PnL сохранять в audit payload. Не выполнять массовый `UPDATE` без возможности восстановить исходное значение.

## 8. Устранить задвоение аналитики моделей

### P0.18. Одна строка decision funnel на цикл
`decide_combined_mode()` не должен вызывать публичный `decide_ml_mode()`, который сам пишет строку в `decision_funnel_log`.
Нужно выделить чистую функцию LogReg без побочных эффектов:
```python
entry_result = evaluate_logreg_entry(...)
combined_result = evaluate_combined_entry(entry_result, direction_result, quote)
await log_combined_decision(combined_result)
```
В COMBINED сохраняется одна строка с обеими ролями моделей.

### P0.19. Атрибуция PnL паре моделей
Для сделки хранить:
- `entry_model_key` и `entry_model_version`;
- `entry_model_source`;
- `direction_model_key` и `direction_model_version`;
- `volatility_regime`;
- `decision_run_id`;
- `strategy_config_version`.

Каноническая единица аналитики:
```text
Entry model version + Direction model version + regime + config version
```
Один realized PnL можно показывать в разрезе роли LogReg и роли LightGBM, но нельзя складывать эти два представления в общий PnL. Общий PnL всегда считается по уникальным `trade_history.id`.

### P0.20. Защита SQL от умножения строк
Агрегаты строить сначала по уникальным сделкам, затем присоединять метаданные моделей.
```sql
WITH unique_trades AS (
    SELECT DISTINCT ON (th.id)
        th.id,
        th.realized_pnl_usdc,
        th.decision_run_id
    FROM trade_history th
    WHERE th.execution_mode = 'PAPER'
      AND th.resolved_at IS NOT NULL
)
SELECT sum(realized_pnl_usdc)
FROM unique_trades;
```
Сумма общего PAPER PnL на странице моделей, странице торговли и в контрольном SQL должна совпадать с точностью до одного цента.

## 9. Диагностика и интерфейс PAPER

### P1.1. Воронка решений
Показывать за выбранный период:
1. рынков оценено;
2. свежая котировка получена;
3. режим LightGBM найден;
4. прогноз направления готов;
5. Entry LogReg готова;
6. порог вероятности пройден;
7. net edge пройден;
8. PAPER request создан;
9. simulated fill получен;
10. рынок разрешён;
11. PnL рассчитан.

Для каждого перехода показывать число и процент потерь.

### P1.2. Причины SKIP
Использовать стабильные reason codes вместо разбора текста:
- `QUOTE_UNAVAILABLE`
- `QUOTE_STALE`
- `DIRECTION_MODEL_NOT_LOADED`
- `DIRECTION_REGIME_UNAVAILABLE`
- `DIRECTION_FEATURES_INVALID`
- `DIRECTION_CONFIDENCE_TOO_LOW`
- `ENTRY_PHASE_MODEL_UNAVAILABLE`
- `ENTRY_MODEL_UNAVAILABLE`
- `WIN_PROBABILITY_TOO_LOW`
- `NET_EDGE_TOO_LOW`
- `PAPER_FILL_REJECTED`
- `FINAL_OUTCOME_MISSING`

Текст на русском формировать в UI по коду причины. В базе сохранять код и технические детали отдельно.

### P1.3. Фильтры аналитики
Добавить:
- диапазон дат;
- актив;
- фаза рынка;
- режим волатильности;
- Entry model source;
- версия LogReg;
- версия LightGBM;
- `TRADE/SKIP`;
- причина SKIP;
- `FILLED/REJECTED`;
- resolved/unresolved.

### P1.4. Основные метрики
Для каждой пары моделей показывать:
- evaluations;
- trades;
- fill rate;
- resolved trades;
- win rate;
- gross PnL;
- simulated fees;
- net PnL;
- average net edge at decision;
- average realized return;
- Brier score и ECE отдельно для Direction и Entry;
- долю `BASE_FALLBACK`;
- долю отсутствующих режимных LightGBM;
- максимальную серию проигрышей.

Max Drawdown и Profit Factor считать по уникальным разрешённым сделкам. До устранения задвоений эти метрики не использовать для выбора моделей.

## 10. Минимальный набор тестов
Чтобы не тратить ресурсы на полный прогон после каждого небольшого изменения, сначала запускать только целевые тесты.
Обязательные тесты:
- Четыре варианта соответствия направления, стороны и `p_candidate_win`.
- Отсутствующий LightGBM regime приводит к `SKIP`, а не к другой режимной модели.
- PAPER использует Phase LogReg, затем Base fallback, затем `SKIP`.
- COMBINED создаёт одну строку decision funnel.
- PAPER fill использует свежий `best_ask`.
- Нулевая или отсутствующая котировка не создаёт успешную позицию.
- Выигрышная и проигрышная сделка дают правильный PnL.
- Пустой `final_outcome` не меняет PnL.
- Повторный resolver не начисляет PnL второй раз.
- SQL/API аналитики не удваивает одну сделку из-за двух моделей.

Рекомендуемый порядок проверки:
```bash
pytest -q \
  tests/test_combined_entry.py \
  tests/test_paper_execution.py \
  tests/test_paper_resolution.py \
  tests/test_model_attribution.py
```
Полный набор тестов запускать один раз перед слиянием, а не после каждой атомарной правки.

## 11. Порядок внедрения

**Этап A. Только наблюдаемость**
- Добавить `decision_run_id`, model attribution, reason codes и версии расчётов.
- Не менять решения и PnL.
- Собрать 24 часа новых данных.
- Проверить, что одна оценка создаёт одну строку.

**Этап B. Исправление COMBINED**
- Вынести LogReg evaluation в чистую функцию.
- Подключить DirectionResult LightGBM.
- Ввести точные статусы недоступности.
- Запретить fallback между режимами LightGBM.
- Оставить в PAPER только Phase -> Base fallback для LogReg.
- Перевести решение на единую формулу net edge.

**Этап C. Исправление PAPER fill**
- Перевести симуляцию на свежий top-of-book ask.
- Добавить quote age и price drift.
- Запретить zero-fill success.
- Сохранять requested и фактические параметры исполнения.

**Этап D. Исправление resolver и PnL**
- Добавить нормализацию final outcome.
- Не разрешать сделку при пустом результате.
- Пересчитывать PnL только по fill.
- Добавить идемпотентность.
- Построить отчёт расхождений истории.
- После ручного просмотра выполнить контролируемый backfill.

**Этап E. Аналитика**
- Сделать воронку.
- Добавить фильтры и reason codes.
- Перевести PnL на уникальные trade IDs.
- Добавить аналитику пар моделей.
- Только после этого сравнивать модели и менять пороги.

## 12. Критерии готовности PAPER-режима
PAPER-контур можно считать пригодным для анализа, если за контрольный период выполнено всё:
- 0 повторных итоговых COMBINED-строк на `decision_run_id`;
- 0 `SKIP`, создавших execution request;
- 0 успешных позиций с нулевым fill;
- 0 realized PnL при пустом `final_outcome`;
- 0 повторных начислений PnL;
- 100% сделок имеют версии обеих фактически использованных моделей;
- 100% сделок имеют decision quote и simulated fill quote;
- общий PAPER PnL совпадает во всех API и контрольном SQL;
- недоступная Direction Model всегда приводит к понятному `SKIP`;
- результаты Phase LogReg и Base fallback разделены;
- технические ошибки не маскируются как прогноз модели или VETO.

После этого нужен непрерывный PAPER-прогон не менее 72 часов без изменения настроек. Оптимизировать пороги и выбирать модели следует только после накопления достаточного числа разрешённых сделок. Минимально разумный ориентир для предварительного сравнения: 30 разрешённых сделок на пару моделей. Для уверенного вывода потребуется больше данных.

## 13. Что делать после исправления PAPER
Только после прохождения критериев выше:
- сравнить пары Direction LightGBM + Entry LogReg;
- отключить явно убыточные или плохо калиброванные пары;
- настроить пороги `min_direction_prob`, `min_win_prob`, `min_net_edge` по out-of-sample данным;
- проверить стабильность по активам, фазам и волатильностным режимам;
- отдельно подготовить план переноса проверенной логики в LIVE.

*SL/TP на этом этапе не анализировать, не включать и не изменять.*
