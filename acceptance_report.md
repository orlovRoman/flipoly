# 📋 Приёмочный отчёт: Реализация замечаний приёмки CT Outsider PAPER

**Ветка:** `feature/paper-ct-outsider`  
**Базовый проверенный коммит:** `b843b7b`  
**Дата приёмки:** 2026-09-09  
**Статус:** ✅ **ВСЕ 24 ПУНКТА ПРИЁМКИ ВЫПОЛНЕНЫ И ПРОТЕСТИРОВАНЫ БЕЗ ТЕСТОВЫХ ОБХОДОВ**

---

## Этап 1. Исходное состояние и анализ (Пункты 1–4)

### 1. Проверяемая версия и рабочее дерево
- **Коммит:** `b843b7b` (`docs: update acceptance report with merge and test counts`).
- **Ветка:** `feature/paper-ct-outsider`.
- **Рабочее дерево:** Рабочая ветка `feature/paper-ct-outsider`. Все 5 замечаний (P1 сквозной цикл, P1 контракт бюджета без подгонки, P1 конкурентность двух сессий, P1 откат аварийного состояния, P2 фактическое decision_at в trade_recorder) полностью закрыты.

### 2. Сопоставление замечаний приёмки с кодовой базой
1. **Время решения и котировки:** Ранее в `SideQuote` отсутствовало строгое разделение `event_at` и `received_at`; при отсутствии таймстемпов котировка не отклонялась, а окно до экспирации могло использовать `cycle_started_at` вместо точного `decision_at`.
2. **Исполнение:** В синтетических тестах использовалась исследовательская функция `simulate_orderbook_execution` вместо реального пайплайна outbox/worker.
3. **Повторные заявки и идемпотентность:** Отсутствовала таблица атомарных броней решений на уровне базы данных, что создавало риск повторного входа в один и тот же маркет при конкурентных циклах или после settlement.
4. **Диагностика моделей:** Поле `non_blocking_models` могло интерпретироваться как факт запуска предсказания, хотя модели для CT не вызываются (только проверяется доступность компонентов).

### 3. План исправлений
- Создание модели `CTDecisionReservation` с первичным ключом `key = f"CT:{spec_id}:{market_id}"`.
- Сервис `reserve_ct_decision` с атомарным `INSERT ON CONFLICT DO NOTHING` для PostgreSQL и SQLite.
- Разделение `cycle_started_at` и `eff_decision_at` в `decide_ct_outsider_mode`.
- Валидация возраста котировок (`event_at` приоритетнее `received_at`), отказ по устареванию (>15 с -> `STALE_QUOTE`), строгий запрет котировок из будущего (`FUTURE_QUOTE_DETECTED`), отказ при отсутствии таймстемпов (`MISSING_QUOTE_TIMESTAMP`).
- Перевод тестов на полный реальный пайплайн: `pre_trade_validator` → `execute_and_record` → `enqueue_open_request` → `claim_one` / `FakeExecutionGateway` (профиль `LIVE_PARITY`) → `_persist_fills` → `rebuild_trade_accounting` → `settle_resolved_position`.
- Добавление тестов конкурентности, восстановления после сбоев в 3 точках и сквозной трассировки ID по БД.

### 4. Неизменность порогов CT и торговой гипотезы
- Спецификация `BTC_CT_T5_V1` осталась полностью неизменной:
  - `min_outsider_spread = 0.05`
  - `max_outsider_ask = 0.40`
  - `decision_window_min_sec = 210.0`
  - `decision_window_max_sec = 300.0`
  - `budget_usdc = 1.00`
  - `taker_fee_rate = 0.002`
- Хеш спецификации `spec_hash` детерминирован и неизменен.

---

## Этап 2. Время решения, котировки и причинность (Пункты 5–8)

### 5. Разделение меток времени и вычисление возраста
- `cycle_started_at` фиксирует старт цикла воркера.
- `eff_decision_at` фиксируется строго единожды после получения актуальных котировок и до запуска классификатора режима/политики.
- Возраст котировки рассчитывается в `SideQuote.compute_age_sec(decision_at)`:
  - Используется `event_at` (время события на бирже); если отсутствует — используется `received_at` (время получения вебсокетом/REST).
  - Если обе метки отсутствуют — котировка не получает нулевой возраст, а валидация возвращает `(False, "MISSING_QUOTE_TIMESTAMP")`.

### 6. Контроль устаревания (> 15 секунд)
- В `SideQuote.is_valid_for_decision`:
  - Если `age_sec > max_staleness_sec` (по умолчанию `15.0` с), возвращается `(False, "STALE_QUOTE")`.
  - Решение переводится в `SKIP` с причиной `QUOTE_ERROR: STALE_QUOTE`.

### 7. Запрет заглядывания в будущее
- Если `event_at > decision_at` или `received_at > decision_at` (или `age_sec < 0.0`), возвращается ошибка с префиксом `FUTURE_QUOTE_DETECTED` (`EVENT_IN_FUTURE` / `RECEIVED_IN_FUTURE` / `NEGATIVE_AGE`). Отрицательный возраст категорически не обнуляется.

### 8. Расчет окна до экспирации
- Окно вычисляется строго как `(expiration - decision_at).total_seconds()`.
- Использование `cycle_started_at` для расчета окна устранено.
- В метаданные решения `timing_diagnostics` записываются `cycle_started_at`, `decision_at`, `time_left_sec`, таймстемпы и возраст обеих котировок.

---

## Этап 3. Тесты на котировки и причинность (Пункт 9)

В `tests/trading/test_ct_policy.py` добавлены и успешно пройдены тесты:
- `test_quote_timing_fresh`: свежая котировка (5.0 с) принимается как валидная.
- `test_quote_timing_stale_rejection`: котировка старше 15 с (16.0 с) отклоняется со `STALE_QUOTE`.
- `test_quote_timing_exact_boundary`: точная граница 15.000 с принимается, а 15.001 с отклоняется со `STALE_QUOTE`.
- `test_quote_timing_missing_timestamp`: котировка без таймстемпов отклоняется с `MISSING_QUOTE_TIMESTAMP`.
- `test_quote_timing_negative_age_future`: котировки из будущего отклоняются с `FUTURE_QUOTE_DETECTED`.
- `test_quote_timing_received_vs_event_precedence`: подтвержден приоритет `event_at` над `received_at` при расчете возраста.

---

## Этап 4. Реальное PAPER-исполнение (Пункты 10–15)

Исследовательская функция `simulate_orderbook_execution` полностью заменена в `tests/trading/test_ct_synthetic_cycle.py` на реальный пайплайн:
`pre_trade_validator` → `execute_and_record` → `enqueue_open_request` → `claim_one` → `FakeExecutionGateway` (`LIVE_PARITY`) → `_persist_fills` → `rebuild_trade_accounting` → `settle_resolved_position`.

### 11. Полное исполнение (Full Fill)
- Тесты: `test_01_synthetic_up_outsider_buy_and_full_fill_win`, `test_02_synthetic_down_outsider_buy_and_full_fill_win`.
- При лимите 0.20 и бюджете 1.00 USDC с комиссией 0.002 исполняется ровно:
  `1.00 / (0.20 * 1.002) = 4.99002` shares (без искусственного завышения бюджета на 1%).
- `spent_usdc` = 1.00000 USDC (0.998004 gross + 0.001996 fee).
- `TradeHistory`: `entry_filled_shares = 4.99002`, `position_status = "OPEN"`.
- При выигрыше YES: `settle_resolved_position` закрывает сделку (`position_status = "CLOSED"`), `realized_pnl_usdc = +3.99002` USDC.
- Для DOWN outsider BUY: ask 0.25, комиссия 0.002 -> `1.00 / (0.25 * 1.002) = 3.99202` shares, чистый PnL = +2.99202 USDC.

### 12. Частичное исполнение (Partial Fill)
- Тест: `test_07_synthetic_partial_fill_liquidity_preserves_unspent_budget`.
- В стакане доступно только 10 shares @ 0.05 ($0.50 глубина) при бюджете $1.00.
- Исполняется 10 shares @ 0.05, затрачено $0.50 + комиссия 0.001 USDC = 0.501 USDC.
- Неиспользованный остаток бюджета $0.499 сохраняется на балансе и **не списывается в убыток**.

### 13. Изменение лимитной цены (Limit Price Change)
- Тест: `test_08_synthetic_limit_price_change_rejection`.
- Решение принято по лимиту 0.20. В момент исполнения лучший ask ушел на 0.21.
- Шлюз `FakeExecutionGateway` отклоняет заявку (`NO_LIQUIDITY_FAK`), 0 fills.
- Запрос финализируется со статусом `REJECTED`, сделка переводится в `ENTRY_FAILED`, баланс не расходуется.

### 14. Многоуровневый стакан и VWAP
- Тест: `test_09_synthetic_multi_level_orderbook_vwap`.
- Стакан: Уровень 1 (5 shares @ 0.10, $0.50), Уровень 2 (10 shares @ 0.12).
- Реальный шлюз без искусственных оверрайдов бюджета или количества акций (`shares_override` убран): заявка с лимитом 0.12 запрашивает `1.00 / 0.12 = 8.333333` shares, выкупает 5.0 shares @ 0.10 ($0.50) и 3.333333 shares @ 0.12 ($0.40).
- Всего куплено: 8.333333 shares за 0.90 USDC.
- Фактическая цена входа (`executed_price` в `TradeHistory`) фиксирует точный VWAP = 0.90 / 8.333333 = **0.108000** USDC/share.

### 15. Расчет settlement при частичном исполнении
- Тест: `test_10_synthetic_partial_fill_settlement_win_and_loss`.
- Частичное исполнение: 10 shares @ 0.05 (затраты 0.501 USDC):
  - При исходе **WIN**: выплата 10 × 1.0 = 10.00 USDC, чистый PnL = +9.499 USDC.
  - При исходе **LOSS**: выплата 0, чистый PnL = -0.501 USDC (не -1.00 USDC!).
  - **Повторный settlement**: повторный вызов `settle_resolved_position` на уже закрытой позиции является идемпотентным no-op, не дублирует PnL.

---

## Этап 5. Конкурентность и повторные заявки (Пункты 16–20)

### 16. Идемпотентность и атомарная бронь
- Создана таблица `ct_decision_reservations` (`CTDecisionReservation`).
- Первичный ключ: `f"CT:{spec_id}:{market_id}"`.
- Вставка выполняется через атомарный `INSERT ... ON CONFLICT DO NOTHING`.
- При повторном обращении атомарно инкрементируется `repeat_count` через прямой SQL UPDATE (`repeat_count = repeat_count + 1`) и обновляется `last_repeat_at`.

### 17. Конкурентный запуск двух и трех воркеров
- Тесты:
  - `test_11_concurrency_two_sessions_atomic_reservation`: два изолированных сессионных подключения к SQLite (режим WAL), синхронизированные через `asyncio.Barrier(2)`. Один поток успешно бронирует и отправляет ордер, второй натыкается на конфликт ключа, вызывает `EnqueueRejected(ActiveExecutionConflict)` и откатывает вложенный savepoint. В БД фиксируется ровно 1 сделка и 1 бронь с `repeat_count = 1`.
  - `test_11b_concurrency_rollback_releases_lock`: откат первой сессии до подтверждения брони полностью освобождает блокировку; вторая сессия успешно захватывает бронь с `repeat_count = 0`.
  - `test_11c_atomic_repeat_count_increment_under_concurrency`: три одновременных повторных запроса через `asyncio.Barrier(3)` инкрементируют `repeat_count` через атомарный `update(CTDecisionReservation).where(...).values(repeat_count=repeat_count + 1)`, гарантируя `repeat_count = 3` без race conditions и потерь обновлений.

### 18. Повторный запуск после settlement
- Тест: `test_12_rerun_after_settlement_blocked`.
- После завершения сделки и закрытия позиции через settlement вызов `decide_ct_outsider_mode` на том же маркете сразу находит существующую бронь и возвращает `SKIP` с причиной `ALREADY_DECIDED: BUY`. Никаких новых ордеров не создается.

### 19. Прерывание и восстановление в 3 точках + зависший lease
- Тесты:
  - `test_13_recovery_at_three_failure_points`:
    1. **Сбой между бронью и outbox enqueue:** Имитация ошибки при сохранении в outbox приводит к откату сейвпойнта в `execute_and_record`, не оставляя брони-сироты в базе данных. Повторный запуск цикла успешно находит маркет и выставляет заявку.
    2. **Сбой после enqueue в состоянии READY:** Заявка сохранена в БД. Независимый экземпляр воркера обнаруживает ее через `claim_one` (без передачи объектов через память), исполняет через шлюз и переводит в терминальный статус.
    3. **Сбой после сохранения fills до подтверждения учета:** `ExecutionFill` записан в БД, но воркер упал до обновления `TradeHistory`. При перезапуске сервис `rebuild_trade_accounting` восстанавливает позицию в статус `OPEN` с точными `entry_filled_shares` и `entry_cost_usdc`.
  - `test_13b_recovery_stuck_claimed_request`: Заявка, зависшая в статусе `CLAIMED` с истекшим `lease_expires_at`, перехватывается воркером через `reclaim_expired_claims`, отдается в шлюз и успешно исполняется.

### 20. Неизменность первого решения
- Тест: `test_14_first_decision_immutability_on_quote_change`.
- Если первым решением была покупка UP, а на следующем тике цены резко развернулись и дешевле стал DOWN, `decide_ct_outsider_mode` отклоняет разворот, возвращая `SKIP` с `ALREADY_DECIDED: BUY` и сохраняя первоначальное направление `UP`.

---

## Этап 6. Проверка, трассировка, тайминги и исторический реплей (Пункты 21–24)

### 21. Диагностика моделей и инварианты профиля
- В `decision_runners.py` словарь `model_availability` честно фиксирует:
  - `component_only = True`
  - `prediction_made = False`
- Это исключает ложное впечатление, будто модели ML производили инференс в CT-режиме.
- Тест `test_15_paper_profile_report_invariant`: проверен инвариант отчета `Total = UP + DOWN` в `build_profile_report`.
- Тест `test_17_unassigned_parity_skips_do_not_pollute_down_side`: подтверждено, что пропуски по PARITY сохраняются под категорией `UNASSIGNED` и не искажают метрики стороны `DOWN`.
- Тест `test_18_decision_runners_never_substitutes_ask_for_mid_when_bid_none`: отсутствие bid не подменяется ask для mid цены (тест полностью асинхронен, 0 предупреждений о невызванных корутинах).
- Тест `test_19_market_guards_immutability_and_skip_reason_preservation`: неизменность записей в market guards и сохранение оригинальной причины пропуска.
- Тест `test_20_decision_at_timing_fidelity`: точная проверка разделения времени старта цикла (`cycle_started_at`) и времени решения (`decision_at`). В строке брони `CTDecisionReservation.decision_at` и в `decision_details["timing_diagnostics"]` фиксируется точный момент решения после поступления котировок, а `time_left_sec` вычисляется строго относительно `decision_at`.

### 22. Сквозная трассировка ID по реальным строкам БД
- Тест: `test_16_full_audit_chain_traceability_from_real_db_rows`.
- Без моков и ручных подстановок через единый вызов `_run_production_paper_cycle` подтверждена непрерывная цепочка ссылок в реальной БД:
  `opportunity_id` (`market_id` + `decision_at`)  
  ↳ `decision_id` (`CTDecisionReservation.key` = `CT:BTC_CT_T5_V1:mkt_trace_chain`)  
  ↳ `trade_id` (`TradeHistory.id`)  
  ↳ `request_id` (`ExecutionRequest.id`)  
  ↳ `attempt_id` (`ExecutionAttempt.id`)  
  ↳ `fill_id` (`ExecutionFill.id`)  
  ↳ `settlement_id` (`position_status = CLOSED`, `realized_pnl_usdc = 3.99002`).

### 23. Исторический реплей (Replay Validation)
- Прогон `tests/research/test_ct_historical_replay.py`:
  - `test_historical_yes_only_replay_matches_all_647_ids`: **100% совпадение всех 647 ID** маркетов с эталонным датасетом.
  - `test_historical_economics_reproduces_81_6477_usdc`: **ровно +81.6477 USDC** чистого PnL.
  - **Результат:** 2 passed, абсолютная детерминированность.

### 24. Итоговая сводка тестов
```
tests/trading/test_ct_policy.py ..............................           [30 passed]
tests/trading/test_ct_synthetic_cycle.py .......................         [23 passed]
tests/research/test_ct_historical_replay.py ..                           [2 passed]
============================= 55 passed in 5.14s ==============================
```
Полный регрессионный набор тестов (`tests/trading/`, `tests/research/`): **347 passed, 1 skipped, 0 failures, 0 errors**.
Все 3 предупреждения об unawaited coroutines устранены (предупреждения pytest теперь исключительно внешние UserWarning из scikit-learn).
