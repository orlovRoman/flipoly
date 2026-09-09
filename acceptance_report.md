# 📋 Приёмочный отчёт: Реализация замечаний приёмки CT Outsider PAPER

**Ветка:** `feature/paper-ct-outsider`  
**Базовый коммит исходного состояния:** `a325099`  
**Дата приёмки:** 2026-09-09  
**Статус:** ✅ **ВСЕ 24 ПУНКТА ПРИЁМКИ ВЫПОЛНЕНЫ И ПРОТЕСТИРОВАНЫ**

---

## Этап 1. Исходное состояние и анализ (Пункты 1–4)

### 1. Проверяемая версия и рабочее дерево
- **Коммит:** `a325099` (`feat(ct-policy): implement BTC_CT_T5_V1 symmetric outsider paper policy and replay`).
- **Ветка:** `feature/paper-ct-outsider`.
- **Рабочее дерево:** Исходно чистое состояние перед началом работ. Все изменения зафиксированы в рамках ветки.

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
- При лимите 0.20 и объеме 100 shares исполняется ровно 5.0 shares.
- `spent_usdc` = 1.00 USDC, комиссия 0.002 USDC.
- `TradeHistory`: `entry_filled_shares = 5.0`, `position_status = "OPEN"`.
- При выигрыше YES: `settle_resolved_position` закрывает сделку (`position_status = "CLOSED"`), `realized_pnl_usdc = +3.998` USDC.

### 12. Частичное исполнение (Partial Fill)
- Тест: `test_07_synthetic_partial_fill_liquidity_preserves_unspent_budget`.
- В стакане доступно только 10 shares @ 0.05 ($0.50 глубина) при бюджете $1.00.
- Исполняется 10 shares @ 0.05, затрачено $0.50 + комиссия 0.001 USDC.
- Неиспользованный остаток бюджета $0.50 сохраняется на балансе и **не списывается в убыток**.

### 13. Изменение лимитной цены (Limit Price Change)
- Тест: `test_08_synthetic_limit_price_change_rejection`.
- Решение принято по лимиту 0.20. В момент исполнения лучший ask ушел на 0.21.
- Шлюз `FakeExecutionGateway` отклоняет заявку (`PAPER_PRICE_MOVED`), 0 fills.
- Запрос финализируется со статусом `REJECTED`, сделка переводится в `ENTRY_FAILED`, баланс не расходуется.

### 14. Многоуровневый стакан и VWAP
- Тест: `test_09_synthetic_multi_level_orderbook_vwap`.
- Стакан: Уровень 1 (5 shares @ 0.10, $0.50), Уровень 2 (10 shares @ 0.12).
- Заявка забирает 5 shares @ 0.10 ($0.50) и 4.166667 shares @ 0.12 ($0.50).
- Всего куплено: 9.166667 shares за 1.00 USDC.
- Фактическая цена входа (`executed_price` в `TradeHistory`) фиксирует точный VWAP = 1.00 / 9.166667 ≈ **0.109091** USDC/share.

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
- При повторном обращении атомарно инкрементируется `repeat_count` и обновляется `last_repeat_at`.

### 17. Конкурентный запуск двух воркеров
- Тест: `test_11_concurrency_atomic_reservation`.
- Два одновременных вызова `execute_and_record` для одного маркета:
  - Первый вызов успешно бронирует запись и отправляет ордер в outbox.
  - Второй вызов натыкается на конфликт ключа, вызывает `EnqueueRejected(ActiveExecutionConflict)` и безопасно откатывает savepoint.
  - В БД сохраняется ровно 1 сделка и 1 бронь с `repeat_count = 1`.

### 18. Повторный запуск после settlement
- Тест: `test_12_rerun_after_settlement_blocked`.
- После завершения сделки и закрытия позиции через settlement вызов `decide_ct_outsider_mode` на том же маркете сразу находит существующую бронь и возвращает `SKIP` с причиной `ALREADY_DECIDED: BUY`. Никаких новых ордеров не создается.

### 19. Прерывание и восстановление в 3 точках
- Тест: `test_13_recovery_at_three_failure_points`.
  1. **После брони до outbox:** Наличие брони при перезапуске предотвращает повторный вход.
  2. **После enqueue в состоянии READY:** Воркер перезапускается, забирает заявку через `claim_one`, передает в шлюз и успешно исполняет.
  3. **После сохранения fills до бухгалтерского подтверждения:** Воркер перезапускается, вызывает `rebuild_trade_accounting`, который по существующим записям `ExecutionFill` восстанавливает позицию в статус `OPEN` с корректными `entry_filled_shares` и `entry_cost_usdc`.

### 20. Неизменность первого решения
- Тест: `test_14_first_decision_immutability_on_quote_change`.
- Если первым решением была покупка UP, а на следующем тике цены резко развернулись и дешевле стал DOWN, `decide_ct_outsider_mode` отклоняет разворот, возвращая `SKIP` с `ALREADY_DECIDED: BUY` и сохраняя первоначальное направление `UP`.

---

## Этап 6. Проверка, трассировка и исторический реплей (Пункты 21–24)

### 21. Диагностика моделей
- В `decision_runners.py` словарь `model_availability` честно фиксирует:
  - `component_only = True`
  - `prediction_made = False`
- Это исключает ложное впечатление, будто модели ML производили инференс в CT-режиме.
- Проверен инвариант отчета `Total = UP + DOWN` (`test_15_paper_profile_report_invariant`).

### 22. Сквозная трассировка ID по реальным строкам БД
- Тест: `test_16_full_audit_chain_traceability_from_real_db_rows`.
- Подтверждена непрерывная цепочка ссылок в базе данных:
  `opportunity_id` (`market_id` + `timestamp`)  
  ↳ `decision_id` (`CTDecisionReservation.key` = `CT:BTC_CT_T5_V1:mkt_trace_chain`)  
  ↳ `trade_id` (`TradeHistory.id`)  
  ↳ `request_id` (`ExecutionRequest.id`)  
  ↳ `attempt_id` (`ExecutionAttempt.id`)  
  ↳ `fill_id` (`ExecutionFill.id`)  
  ↳ `settlement_id` (`position_status = CLOSED`, `realized_pnl_usdc`).

### 23. Исторический реплей (Replay Validation)
- Прогон `tests/research/test_ct_historical_replay.py`:
  - **Сделок:** 647
  - **PnL:** +81.6477 USDC
  - **Результат:** 2 passed, 100% совпадение с эталоном.

### 24. Итоговая сводка тестов
```
tests/trading/test_ct_policy.py ..............................           [30 passed]
tests/trading/test_ct_synthetic_cycle.py ...................             [19 passed]
tests/research/test_ct_historical_replay.py ..                           [2 passed]
tests/execution/test_paper_e2e.py ....                                   [4 passed]
============================= 55 passed in 5.8s ==============================
```
Полный набор тестов (`tests/trading/`, `tests/execution/`, `tests/research/`): **414 passed, 3 skipped**.
