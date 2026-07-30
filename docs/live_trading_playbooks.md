# Операционные плейбуки LIVE-торговли

Данный документ содержит операционные регламенты и подробные пошаговые инструкции для проведения тестовых и боевых прогонов модуля LIVE-зеркалирования в проекте **polyflip**.

---

## Этап 10. SHADOW-прогон (dry run с реальными ордерами на 0$)

### Цель
Безопасная проверки полной цепочки зеркалирования ордеров `PAPER -> LiveMirrorCandidate -> Release Gate (SHADOW)` без использования реального депозита USDC и без совершения финансовых операций в сети Polygon.

### Условия запуска
1. Запущен и функционирует основной PAPER-стек (`polyflip_db`, `polyflip_api`, `polyflip_execution_worker_paper`).
2. Применены последние миграции Alembic.
3. Отсутствуют блокирующие ошибки в логах основной БД и API.
4. В PAPER-режиме генерируются сигналы и создаются исполненные ордера (`state IN ('FILLED', 'PARTIALLY_FILLED_FINAL')`).

### Команды
Запуск изолированного тестового окружения в режиме SHADOW:
```bash
docker compose -f docker-compose.live-test.yml --profile live-test up -d
```

Мониторинг работы воркеров зеркалирования и шлюза выпуска:
```bash
docker compose -f docker-compose.live-test.yml logs -f live_mirror_worker_test
docker compose -f docker-compose.live-test.yml logs -f release_gate_worker_test
```

### Проверки
1. **Проверка создания SHADOW-кандидатов в БД:**
   ```sql
   SELECT id, source_paper_request_id, source_paper_trade_id, target_mode, state, signal_hash, created_at
   FROM live_mirror_candidates
   WHERE target_mode = 'SHADOW'
   ORDER BY created_at DESC
   LIMIT 10;
   ```

2. **Проверка работы Release Gate (выпуск SHADOW-ордеров):**
   ```sql
   SELECT id, mode, position_status, created_at
   FROM trade_history
   WHERE mode = 'SHADOW'
   ORDER BY created_at DESC
   LIMIT 10;

   SELECT id, requested_mode, intent, state, created_at
   FROM execution_requests
   WHERE requested_mode = 'SHADOW'
   ORDER BY created_at DESC
   LIMIT 10;
   ```

3. **Проверка статуса через REST API:**
   ```bash
   curl -s -H "X-API-Key: test-key" http://localhost:8001/api/execution/status
   curl -s -H "X-API-Key: test-key" "http://localhost:8001/api/execution/candidates?state=NEW"
   ```

### Критерии успеха
* Отсутствие дубликатов `LiveMirrorCandidate` для одной исходной PAPER-заявки (соблюдение уникального индекса по `source_paper_request_id` + `target_mode`).
* Успешный перевод состояний кандидатов out-of-band: `NEW` -> `ELIGIBLE` -> `RELEASED`.
* Создание соответственных записей в `trade_history` и `execution_requests` со статусом `mode='SHADOW'`.
* Исходные записи `PAPER` остаются нетронутыми (строгая декуплизация и неизменяемость PAPER-истории).

### Откат
Для остановки тестового SHADOW-окружения и очистки контейнеров:
```bash
docker compose -f docker-compose.live-test.yml --profile live-test down
```
Отключение флага mirror-воркера через API (при необходимости):
```bash
curl -X PUT http://localhost:8001/api/execution/mirror-switch \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test-key" \
  -d '{"enabled": false}'
```

---

## Этап 11. Первый микро-LIVE прогон

### Цель
Первичный ввод модуля LIVE-торговли в производственную эксплуатацию с реальным исполнением сделок на Polymarket при минимально допустимых рисках (микро-лимит объема сделок, постоянный операторский контроль).

### Условия запуска (pre-flight чеклист)
- [ ] Успешно пройден Stage 10 (SHADOW-прогон без ошибок в течение минимум 24 часов).
- [ ] На целевом Polygon-кошельке (`POLYGON_ADDRESS`) имеется достаточное количество NATIVE токенов (POL/MATIC) для оплаты gas fees.
- [ ] На балансе кошелька имеется тестовый депозит USDC (не менее $5.00 USDC).
- [ ] В окружении корректно заданы `POLYGON_PRIVATE_KEY` и `POLYGON_ADDRESS`.
- [ ] Выполнено подтверждение разрешения расходов (Collateral Allowance & Conditional Allowance для контрактов Polymarket CTF Exchange).
- [ ] Сформирована внешняя сеть `polyflip_net` (`docker network create polyflip_net`).

### Шаги
1. **Поднятие production-сервисов LIVE v2:**
   ```bash
   docker compose -f docker-compose.live-v2.yml --profile live-v2 up -d
   ```

2. **Проверка готовности воркера исполнения:**
   ```bash
   curl -s -H "X-API-Key: test-key" http://localhost:8001/api/execution/status
   ```
   *Убедитесь, что `gateway_ready: true`, `credentials_loaded: true`, `balance_usdc >= 5.0`.*

3. **Включение воркера зеркалирования (Mirror Worker):**
   ```bash
   curl -X PUT http://localhost:8001/api/execution/mirror-switch \
     -H "Content-Type: application/json" \
     -H "X-API-Key: test-key" \
     -d '{"enabled": true}'
   ```

4. **Установка безопасного режимов выпуска `MANUAL` (ручное подтверждение первых сделок):**
   ```bash
   curl -X PUT http://localhost:8001/api/execution/release-mode \
     -H "Content-Type: application/json" \
     -H "X-API-Key: test-key" \
     -d '{"mode": "MANUAL"}'
   ```

5. **Ручной выпуск первого кандидата в LIVE:**
   Получить id первого кандидата со статусом `NEW`:
   ```bash
   curl -s -H "X-API-Key: test-key" "http://localhost:8001/api/execution/candidates?state=NEW"
   ```
   Одобрить кандидат вручную:
   ```bash
   curl -X POST http://localhost:8001/api/execution/candidates/<CANDIDATE_ID>/release \
     -H "X-API-Key: test-key"
   ```

6. **Активация главного Kill-Switch LIVE-торговли:**
   ```bash
   curl -X PUT http://localhost:8001/api/execution/kill-switch \
     -H "Content-Type: application/json" \
     -H "X-API-Key: test-key" \
     -d '{"enabled": true}'
   ```

7. **Перевод Release Gate в автоматический режим (после успешной тестовой сделки):**
   ```bash
   curl -X PUT http://localhost:8001/api/execution/release-mode \
     -H "Content-Type: application/json" \
     -H "X-API-Key: test-key" \
     -d '{"mode": "AUTO"}'
   ```

### Мониторинг
* **Просмотр логов воркеров в реальном времени:**
  ```bash
  docker compose -f docker-compose.live-v2.yml --profile live-v2 logs -f --tail=100
  ```
* **Мониторинг активных LIVE-позиций через SQL:**
  ```sql
  SELECT id, market_id, asset, outcome_bought, entry_cost_usdc, position_status, created_at
  FROM trade_history
  WHERE mode = 'LIVE' AND position_status IN ('OPEN', 'PARTIALLY_FILLED')
  ORDER BY created_at DESC;
  ```
* **Проверка состояния воркера и задержки heartbeat:**
  ```sql
  SELECT worker_id, execution_mode, heartbeat_at, gateway_ready, balance_usdc, last_error_message
  FROM execution_worker_status
  WHERE execution_mode = 'LIVE';
  ```

### Критерии успеха
* Исполненный PAPER OPEN ордер успешно конвертируется в ордер Polymarket CLOB.
* Хэш ончейн-транзакции фиксируется в `chain_transactions`.
* Статус позиции переходит в `OPEN` со списанием соответствующей суммы USDC.
* В таблице `execution_events` отсутствуют критические ошибки (`CRITICAL`, `ERROR`).

### Аварийный останов
При обнаружении аномального поведения, проскальзывания цен или рассинхронизации баланса выполнить **Immediate Emergency Shutdown**:

1. **Мгновенное выключение главного Kill-Switch через API:**
   ```bash
   curl -X PUT http://localhost:8001/api/execution/kill-switch \
     -H "Content-Type: application/json" \
     -H "X-API-Key: test-key" \
     -d '{"enabled": false}'
   ```
2. **Отключение mirror-воркера:**
   ```bash
   curl -X PUT http://localhost:8001/api/execution/mirror-switch \
     -H "Content-Type: application/json" \
     -H "X-API-Key: test-key" \
     -d '{"enabled": false}'
   ```
3. **Остановка LIVE-контейнеров:**
   ```bash
   docker compose -f docker-compose.live-v2.yml --profile live-v2 down
   ```
4. **Прямое сбрасывание флага в БД (при недоступности API):**
   ```sql
   INSERT INTO runtime_settings (key, value, updated_at, updated_by)
   VALUES ('LIVE_TRADING_ENABLED', 'false', NOW(), 'emergency_operator')
   ON CONFLICT (key) DO UPDATE SET value = 'false', updated_at = NOW();
   ```

---

## Этап 12. Закрытие позиций и редемпция

### Цель
Гарантированное и безопасное закрытие открытых LIVE-позиций (продажа токенов исхода до экспирации или погашение токенов победившего исхода после завершения рынка).

### Автоматическое закрытие (stop-loss / take-profit)
* При срабатывании условий Stop-Loss или Take-Profit модуль `polyflip.execution.worker` генерирует интент `CLOSE` со статусом `requested_mode='LIVE'`.
* **SQL-запрос для контроля автоматических закрытий:**
  ```sql
  SELECT id, market_id, asset, position_status, stop_loss_status, take_profit_status, realized_pnl_usdc, updated_at
  FROM trade_history
  WHERE mode = 'LIVE' AND position_status = 'CLOSED'
  ORDER BY updated_at DESC
  LIMIT 20;
  ```

### Ручное закрытие
Если требуется принудительно закрыть LIVE-позицию (например, перед техническими работами):
1. **Найти ID активной сделки:**
   ```sql
   SELECT id, market_id, outcome_bought, entry_cost_usdc
   FROM trade_history
   WHERE mode = 'LIVE' AND position_status = 'OPEN';
   ```
2. **Создать ручную заявку на закрытие:**
   ```sql
   INSERT INTO execution_requests (
       id, trade_history_id, requested_mode, intent, market_id, outcome_to_buy, target_amount_usdc, state, created_at, updated_at
   )
   SELECT
       gen_random_uuid(), id, 'LIVE', 'CLOSE', market_id, outcome_bought, entry_cost_usdc, 'PENDING', NOW(), NOW()
   FROM trade_history
   WHERE id = <TRADE_ID> AND mode = 'LIVE' AND position_status = 'OPEN';
   ```

### Редемпция токенов
После финализации рынка на Polymarket (Market Settlement / Resolution) выигрышные токены исхода должны быть обменяны обратно на USDC по курсу $1.00 за токен.

1. **Запуск службы автоматического погашения (Settlement & Redemption):**
   ```bash
   python -m polyflip.execution.settlement_service --mode=LIVE
   ```
2. **Проверка подтверждений транзакций погашения в БД:**
   ```sql
   SELECT tx_hash, operation, network, gas_paid_usdc, confirmed_at
   FROM chain_transactions
   WHERE operation = 'REDEEM'
   ORDER BY confirmed_at DESC
   LIMIT 10;
   ```

### Проверки после закрытия
1. **Проверка отсутствия подвисших токенов:**
   ```sql
   SELECT id, market_id, remaining_shares, position_status
   FROM trade_history
   WHERE mode = 'LIVE' AND remaining_shares > 0 AND position_status = 'CLOSED';
   ```
   *Ожидается 0 строк.*
2. **Сверка баланса кошелька и совокупного PnL:**
   ```bash
   curl -s -H "X-API-Key: test-key" http://localhost:8001/api/execution/status
   ```
   *Поле `balance_usdc` должно отражать возвращенный депозит с учетом зафиксированного `realized_pnl_usdc`.*
