# Отчет об устранении блокирующих расхождений для Live-калибровки LP Rewards

В рамках данной итерации были детально проанализированы и устранены все выявленные блокирующие расхождения и скрытые заглушки, препятствовавшие безопасной live-калибровке и корректному учету в исследовании LP Rewards (ветка `research/lp-rewards`).

---

## 1. Спецификация EIP-712 и контракты CLOB V2

- **Канонический адрес контракта**: В `polyflip/research/lp_rewards/execution.py` обновлен адрес биржевого контракта CTF Exchange V2 на официальный:
  ```python
  CTF_EXCHANGE_ADDRESS = "0xE11118001712aA868d4aB713A2c8f85fBB9161aB"
  ```
- **Структура EIP-712 Order Struct V2**:
  - Полностью удалены устаревшие поля спецификации V1 (`taker`, `expiration`, `nonce`, `feeRateBps`).
  - Добавлены и приведены к типам поля V2:
    - `timestamp`: целое число в миллисекундах (`int(time.time() * 1000)`).
    - `metadata`: тип `bytes32` (`"0x" + "00" * 32`).
    - `builder`: тип `bytes32` (`"0x" + "00" * 32`).
    - `signatureType`: `0` (EOA).
  - Сформированный объект типизированных данных валидируется и передается в метод `post_order` клиента CLOB.

---

## 2. Переход на официальный SDK CLOB V2 (`py-clob-client-v2`)

- **Установка и зависимости**:
  - Установлен и добавлен в `pyproject.toml` официальный пакет `py-clob-client-v2 = "^1.1.0"`.
- **Инициализация клиента в `06_run_live_calibration.py`**:
  - Вместо legacy `py_clob_client` используется `from py_clob_client_v2 import ClobClient, ApiCreds`.
  - Устранен небезопасный fallback на `MockClobClient`. При любой ошибке инициализации или отсутствия боевых учетных записей раннер немедленно завершает работу (`sys.exit(1)`).

---

## 3. Валидация баланса и Allowance токена pUSD через On-Chain JSON-RPC

- **Адрес токена pUSD на Polygon**:
  ```python
  PUSD_ADDRESS = "0xC011a7E12a19f7b1f670d46f03b03f3342e82dfb"
  ```
- **Прямые JSON-RPC вызовы**:
  - Реализованы методы `get_pusd_balance_onchain(wallet_address)` и `get_pusd_allowance_onchain(wallet_address, spender_address)` через HTTP JSON-RPC `eth_call` к ноде Polygon (`https://polygon-rpc.com` или переменная `POLYGON_RPC_URL`).
  - Кодирование селекторов: `0x70a08231` (`balanceOf`) и `0xdd62ed3e` (`allowance`) с 6 десятичными знаками (USDC/pUSD 10^6).
- **Принцип Fail-Closed**:
  - При любой сетевой ошибке или недоступности ноды RPC возвращается `Decimal("0.0")`, что приводит к немедленному отклонению ордера по `Insufficient live balance` или `Insufficient collateral allowance`.

---

## 4. Полноценный FSM цикл в `06_run_live_calibration.py`

- Вместо 5-секундного разового дымового теста реализован непрерывный цикл `while True` с интервалом котирования.
- На каждом шаге опрашивается состояние `MarketQuotingFSM`, проверяется лимит рабочего капитала через `CapitalAllocator`, генерируются котировки, отправляются через `LiveOrderExecutor` и сохраняются в `live_open_orders.json`.
- При прерывании или ошибке выполняется блок `finally: executor.cancel_all_orders()`, гарантирующий снятие всех активных заявок.

---

## 5. Конкурентный скоринг и учет времени котирования

- **Модульная функция `subtract_orders`**: В модуле `scoring.py` функция `subtract_orders` вынесена на уровень модуля и вычитает собственные виртуальные ордера из публичного стакана до расчета конкурентного знаменателя $Q_{one}$ и $Q_{two}$. Это исключает искусственное завышение объемов стакана нашими же заявками.
- **Расчет `quote_hours`**: В `03_run_shadow_collector.py` расчет `quote_hours` переведен с общего астрономического времени жизни процесса на взвешенное время активного присутствия котировок в рынке:
  $$\text{quote\_hours} = \frac{\text{avg\_ticks} \times \text{interval\_sec}}{3600.0}$$

---

## 6. Сверка вознаграждений (Reconciliation) и Gate B

- **L2 аутентификация в `reward_calibration.py`**:
  - При запросе `/rewards/user` теперь проверяется наличие всех четырех обязательных L2 заголовков: `POLY_ADDRESS`, `POLY_SIGNATURE`, `POLY_TIMESTAMP`, `POLY_PASSPHRASE`. При их отсутствии выбрасывается `PermissionError`.
  - Реализована пагинация через параметр `next_cursor` вплоть до терминального маркера `"LTE="`.
- **Дневная агрегация в `08_reconcile_rewards.py`**:
  - Даты из API нормализуются до формата `YYYY-MM-DD`.
  - Фактические начисления суммируются за весь день по всем рынкам для корректного сопоставления с общим дневным счетчиком `total_rewards_accrued`.
- **Fail-Closed в `09_evaluate_gate_b.py`**:
  - Значение по умолчанию для `mean_prediction_error` установлено в `Decimal("1.0")` (100% ошибка). При отсутствии файлов сверки Gate B гарантированно отклоняется (`INCONCLUSIVE` / `TARGET_NOT_CONFIRMED`), а не пропускается с фиктивной нулевой ошибкой.

---

## 7. Строгие проверки Gate A в `evaluation.py`

- **Обязательное наличие `protocol_hash`**: Проверка отклоняет оценку, если `protocol_hash` отсутствует в записях или не совпадает с ожидаемым хэшем протокола.
- **Контроль уникальности дат**: Проверяется отсутствие дублирующихся календарных дат в ежедневных отчетах.

---

## 8. Самопроверки и результаты тестирования

В `tests/research/lp_rewards/test_deep_fixes.py` добавлено 18 специализированных unit-тестов:
1. `test_executor_clob_v2_exact_struct_and_address`: проверка официального адреса `0xE1111800...`, отсутствия полей V1 и типов V2.
2. `test_pusd_onchain_rpc_balance_and_allowance_mocked`: проверка RPC-запросов к pUSD, десериализации 6 знаков и fail-closed блокировки при сбое ноды.
3. `test_reconcile_rewards_l2_headers_missing_raises_permission_error`: проверка требования L2 заголовков.
4. `test_reconcile_rewards_pagination_and_date_sum`: проверка пагинации до `"LTE="` и сложения сумм выплат.
5. `test_gate_b_evaluator_fail_closed_when_recon_missing`: проверка отклонения Gate B при отсутствии логов сверки.
6. `test_shadow_collector_quote_hours_calculation`: проверка формулы реальных часов котирования.
7. `test_subtract_orders_deduplication`: изолированная проверка вычитания собственных заявок из стакана.
8. `test_clob_v2_sdk_import_and_init`: проверка импорта и инициализации SDK `py_clob_client_v2`.
9. `test_evaluate_gate_a_missing_protocol_hash_rejected`: проверка отсечения Gate A при отсутствии хэша протокола.
10. `test_evaluate_gate_a_duplicate_dates_rejected`: проверка отсечения Gate A при дублировании календарных дат.

### Результат прогона тестов:
```bash
python -m pytest tests/research/lp_rewards -v
============================= 78 passed in 2.43s ==============================
```
Все **78 тестов** в тестовом наборе проходят успешно.


## 10. Безусловный Fail-Closed контроль изолированного кошелька и синхронизация Poetry Lock (Итерация 3)

В третьей итерации устранены последние критические замечания аудита:

1. **Безусловный Fail-Closed в Gate B (`09_evaluate_gate_b.py`)**:
   - Переменная окружения `LP_ISOLATED_WALLET_ADDRESS` теперь проверяется **в самом начале** функции `main()`, до загрузки файлов и расчетов. При отсутствии или пустой строке скрипт немедленно логирует ошибку `LP_ISOLATED_WALLET_ADDRESS is missing. Rejecting Gate B.` и завершается с `sys.exit(1)`.
   - Проверка каталога сверки `storage_path / "reconciliation"` и файлов `rewards_*.json` переведена в строгий режим fail-closed: если файлов сверки нет, выполнение не продолжается с дефолтной ошибкой, а немедленно прерывается с `sys.exit(1)`.
   - Добавлены проверки несовпадения кошелька в отчете сверки с изолированным кошельком и несовпадения хэша протокола.

2. **Безусловный Fail-Closed кошелька в `LiveOrderExecutor.submit_order()` (`execution.py`)**:
   - В метод `submit_order()` добавлена безусловная проверка: `self.wallet_address` обязан быть задан, не быть пустым и не быть нулевым адресом (`0x0000000000000000000000000000000000000000`). В противном случае выбрасывается `PermissionError("Live execution requires configured non-zero wallet_address")`.
   - Метод `self.verify_isolated_wallet()` вызывается **безусловно**, даже если параметры `live_balance` и `allowance` были переданы вручную. Это исключает генерацию ордеров с нулевым `maker` при любых условиях.

3. **Синхронизация зависимостей и `poetry.lock`**:
   - В `pyproject.toml` в секцию `[tool.poetry.dependencies]` явно добавлена зависимость `websockets = ">=12.0,<16.0"`, а также подтверждены `py-clob-client-v2 = "^1.1.0"`, `eth-account = "^0.13.0"`, `pyarrow = "^17.0.0"`. В секции dev-зависимостей подтвержден `respx = "^0.21.1"`.
   - Запущен `poetry lock` (через Poetry 2.4.3), перегенерировавший `poetry.lock`. Лок-файл теперь полностью содержит `py-clob-client-v2`, `websockets`, `eth-account`, `pyarrow`, `fastapi`, `asyncpg`.
   - Синхронизирован `uv.lock`.
   - Проверена сборка зависимостей для `Dockerfile` через `poetry export -f requirements.txt --without-hashes`.

4. **Инструкция по запуску тестов на сервере**:
   > ⚠️ **Важное пояснение по серверному окружению**:
   > На боевом сервере системный бинарник `/usr/bin/python3` не содержит библиотек проекта. Запуск `python3 -m pytest ...` приводит к ошибкам `ModuleNotFoundError` (`websockets`, `pyarrow`, `respx`, `eth_account`, `py_clob_client_v2`).
   >
   > Для корректного запуска тестов на сервере **необходимо использовать виртуальное окружение Poetry**:
   > ```bash
   > # Вариант 1 (рекомендуемый): через Poetry CLI
   > poetry run pytest tests/research/lp_rewards -v
   >
   > # Вариант 2: через активированное виртуальное окружение
   > source .venv/bin/activate
   > pytest tests/research/lp_rewards -v
   >
   > # Вариант 3: прямой вызов python из virtualenv
   > ./.venv/bin/python -m pytest tests/research/lp_rewards -v
   > ```

5. **Результаты локального тестирования и валидации**:
   - Добавлены unit-тесты на немедленное падение Gate B при отсутствии кошелька или отчетов сверки, а также тесты на отклонение нулевого и отсутствующего адреса кошелька в `LiveOrderExecutor`.
   - Результат прогона всего тестового набора:
     ```bash
     pytest tests/research/lp_rewards -v
     ============================= 95 passed in 4.14s ==============================
     ```
   - Команда `git diff --check` проходит с нулевым кодом возврата (нет trailing whitespace и некорректных переводов строк).
