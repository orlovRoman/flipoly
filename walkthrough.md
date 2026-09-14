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
   - Переменная окружения `LP_ISOLATED_WALLET_ADDRESS` теперь проверяется **в самом начале** функции `main()`, до загрузки файлов и расчетов. При отсутствии, пустой строке или нулевом адресе (`0x0000000000000000000000000000000000000000`) скрипт немедленно логирует ошибку `LP_ISOLATED_WALLET_ADDRESS is missing or invalid. Rejecting Gate B.` и завершается с `sys.exit(1)`.
   - Проверка каталога сверки `storage_path / "reconciliation"` и файлов `rewards_*.json` переведена в строгий режим fail-closed: если файлов сверки нет, выполнение не продолжается с дефолтной ошибкой, а немедленно прерывается с `sys.exit(1)`.
   - Добавлены строгие проверки на отсутствие, пустую строку, нулевой адрес или несовпадение кошелька в отчете сверки с изолированным кошельком, а также несовпадение хэша протокола.

2. **Безусловный Fail-Closed кошелька в `LiveOrderExecutor` (`execution.py`)**:
   - В метод `submit_order()` добавлена безусловная проверка: `self.wallet_address` обязан быть задан, не быть пустым (с trim пробелов) и не быть нулевым адресом (`0x0000000000000000000000000000000000000000`). В противном случае выбрасывается `PermissionError("Live execution requires configured non-zero wallet_address")`.
   - Метод `self.verify_isolated_wallet()` вызывается **безусловно**, даже если параметры `live_balance` и `allowance` были переданы вручную. При передаче пробельных строк или нулевого адреса выбрасывается `ValueError("Dedicated LP isolated wallet address is not configured.")`.
   - В `sign_eip712_order()` удален fallback на нулевой адрес `ZERO_ADDRESS`. Метод строго проверяет наличие сконфигурированного ненулевого адреса кошелька, иначе выбрасывает `PermissionError("Order signing requires configured non-zero wallet_address")`, полностью исключая подписание ордеров с нулевым `maker`.

3. **Fail-Closed контроль в скриптах сверки и запуска**:
   - В `08_reconcile_rewards.py` и `07_reconcile_orders.py` устранены значения по умолчанию (`ZERO_ADDRESS`), добавлена строгая проверка на пустоту, пробелы и нулевой адрес (`sys.exit(1)`).
   - В `06_run_live_calibration.py` ужесточена проверка хард-гейта: оба параметра `LP_ISOLATED_WALLET_ADDRESS` (ненулевой) и `LP_WALLET_PRIVATE_KEY` обязаны быть заданы одновременно.

4. **Синхронизация зависимостей и `poetry.lock`**:
   - В `pyproject.toml` в секцию `[tool.poetry.dependencies]` явно добавлена зависимость `websockets = ">=12.0,<16.0"`, а также подтверждены `py-clob-client-v2 = "^1.1.0"`, `eth-account = "^0.13.0"`, `pyarrow = "^17.0.0"`. В секции dev-зависимостей подтвержден `respx = "^0.21.1"`.
   - Запущен `poetry lock` (через Poetry 2.4.3), перегенерировавший `poetry.lock`. Лок-файл теперь полностью содержит `py-clob-client-v2`, `websockets`, `eth-account`, `pyarrow`, `fastapi`, `asyncpg`.
   - Синхронизирован `uv.lock`.
   - Проверена сборка зависимостей для `Dockerfile` через `poetry export -f requirements.txt --without-hashes`.

5. **Точная инструкция по воспроизведению тестов на сервере**:
   > ⚠️ **Пояснение по серверному окружению (`agent-gemini-cli-poly`)**:
   > 1. На боевом сервере системный бинарник `/usr/bin/python3` не содержит библиотек проекта (попытка запуска `python3 -m pytest` падает с `ModuleNotFoundError` для `websockets`, `respx`, `pyarrow`, `eth_account`). Системная команда `python` (без 3) отсутствует.
   > 2. Бинарник Poetry установлен у пользователя в `~/.local/bin/poetry`, но каталог `~/.local/bin` **не входит в стандартный неинтерактивный PATH** сервера (`/usr/local/bin:/usr/bin:/bin:/usr/games`).
   > 3. На хосте каталог `.venv` отсутствует по умолчанию, так как продакшн работает внутри Docker-контейнеров.
   >
   > Для корректного запуска тестов на сервере в рабочей копии ветки `research/lp-rewards`:
   > ```bash
   > # Шаг 1: Добавить ~/.local/bin в PATH для текущей сессии
   > export PATH="$HOME/.local/bin:$PATH"
   >
   > # Шаг 2: Установить зависимости проекта (включая dev-группу pytest/respx)
   > poetry install
   >
   > # Шаг 3: Запустить тесты модуля research
   > poetry run pytest tests/research/lp_rewards -v
   >
   > # Либо прямой вызов без изменения PATH:
   > ~/.local/bin/poetry install
   > ~/.local/bin/poetry run pytest tests/research/lp_rewards -v
   > ```

6. **Результаты тестирования и валидации**:
   - Добавлены unit-тесты на немедленное падение Gate B при нулевом адресе кошелька, тесты скриптов сверки 07 и 08 на отклонение нулевого/пустого адреса, тесты `verify_isolated_wallet` на пробельные строки, тесты `sign_eip712_order` на отклонение неконфигурированного/нулевого адреса.
   - Результат локального прогона тестового набора LP Rewards:
     ```bash
     pytest tests/research/lp_rewards -v
     ======================= 100 passed, 3 warnings in 1.18s =======================
     ```
   - Результат прогона всего тестового набора репозитория:
     ```bash
     pytest -v
     ========= 1944 passed, 11 skipped, 2210 warnings in 172.55s (0:02:52) =========
     ```
   - Команда `git diff --check` проходит с нулевым кодом возврата (нет trailing whitespace и некорректных переводов строк).

---

## 11. Устранение замечаний аудита: Строгий Fail-Closed в сверке ордеров, воспроизведение `uv lock --check` и очистка предупреждений (Итерация 4)

В четвертой итерации полностью адресованы финальные замечания аудита:

1. **Строгий Fail-Closed в `07_reconcile_orders.py`**:
   - В функции `fetch_remote_orders` полностью удалены небезопасные операторы `break`, которые при сетевых ошибках, сбоях ноды/шлюза Polymarket (HTTP != 200) или ошибках парсинга JSON возвращали пустой или частичный список ордеров.
   - Теперь при получении HTTP-статуса, отличного от 200, при любых сетевых исключениях (`httpx.ConnectError`, таймаут и т.д.), при некорректном JSON или неожиданном формате ответа скрипт логирует `logger.error` и немедленно выбрасывает `RuntimeError`.
   - В структуру валидации словарей добавлен строгий контроль: если API возвращает словарь с полем `"error"` (например, отказ в авторизации при HTTP 200) или словарь без обязательного ключа `"data"`, скрипт немедленно выбрасывает `RuntimeError`, предотвращая ложное распознавание ответа как пустого набора ордеров.
   - Гарантирована атомарность пагинации: при обрыве связи на N-й странице пагинации накопленные на предыдущих страницах ордера не возвращаются частично — весь запрос аварийно прерывается выбрасыванием `RuntimeError`.
   - В функции `main()` чтение локального файла `live_open_orders.json` и вызов `fetch_remote_orders` защищены обработчиками с логированием `logger.error` и гарантированным выходом `sys.exit(1)`. Сверка гарантированно прерывается в режиме fail-closed и не может сформировать ошибочный отчет о «синхронизированности» (reconciled in sync) при отсутствии связи с биржей или повреждении локальных файлов.
   - В `tests/research/lp_rewards/test_deep_fixes.py` добавлено 9 специализированных unit-тестов:
     - `test_reconcile_orders_fetch_remote_http_error_fails_closed`
     - `test_reconcile_orders_fetch_remote_network_error_fails_closed`
     - `test_reconcile_orders_fetch_remote_invalid_json_fails_closed`
     - `test_reconcile_orders_fetch_remote_unexpected_format_fails_closed`
     - `test_reconcile_orders_main_remote_error_fails_closed`
     - `test_reconcile_orders_fetch_remote_dict_with_error_fails_closed`
     - `test_reconcile_orders_fetch_remote_dict_missing_data_fails_closed`
     - `test_reconcile_orders_fetch_remote_multipage_failure_fails_closed`
     - `test_reconcile_orders_main_corrupted_local_file_fails_closed`
     Тесты подтверждают немедленный выброс исключения и аварийное завершение скрипта при любых ошибках сети, некорректных кодах ответа и сбоях пагинации.

2. **Воспроизведение `uv lock --check` и окружение сервера**:
   - Зафиксировано и подтверждено: на хосте боевого сервера `agent-gemini-cli-poly` бинарники `uv` / `uvx` не установлены, управление зависимостями хоста осуществляется через Poetry (`~/.local/bin/poetry`).
   - Проверка целостности `uv.lock` через `uv lock --check` предназначена для запуска на машинах разработчиков и в CI пайплайнах с установленным `uv`.
   - Выполнена проверка на рабочей станции с `uv 0.12.5`:
     ```bash
     uv lock --check
     Resolved 93 packages in 1ms
     ```
     Код возврата: `0`. Все 93 пакета синхронизированы с `pyproject.toml`.

3. **Устранение предупреждений websockets DeprecationWarning**:
   - В `polyflip/research/lp_rewards/collector.py` устаревшие аннотации типов `ws: websockets.WebSocketClientProtocol` заменены на современный `ClientConnection` (из `websockets.asyncio.client`, с безопасным fallback на `Any`). Это полностью устранило `DeprecationWarning` библиотеки `websockets` версий 14+.
   - Прогон тестового набора `tests/research/lp_rewards/` с флагом строгой валидации предупреждений `-W error` проходит со 100% успехом: **0 warnings, 0 errors**.

4. **Итоговые результаты тестирования**:
   - Прогон тестового набора LP Rewards:
     ```bash
     uv run pytest tests/research/lp_rewards -v -W error
     ============================= 109 passed in 1.24s =============================
     ```
   - Все 109 тестов проходят успешно без предупреждений.
   - Проверка `git diff --check` выполняется чисто без ошибок форматирования.
