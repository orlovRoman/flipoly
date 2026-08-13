# AI Lab Phase 4 — безопасный executor/adapter-контур

Фаза 4 подключает исполнителя к очереди AI Lab, но не к боевой торговле. Она
делает один шаг воспроизводимым и наблюдаемым:

- worker атомарно вызывает `claim_next_step`;
- захватывает только `config_id`, `config_hash`, scope и payload;
- фиксирует claim и освобождает транзакцию **до** долгого обучения или backtest;
- вызывает только явно зарегистрированный offline-адаптер;
- сохраняет `TRAIN`, `OOT` или `POLYMARKET_OOT` через `record_result`;
- пишет в шаг краткий `summary`, `error_code` и `error_message`;
- при отсутствии адаптера или исключении создаёт аудируемый `FAILED`, а не падает
  всем worker-процессом.

## Контракт адаптера

`polyflip.ai_lab.executor` содержит:

- `StepContext` — неизменяемый набор входов шага. В него не передаются ORM
  объекты или открытая сессия;
- `AdapterResult` — статус, метрики, slices, trades, net PnL, drawdown, artifact
  id и provenance (`code_sha`, fingerprint и окна);
- `AdapterRegistry` — явный allow-list только для:
  `TRAIN_MODEL`, `RUN_OOT_BACKTEST`, `RUN_POLYMARKET_OOT`;
- `execute_next_step` и ограниченный `execute_steps`.

Пример регистрации во внешнем worker:

```python
registry = (
    AdapterRegistry()
    .register("TRAIN_MODEL", train_adapter)
    .register("RUN_OOT_BACKTEST", oot_adapter)
    .register("RUN_POLYMARKET_OOT", polymarket_oot_adapter)
)
outcomes = await execute_steps(session, run_id, registry, max_steps=1)
```

Адаптеры должны принимать `StepContext` и вернуть `AdapterResult` с правильным
evaluation_kind. Они могут использовать только безопасные train/backtest
сервисы и должны создавать immutable artifact; они не должны активировать модель,
менять RuntimeSettings или отправлять ордера.

## Что происходит при ошибке

- адаптер не зарегистрирован: `FAILED/ADAPTER_NOT_REGISTERED`;
- адаптер выбросил исключение: `FAILED/ADAPTER_EXECUTION_FAILED`;
- неправильный action/config/result kind: `FAILED/INVALID_STEP_INPUT` или
  контролируемая ошибка валидации;
- результат записывается только после завершения адаптера; при ошибке БД
  транзакция откатывается, чтобы worker не создавал ложный успех.

`execute_steps` имеет обязательный `max_steps`; это предохранитель от
бесконечного автономного цикла. Повторный запуск должен быть отдельной попыткой
с новым шагом/явной политикой retry, а не молчаливым перезапуском.

## Граница безопасности

В реестр невозможно добавить `ACTIVATE_MODEL`, `CHANGE_LIVE_POLICY`,
`EXECUTE_LIVE`, `PLACE_ORDER` или неизвестное действие. Следовательно, эта
фаза не может:

- включить модель в ACTIVE;
- изменить торговые пороги и RuntimeSettings;
- поставить, отменить или заменить ордер;
- обойти human approval и SHADOW.

Следующий этап — написать конкретные read-only/train/backtest адаптеры к
существующим сервисам и интегрировать worker с расписанием. Их нужно подключать
по одному, с тестовым набором и теми же provenance-полями.
