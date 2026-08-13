# AI Lab Phase 3 — безопасная оркестрация экспериментов

Фаза 3 добавляет управляемый контур выполнения экспериментов. Он сохраняет шаги
и результаты в БД, строит отчёт по реальному Polymarket-OOT и переводит
рекомендованный артефакт в пассивный SHADOW. Контур не активирует модель, не
изменяет RuntimeSettings и не отправляет ордера.

## Жизненный цикл

1. Создать permission-профиль с разрешениями `CREATE_EXPERIMENT`,
   `TRAIN_MODEL`, `RUN_OOT_BACKTEST`, `RUN_POLYMARKET_OOT` и
   `PROMOTE_TO_SHADOW`.
2. Создать AI run с этим конкретным `permission_id` и бюджетом.
3. Создать или выбрать immutable `experiment_configs`.
4. Вызвать `POST /api/ai-lab/runs/{id}/plan` с `config_ids`.
   Для каждой конфигурации создаются шаги TRAIN_MODEL, RUN_OOT_BACKTEST и
   RUN_POLYMARKET_OOT.
5. Worker/агент вызывает `POST /runs/{id}/steps/claim`, выполняет ровно
   возвращённый безопасный адаптер и записывает результат через
   `POST /runs/{id}/results`.
6. После результатов вызвать `POST /runs/{id}/evaluate`. Отчёт использует
   медианные PnL, число сделок и просадку по Polymarket-OOT. AUC/ECE остаются
   диагностикой и не могут заменить PnL-сэмпл.
7. Только если есть кандидат с достаточным Polymarket-OOT-сэмплом, вызвать
   `POST /runs/{id}/shadow`. Проверяется, что артефакт прикреплён к
   рекомендованной конфигурации. Создаётся `ai_shadow_assignments` со
   статусом PENDING.

## API

- `POST /api/ai-lab/runs/{id}/plan`
  ```json
  {"config_ids": [101, 102, 103]}
  ```
- `POST /api/ai-lab/runs/{id}/steps/claim`
- `POST /api/ai-lab/runs/{id}/results`
  ```json
  {
    "config_id": 101,
    "evaluation_kind": "POLYMARKET_OOT",
    "status": "SUCCEEDED",
    "metrics": {"auc": 0.74, "ece": 0.02, "brier": 0.19},
    "trade_count": 37,
    "net_pnl": 4.12,
    "max_drawdown": -1.3,
    "artifact_id": 501
  }
  ```
- `POST /api/ai-lab/runs/{id}/evaluate`
- `POST /api/ai-lab/runs/{id}/shadow`

Все запросы проходят через API-key dependency и immutable permission snapshot
run. Терминальные состояния нельзя использовать для продолжения работы.
`ACTIVE` отсутствует в автономном графе переходов; активация остаётся
отдельным human-approval шагом будущей фазы.

## Исполнительные адаптеры

Оркестратор намеренно не импортирует Trainer, BacktestRunner или Polymarket
gateway. Worker следующей итерации должен:

- взять `input_payload.config_id` из claim;
- запустить существующий безопасный train/backtest адаптер без изменения
  активных моделей;
- сохранить code SHA, dataset fingerprint, окна, slice-метрики и artifact id
  вместе с результатом;
- повторять шаг только через новую попытку с явной записью ошибки.

Такой разрыв между планировщиком и исполнителем позволяет подключать Codex или
другой AI-агент, не давая ему прямого доступа к LIVE-контуру.
