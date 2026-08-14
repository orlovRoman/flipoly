# AI Lab Phase 7 — supervised scheduler

Фаза 7 автоматизирует повторные вызовы bounded worker, но не превращает AI Lab в бесконтрольный daemon.

## Endpoint

```http
POST /api/ai-lab/runs/{run_id}/worker/lgbm/schedule
X-API-Key: ...
Content-Type: application/json

{
  "max_iterations": 3,
  "max_steps": 1,
  "interval_seconds": 2,
  "lease_ttl_seconds": 120
}
```

Ограничения:

- `max_iterations`: 1–20;
- `max_steps`: 1–10;
- `interval_seconds`: 0–60;
- `lease_ttl_seconds`: 30–3600 секунд.

Один вызов заканчивается, когда:

- очередь шагов пуста;
- адаптер вернул `FAILED` или `INSUFFICIENT_DATA`;
- достигнут лимит итераций;
- lease потерян.

## Защита от параллельного запуска

Для каждого run создаётся отдельная запись `ai_worker_leases` с уникальным `run_id`.
Lease блокируется на уровне БД, имеет heartbeat и срок истечения. Второй процесс
не начинает шаги, пока действующий lease не истёк. После завершения scheduler
lease удаляется.

Шаги по-прежнему claim-ятся через `FOR UPDATE SKIP LOCKED`, поэтому повторная
отправка запроса не выполняет один шаг дважды.

## Операционный цикл

1. Создать permission, run и immutable configs.
2. Создать план.
3. Запускать scheduler небольшими батчами через cron или внешний AI-agent.
4. Читать outcomes и `GET /api/ai-lab/runs/{id}`.
5. После окончания очереди вызвать `/evaluate`.
6. Проверить медианный Polymarket-OOT отчёт.
7. Перевести кандидата в SHADOW отдельным действием.

Scheduler не активирует модель, не меняет RuntimeSettings, не меняет live-policy и
не отправляет ордера. После слияния PR применить:

```bash
alembic upgrade head
```
