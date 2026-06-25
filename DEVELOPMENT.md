# PolyFlip — Порядок Разработки

---

## Фазы разработки

Проект разбит на фазы. Фазы 0–5 — аналитика (MVP). Фаза 6 — торговля (условная, только если аналитика покажет edge).

---

## Фаза 0: Инфраструктура (1-2 дня)

**Цель:** Рабочий скелет проекта, который можно запустить.

### Задачи

- [ ] Инициализировать Poetry-проект (`pyproject.toml`)
- [ ] Создать структуру директорий (пакет `polyflip/`, `tests/`, `alembic/`)
- [ ] Настроить `polyflip/config.py` (Pydantic Settings, чтение из `.env`)
- [ ] Написать `Dockerfile` (multi-stage: Poetry install → uvicorn)
- [ ] Написать `docker-compose.yml` (api + postgres + collector)
- [ ] Создать `.env.example` с дефолтными значениями
- [ ] Настроить SQLAlchemy async engine + sessionmaker (`db/connection.py`)
- [ ] Описать ORM-модели (`db/models.py`): `market_snapshots`, `model_metadata`
- [ ] Создать начальную миграцию Alembic (`001_initial.py`)
- [ ] Добавить `.gitignore`, `.dockerignore`

### Проверка

```bash
# Docker-compose поднимается без ошибок
docker-compose up -d
docker-compose logs api   # "Uvicorn running on 0.0.0.0:8001"

# Миграции применяются
docker-compose exec api alembic upgrade head

# Health-check работает
curl http://localhost:8001/health
# → {"status": "ok"}
```

---

## Фаза 1: API-скелет + Аутентификация (1 день)

**Цель:** Работающие эндпоинты с заглушками, защищённые API-ключом.

### Задачи

- [ ] Создать `api/schemas.py` — Pydantic-модели: `PredictRequest`, `PredictResponse`, `StatsResponse`
- [ ] Создать `api/auth.py` — middleware для проверки `X-API-Key`
- [ ] Создать `api/main.py` — FastAPI app с роутами:
  - `POST /predict` → заглушка (возвращает `flip_prob: 0.25`)
  - `GET /stats/{asset}` → заглушка (пустые bins)
  - `GET /health` → `{"status": "ok"}`
- [ ] Настроить structlog для JSON-логирования

### Проверка

```bash
# Без ключа — 401
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"asset": "BTC", "time_left_min": 5.0}'
# → 401 Unauthorized

# С ключом — 200
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key-here" \
  -d '{"asset": "BTC", "time_left_min": 5.0}'
# → {"asset": "BTC", "time_left_min": 5.0, "flip_prob": 0.25, ...}

# Swagger документация
# http://localhost:8001/docs
```

---

## Фаза 2: Сбор данных (2-3 дня)

**Цель:** Коллектор наполняет БД историческими данными завершённых 15m рынков.

### Задачи

- [ ] Создать `collector/polymarket_client.py`:
  - Обёртка над httpx для CLOB API
  - Методы: `get_markets()`, `get_market_trades()`, `get_orderbook_snapshot()`
  - Rate limiting, retry с экспоненциальным backoff
- [ ] Создать `collector/gamma_client.py`:
  - Поиск 15m рынков по тегам/категориям
  - Получение метаданных: `end_date`, `outcome`, `question`
- [ ] Создать `collector/market_resolver.py`:
  - Определение `final_outcome` (YES/NO) из данных resolution
  - Определение `flip_vs_final` для каждого snapshot
- [ ] Создать `collector/pipeline.py`:
  - Оркестрация: найти завершённые рынки → скачать историю → обработать → сохранить
  - Дедупликация по `market_id` (не скачивать повторно)
  - Логирование прогресса: сколько рынков найдено / обработано / сохранено
- [ ] Создать `collector/run_once.py`:
  - CLI-скрипт для начальной загрузки (`python -m polyflip.collector.run_once`)
  - Параметры: `--since` (дата начала), `--max-markets` (лимит)
- [ ] Создать `features/builder.py`:
  - `build_market_timeseries(market_data)` → список snapshot-записей
  - Для каждого trade/snapshot: вычислить `time_to_expiry_min`, `mid_price`, `flip_vs_final`

### Проверка

```bash
# Начальная загрузка
docker-compose run collector python -m polyflip.collector.run_once --max-markets 50

# Проверить, что данные в БД
docker-compose exec db psql -U polyflip -c "
  SELECT asset, COUNT(*), MIN(time_to_expiry_min), MAX(time_to_expiry_min)
  FROM market_snapshots
  GROUP BY asset;
"
# Ожидаем: BTC и ETH с сотнями/тысячами точек

# Проверить распределение
docker-compose exec db psql -U polyflip -c "
  SELECT asset,
         ROUND(time_to_expiry_min::numeric) as t_min,
         COUNT(*) as n,
         AVG(flip_vs_final::int) as flip_rate
  FROM market_snapshots
  GROUP BY asset, ROUND(time_to_expiry_min::numeric)
  ORDER BY asset, t_min;
"
```

---

## Фаза 3: ML-модели + Scheduler (1-2 дня)

**Цель:** Модели обучены, инференс работает, scheduler запускает переобучение.

### Задачи

- [ ] Создать `models/trainer.py`:
  - `train_model(asset)` — выгрузить данные из БД, обучить LogisticRegression
  - Фичи: `[time_left_min]` (v1), расширяемо до `[time_left, spread, ...]` (v2)
  - Валидация: train/test split, логирование accuracy
  - Порог: не обучать модель, если точек < `MIN_SAMPLES_FOR_MODEL`
  - `retrain_all()` — обучить для всех активов
- [ ] Создать `models/store.py`:
  - `save_model(asset, model)` → сериализация в BYTEA, запись в `model_registry`
  - `load_model(asset)` → загрузка из `model_registry` WHERE `is_active = true`
  - Версионирование + откат (новая версия = новая запись, старая помечается `is_active = false`)
- [ ] Создать `models/predictor.py`:
  - `FlipPredictor` — класс с кешированными моделями
  - `predict(asset, time_left_min)` → ML-предсказание или empirical fallback
  - `get_stats(asset)` → статистика по бинам для дашборда
- [ ] Создать `scheduler/jobs.py`:
  - `start_scheduler()` — APScheduler с двумя задачами:
    - `collect_completed_markets` каждые N минут
    - `retrain_all` каждые M часов
  - Graceful startup/shutdown

### Проверка

```bash
# Обучить модели вручную
docker-compose exec api python -c "
import asyncio
from polyflip.models.trainer import retrain_all
asyncio.run(retrain_all())
"

# Проверить, что модели сохранены в model_registry
docker-compose exec db psql -U polyflip -c "
  SELECT asset, version, is_active, trained_at, accuracy
  FROM model_registry ORDER BY asset, version;
"
# → BTC v1 active, ETH v1 active

# Проверить предсказание через API
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key-here" \
  -d '{"asset": "BTC", "time_left_min": 5.0}'
# → {"flip_prob": 0.31, "confidence": "high", ...}

# Проверить статистику
curl http://localhost:8001/stats/BTC \
  -H "X-API-Key: your-key-here"
# → {"asset": "BTC", "bins": [...], "total_markets": 142, ...}
```

---

## Фаза 4: Дашборд с настройками (1-2 дня)

**Цель:** Встроенная HTML-страница с аналитикой, статусом и управлением настройками.

### Задачи

- [ ] Создать `templates/dashboard.html`:
  - Jinja2 шаблон, подключение Chart.js через CDN
  - **Вкладка «Аналитика»:**
    - Dropdown для выбора актива (BTC, ETH, ...)
    - Line chart: ось X = `time_left_min` (15→0), ось Y = `P(flip)` (0→1)
    - Таблица под графиком: `time_min`, `flip_prob`, `n_points`, `confidence`
  - **Вкладка «Статус»:**
    - Здоровье системы (health), загруженные модели, drift-статус
    - Последний сбор данных, кол-во рынков
  - **Вкладка «Настройки»:**
    - Формы для всех параметров (сгруппированы: Активы, Scheduler, ML, Торговля, Алерты)
    - Кнопка «Сохранить» → PUT /api/settings
    - Индикация `source` (откуда взято значение: db / env / default)
  - Auto-refresh каждые 5 минут (аналитика и статус)
- [ ] Создать `api/dashboard.py`:
  - `GET /dashboard` → Jinja2 render `dashboard.html`
  - `GET /api/dashboard-data/{asset}` → JSON для графика
  - `GET /api/settings` → текущие настройки (value + source + type)
  - `PUT /api/settings` → сохранение в `runtime_settings`
- [ ] Подключить `Jinja2Templates` в `api/main.py`
- [ ] Стилизация: тёмная тема, адаптивная вёрстка

### Проверка

```bash
# Открыть в браузере
# http://localhost:8001/dashboard

# Проверить:
# ✅ Вкладка «Аналитика»: dropdown с активами, график P(flip), таблица
# ✅ Вкладка «Статус»: модели, drift, last_collect
# ✅ Вкладка «Настройки»: формы, сохранение, индикация source
# ✅ Изменение настроек вступает в силу без перезапуска
# ✅ Страница корректно отображается на мобильном
```

---

## Фаза 5: Тестирование и Hardening (1-2 дня)

**Цель:** Покрытие тестами, обработка edge-cases, документация.

### Задачи

- [ ] Unit-тесты:
  - `tests/unit/test_features.py` — `build_market_timeseries()` на фиктивных данных
  - `tests/unit/test_predictor.py` — инференс с мок-моделью + fallback
  - `tests/unit/test_trainer.py` — обучение на минимальных данных
- [ ] Integration-тесты:
  - `tests/integration/test_api.py` — тесты API-эндпоинтов через `TestClient`
  - Проверка аутентификации (401 без ключа, 200 с ключом)
  - Проверка валидации (невалидные параметры → 422)
- [ ] Edge-cases:
  - Пустая БД (нет данных) → корректный fallback
  - Неизвестный актив → 404 или дефолтный ответ
  - Модель ещё не обучена → empirical fallback
  - Polymarket API недоступен → retry + graceful degradation
- [ ] Документация:
  - Swagger/OpenAPI автогенерация (/docs)
  - Финальное обновление README.md
  - Комментарии в ключевых модулях

### Проверка

```bash
# Запустить все тесты
poetry run pytest tests/ -v

# Запустить только unit-тесты (быстро)
poetry run pytest tests/unit/ -v

# Проверить покрытие
poetry run pytest tests/ --cov=polyflip --cov-report=term-missing
```

---

## Итого: ориентировочные сроки

### Аналитика (MVP)

| Фаза | Ориентир |
|---|---|
| 0. Инфраструктура | 1-2 дня |
| 1. API-скелет | 1 день |
| 2. Сбор данных | 2-3 дня |
| 3. ML + Scheduler | 1-2 дня |
| 4. Дашборд + Настройки | 1-2 дня |
| 5. Тесты + Hardening | 1-2 дня |
| **Итого аналитика** | **~7-12 дней** |

### Торговля (условная)

| Фаза | Ориентир | Условие |
|---|---|---|
| 6. Торговый слой | 2-3 дня | Аналитика показала edge |

---

## Зависимости между фазами

```
Фаза 0 (Инфра)
    │
    ▼
Фаза 1 (API-скелет)
    │
    ├──────────────────┐
    ▼                  ▼
Фаза 2 (Данные)    Фаза 4 (Дашборд)*
    │
    ▼
Фаза 3 (ML)
    │
    ▼
Фаза 5 (Тесты)
    │
    ▼
АНАЛИЗ ДАННЫХ: есть ли edge?
    │
    ├── Да ──► Фаза 6 (Торговля)
    └── Нет ─► Рынок эффективен, торговля не нужна

* Дашборд можно начать параллельно с Фазой 2,
  но полноценно протестировать — только после Фазы 3.
```
