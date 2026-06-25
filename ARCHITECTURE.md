# PolyFlip — Архитектура

---

## 1. Обзор

PolyFlip — самодостаточный аналитический микросервис и мозг торговой стратегии.
Бот-исполнитель (будущий) вызывает `GET /signal` и размещает ордера. Вся логика принятия решений — здесь.

```
┌─────────────────────────────────────────────────────┐
│                  Дашборд (HTML)                      │
│          Chart.js + dropdown выбора актива           │
├─────────────────────────────────────────────────────┤
│               REST API (FastAPI)                     │
│ /signal │ /predict │ /stats │ /backtest │ /retrain  │
├─────────────────────────────────────────────────────┤
│          Trading Signal Layer (strategy/)            │
│   edge calc │ position sizing │ action decision      │
├─────────────────────────────────────────────────────┤
│        ML + Feature Engineering (6 фичей)            │
│    LogisticRegression │ FlipPredictor │ Trainer      │
├─────────────────────────────────────────────────────┤
│            Data Layer (PostgreSQL)                    │
│ market_snapshots │ live_markets │ model_registry     │
├─────────────────────────────────────────────────────┤
│     Scheduler (отдельный контейнер, APScheduler)     │
│ collect 1h │ live poll 1m │ retrain 24h │ alerting   │
└─────────────────────────────────────────────────────┘
```

---

## 2. Технологический стек

| Слой | Технология | Почему |
|---|---|---|
| API | FastAPI + Uvicorn | Async, автодокументация (Swagger), легковесный |
| Rate limiting | slowapi | Защита от перегрузки, настраиваемые лимиты per-endpoint |
| Данные | pandas + numpy | Стандарт для табличного анализа |
| ML | scikit-learn (LogisticRegression) | Простая, интерпретируемая, быстрый инференс |
| БД | PostgreSQL 16 + SQLAlchemy (asyncpg) | Надёжнее SQLite для накопительных данных, async |
| Миграции | Alembic | Версионирование схемы БД |
| Scheduler | APScheduler (AsyncIOScheduler) | Периодический сбор данных + переобучение |
| HTTP клиент | httpx (async) | Неблокирующие запросы к Polymarket API |
| Валидация | Pydantic v2 + pydantic-settings | Типизация на границах API + конфиг из env |
| Контейнер | Docker + docker-compose | Изоляция, воспроизводимый деплой |
| Тесты | pytest + pytest-asyncio | Unit + integration тесты |
| Логи | structlog | Структурированные JSON-логи |
| Дашборд | Jinja2 + Chart.js | Встроенные графики без отдельного фронтенда |
| Пакетный менеджер | Poetry | Надёжный lock-файл, стандартный pyproject.toml |

---

## 3. Структура проекта

```
polyflip/
├── README.md                      # Документация проекта
├── ARCHITECTURE.md                # Этот файл
├── DEVELOPMENT.md                 # Порядок разработки
├── API.md                         # Спецификация API
├── TESTING.md                     # Стратегия тестирования
├── docker-compose.yml             # Оркестрация контейнеров
├── Dockerfile                     # Сборка образа
├── pyproject.toml                 # Зависимости (Poetry)
├── poetry.lock                    # Закреплённые версии
├── .env.example                   # Шаблон переменных окружения
├── alembic.ini                    # Конфиг Alembic
│
├── alembic/                       # Миграции БД
│   ├── env.py
│   └── versions/
│       ├── 001_initial.py         # Начальная схема
│       └── 002_model_registry.py  # model_registry + predictions_log
│
├── polyflip/                      # Основной Python-пакет
│   ├── __init__.py
│   │
│   ├── config.py                  # Pydantic Settings + чтение из runtime_settings (БД)
│   ├── cli.py                     # CLI-команды (seed, retrain, и т.д.)
│   │
│   ├── api/                       # HTTP-слой
│   │   ├── __init__.py
│   │   ├── main.py                # FastAPI app, роуты, lifespan
│   │   ├── schemas.py             # Pydantic-модели запросов/ответов
│   │   ├── auth.py                # Dependency X-API-Key
│   │   ├── rate_limit.py          # slowapi rate limiting
│   │   └── dashboard.py           # Роут /dashboard, Jinja2 рендеринг
│   │
│   ├── collector/                 # Сбор данных с Polymarket
│   │   ├── __init__.py
│   │   ├── polymarket_client.py   # httpx-клиент к CLOB API
│   │   ├── gamma_client.py        # httpx-клиент к Gamma API (метаданные)
│   │   ├── market_resolver.py     # Определение final_outcome
│   │   ├── live_poller.py         # Поллинг live-цен активных 15m рынков
│   │   └── pipeline.py            # Оркестрация: найти → скачать → сохранить
│   │
│   ├── features/                  # Feature engineering
│   │   ├── __init__.py
│   │   └── builder.py             # build_feature_vector()
│   │
│   ├── strategy/                  # Торговая стратегия (ОТЛОЖЕНА — реализуется после аналитики)
│   │   ├── __init__.py
│   │   ├── edge.py                # Расчёт edge (EV) на основе flip_prob и market_price
│   │   └── sizing.py              # Position sizing (фикс. % банкролла)
│   │
│   ├── models/                    # ML
│   │   ├── __init__.py
│   │   ├── trainer.py             # Retrain логика (per-asset) + drift detection
│   │   ├── predictor.py           # Инференс + empirical fallback
│   │   └── store.py               # Сериализация в БД (BYTEA) + версионирование
│   │
│   ├── db/                        # Уровень данных
│   │   ├── __init__.py
│   │   ├── connection.py          # SQLAlchemy async engine + sessionmaker
│   │   └── models.py              # ORM-модели таблиц
│   │
│   ├── scheduler/                 # Фоновые задачи (отдельный контейнер)
│   │   ├── __init__.py
│   │   ├── jobs.py                # APScheduler: collect + retrain + live_poll + alert
│   │   └── alerting.py            # Алерты при сбоях коллектора
│   │
│   └── templates/                 # HTML-шаблоны дашборда
│       └── dashboard.html         # Jinja2 шаблон с Chart.js
│
└── tests/                         # Тесты
    ├── conftest.py                # Общие фикстуры
    ├── unit/
    │   ├── test_features.py       # Тесты feature engineering
    │   ├── test_predictor.py      # Тесты инференса
    │   ├── test_trainer.py        # Тесты обучения
    │   ├── test_drift.py          # Тесты drift detection
    │   ├── test_edge.py           # Тесты расчёта edge
    │   └── test_sizing.py         # Тесты position sizing
    └── integration/
        ├── test_api.py            # Тесты API-эндпоинтов
        ├── test_signal.py         # Тесты /signal эндпоинта
        └── test_backtest.py       # Тесты backtesting
```

---

## 4. Feature Engineering (ML-слой)

> **Принцип:** ML-модель оправдана только если даёт значимый прирост над эмпирической таблицей `AVG(flip) WHERE time_bin = X`.
> Для этого модель должна работать с несколькими информативными признаками.

### 4.1 Фичи модели

| # | Фича | Тип | Описание | Откуда берётся |
|---|---|---|---|---|
| 1 | `time_left_min` | float | Минуты до закрытия (0–15) | Вычисляется из `end_date - timestamp` |
| 2 | `mid_price` | float | Цена YES в момент наблюдения (0–1) | CLOB API orderbook / trade |
| 3 | `spread` | float | Разница bid/ask | CLOB API orderbook |
| 4 | `volume_last_5min` | float | Объём торгов за последние 5 минут | CLOB API trade history, агрегация |
| 5 | `price_velocity` | float | (price_now - price_5min_ago) / 5 | Вычисляется из истории цен |
| 6 | `hour_of_day` | int | Час (UTC) создания рынка | Из `timestamp` |

### 4.2 Почему каждая фича важна

- **`mid_price`** — чем увереннее рынок в исходе (цена близка к 0 или 1), тем flip неожиданнее и реже. Без этой фичи модель не различает «рынок показывает 0.52 за YES» и «рынок показывает 0.95 за YES».
- **`spread`** — широкий спред = низкая ликвидность = цена менее надёжна. Flip на illiquid рынке — это не то же самое, что flip на рынке с tight spread.
- **`volume_last_5min`** — резкий рост объёма торгов — один из лучших сигналов разворота. Informed traders входят в рынок перед flip.
- **`price_velocity`** — направление и скорость движения цены. Если цена быстро падает при time_left=3, это сильно отличается от стабильной цены.
- **`hour_of_day`** — рынки в разное время суток ведут себя по-разному (активность трейдеров, волатильность базового актива).

### 4.3 Feature vector

```python
# features/builder.py

def build_feature_vector(snapshot: dict) -> list[float]:
    """
    Входные данные: snapshot рынка с полями из market_snapshots.
    Возвращает: [time_left_min, mid_price, spread, volume_last_5min,
                 price_velocity, hour_of_day]
    """
    return [
        snapshot["time_to_expiry_min"],
        snapshot["mid_price"],
        snapshot["spread"],
        snapshot["volume_last_5min"],
        snapshot["price_velocity"],
        snapshot["hour_of_day"],
    ]
```

### 4.4 Confidence: теперь обоснованно

| Уровень | Условие | Что значит |
|---|---|---|
| `"high"` | ML-модель обучена, accuracy > baseline, drift не обнаружен | Модель работает и актуальна |
| `"medium"` | ML-модель есть, но drift обнаружен, ИЛИ empirical с ≥30 точками | Модель устарела или данных достаточно для эмпирики |
| `"low"` | Нет модели и <30 точек в бине | Мало данных, ответ ненадёжен |

**Baseline** = `AVG(flip_vs_final) WHERE time_bin = X` (эмпирическая таблица). ML-модель должна превосходить baseline на валидационной выборке, иначе используется empirical fallback.

---

## 5. Потоки данных

### 5.1 Сбор данных (Collector Pipeline)

```
Polymarket CLOB API          Gamma API
       │                         │
       ▼                         ▼
polymarket_client.py     gamma_client.py
       │                         │
       └──────────┬──────────────┘
                  ▼
           pipeline.py
                  │
                  ▼
       market_resolver.py
       (определяет final_outcome)
                  │
                  ▼
       features/builder.py
       (вычисляет spread, volume, velocity, hour)
                  │
                  ▼
        PostgreSQL: market_snapshots
```

**Что собираем:**
1. Найти завершённые 15m рынки для BTC/ETH через Gamma API (фильтр по тегам/категориям)
2. Для каждого рынка загрузить историю цен через CLOB API (orderbook snapshots / trade history)
3. Определить финальный исход (YES/NO) из данных resolution
4. Для каждой точки во времени вычислить все 6 фичей + `flip_vs_final`
5. Записать в `market_snapshots`

### 5.2 Обучение модели (Trainer)

```
PostgreSQL: market_snapshots
            │
            ▼
      builder.py
      (feature engineering)
            │
            ▼
    ┌────────────────────────────────────────────────┐
    │  X: [[time_left, mid_price, spread,            │
    │       volume_5min, price_velocity, hour]]       │
    │  y: [flip_vs_final]                            │
    └────────────────────────────────────────────────┘
            │
            ▼
    LogisticRegression.fit()
            │
            ├── accuracy > baseline? ──► Да ──► сохранить в model_registry
            │                           Нет ──► лог предупреждения, не заменять
            │
            ▼
    PostgreSQL: model_registry (model_blob BYTEA)
```

### 5.3 Инференс (Predictor)

```
POST /predict
  { asset: "BTC", time_left_min: 5.0,
    current_spread: 0.03, mid_price: 0.72 }
            │
            ▼
    FlipPredictor._models["BTC"]
            │
    ┌───────┴───────┐
    │ Модель есть   │
    │ и нет drift?  │
    ├── Да ─────────┤── model.predict_proba([[features...]])[0][1]
    │               │   confidence = "high"
    ├── Нет ────────┤── _empirical(): AVG(flip_vs_final)
    │               │   WHERE ROUND(time_to_expiry_min) = 5
    │               │   confidence = "medium" / "low"
    └───────────────┘
            │
            ▼
    Запись в predictions_log (для backtesting)
            │
            ▼
    PredictResponse
    { flip_prob: 0.3142, confidence: "high", ... }
```

### 5.4 Дашборд

```
GET /dashboard
       │
       ▼
  Jinja2 рендерит dashboard.html (3 вкладки)
       │
       ▼
  Браузер загружает Chart.js
       │
       ├── Вкладка «Аналитика»
       │   JS делает fetch("/api/dashboard-data/{asset}")
       │   → График P(flip) по time_left + таблица bins
       │
       ├── Вкладка «Статус»
       │   JS делает fetch("/health") + fetch("/assets")
       │   → Здоровье системы, модели, drift, last_collect
       │
       └── Вкладка «Настройки»
           JS делает GET /api/settings → заполняет формы
           При сохранении: PUT /api/settings → обновляет runtime_settings в БД
           Изменения вступают в силу немедленно (без перезапуска)
```

**Параметры, редактируемые через дашборд:**

| Группа | Параметры |
|---|---|
| Активы | ASSETS (добавить/убрать актив) |
| Scheduler | COLLECT_INTERVAL_MINUTES, LIVE_POLL_INTERVAL_SECONDS, RETRAIN_INTERVAL_HOURS |
| ML | MIN_SAMPLES_FOR_MODEL, DRIFT_THRESHOLD |
| Торговля (будущее) | MIN_EDGE, BET_FRACTION, TRADING_ENABLED (kill-switch) |
| Алерты | ALERT_WEBHOOK_URL, COLLECTOR_STALE_HOURS |
| Rate limit | RATE_LIMIT |

### 5.5 Model Drift Detection

```
При каждом retrain:
    │
    ▼
  Вычислить accuracy на свежих данных (последние 7 дней)
    │
    ▼
  Сравнить с accuracy при обучении
    │
    ├── |delta| < threshold (0.05) ──► OK
    │
    └── |delta| >= threshold ──► drift = True
                                 ├── лог предупреждения
                                 ├── confidence понижается до "medium"
                                 └── алерт (если настроен)
```

---

## 6. Схема базы данных

### Таблица `market_snapshots`

| Колонка | Тип | Описание |
|---|---|---|
| id | BIGINT PK | Автоинкремент |
| market_id | VARCHAR(128) | ID рынка на Polymarket |
| asset | VARCHAR(16) | "BTC" / "ETH" |
| timestamp | TIMESTAMPTZ | Время наблюдения |
| time_to_expiry_min | FLOAT | Минуты до закрытия рынка |
| mid_price | FLOAT | Средняя цена YES в момент наблюдения |
| spread | FLOAT | Разница bid/ask |
| volume_last_5min | FLOAT | Объём торгов за последние 5 минут |
| price_velocity | FLOAT | Скорость изменения цены (Δprice/Δtime) |
| hour_of_day | SMALLINT | Час UTC (0–23) |
| final_outcome | BOOLEAN | Финальный исход рынка (YES=true) |
| flip_vs_final | BOOLEAN | Цена предсказывала другой исход? |
| created_at | TIMESTAMPTZ | Когда записано |

**Индексы:**
- `idx_snapshots_asset_time` — (asset, time_to_expiry_min) — основной запрос
- `idx_snapshots_market` — (market_id) — дедупликация
- `idx_snapshots_created` — (created_at) — для drift detection (фильтр по дате)

### Таблица `model_registry`

Хранение моделей в БД вместо файловой системы. Устраняет зависимость от filesystem, упрощает откат к предыдущей версии.

| Колонка | Тип | Описание |
|---|---|---|
| id | SERIAL PK | Автоинкремент |
| asset | VARCHAR(16) | Актив |
| version | INT | Версия модели (автоинкремент per-asset) |
| trained_at | TIMESTAMPTZ | Дата обучения |
| data_points | INT | Кол-во точек в обучающей выборке |
| accuracy | FLOAT | Точность на валидации |
| baseline_accuracy | FLOAT | Точность empirical baseline при обучении |
| feature_names | JSONB | Список фичей, на которых обучена |
| model_blob | BYTEA | Сериализованная модель (pickle → bytes) |
| is_active | BOOLEAN DEFAULT false | Активная версия для инференса |
| created_at | TIMESTAMPTZ | Когда записано |

**Индексы:**
- `idx_model_asset_active` — (asset, is_active) WHERE is_active = true — быстрый поиск текущей модели
- `idx_model_asset_version` — (asset, version DESC) — откат к предыдущей версии

**Версионирование:** при каждом retrain создаётся новая запись с `version = max(version) + 1`. Старая модель помечается `is_active = false`. Откат = `UPDATE SET is_active = true WHERE version = N`.

### Таблица `predictions_log`

Лог предсказаний — для backtesting и drift detection.

| Колонка | Тип | Описание |
|---|---|---|
| id | BIGINT PK | Автоинкремент |
| asset | VARCHAR(16) | Актив |
| time_left_min | FLOAT | Запрошенное время |
| flip_prob | FLOAT | Предсказанная вероятность |
| confidence | VARCHAR(16) | high / medium / low |
| model_version | INT | Версия модели (или null для empirical) |
| actual_outcome | BOOLEAN NULL | Фактический flip (заполняется post-factum) |
| predicted_at | TIMESTAMPTZ | Когда предсказание сделано |

**Индексы:**
- `idx_predictions_asset_time` — (asset, predicted_at) — для backtesting

### Таблица `collector_status`

Отслеживание состояния коллектора для healthcheck и alerting.

| Колонка | Тип | Описание |
|---|---|---|
| id | SERIAL PK | Автоинкремент |
| run_at | TIMESTAMPTZ | Когда запускался коллектор |
| status | VARCHAR(16) | "success" / "partial" / "error" |
| markets_found | INT | Сколько рынков найдено |
| markets_saved | INT | Сколько успешно сохранено |
| error_message | TEXT NULL | Текст ошибки (если есть) |
| duration_sec | FLOAT | Длительность сбора |

### Таблица `runtime_settings`

Хранение настроек, изменяемых через дашборд. Имеет приоритет над env vars.

| Колонка | Тип | Описание |
|---|---|---|
| key | VARCHAR(64) PK | Имя параметра (напр. `ASSETS`, `MIN_EDGE`) |
| value | TEXT | Значение (хранится как строка, парсится при чтении) |
| updated_at | TIMESTAMPTZ | Когда изменено |
| updated_by | VARCHAR(64) | "dashboard" / "api" / "cli" |

**Пример записей:**

| key | value | updated_by |
|---|---|---|
| `ASSETS` | `BTC,ETH,SOL` | dashboard |
| `MIN_EDGE` | `0.07` | dashboard |
| `COLLECT_INTERVAL_MINUTES` | `30` | api |
| `TRADING_ENABLED` | `false` | dashboard |

### Таблица `live_markets`

Кеш текущего состояния активных 15-минутных рынков. Обновляется live poller каждую минуту.

| Колонка | Тип | Описание |
|---|---|---|
| id | SERIAL PK | Автоинкремент |
| market_id | VARCHAR(128) UNIQUE | ID рынка на Polymarket |
| condition_id | VARCHAR(128) | Condition ID для CLOB API |
| asset | VARCHAR(16) | "BTC" / "ETH" |
| question | TEXT | Текст вопроса рынка |
| end_date | TIMESTAMPTZ | Когда рынок закроется |
| current_price_yes | FLOAT | Текущая цена YES (mid) |
| current_price_no | FLOAT | Текущая цена NO (mid) |
| current_spread | FLOAT | Текущий спред bid/ask |
| volume_last_5min | FLOAT | Объём за последние 5 минут |
| price_velocity | FLOAT | Скорость изменения цены |
| updated_at | TIMESTAMPTZ | Когда обновлено |

**Индексы:**
- `idx_live_markets_asset` — (asset) — фильтр по активу
- `idx_live_markets_end` — (end_date) — удаление истёкших

**Жизненный цикл:** Записи создаются live poller, обновляются каждую минуту, удаляются после закрытия рынка (данные уходят в `market_snapshots`).

---

## 7. Конфигурация

Все параметры читаются из переменных окружения (с дефолтами):

### Общие

**Приоритет загрузки:** `runtime_settings (БД)` → `env vars (.env)` → `дефолты в коде`

При изменении настроек через дашборд значение записывается в `runtime_settings`. При следующем чтении config.py проверяет БД первым. Это позволяет менять параметры без перезапуска контейнеров.

| Переменная | Дефолт | Описание |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://polyflip:secret@db/polyflip` | Строка подключения к PostgreSQL |
| `POLYMARKET_CLOB_URL` | `https://clob.polymarket.com` | Базовый URL CLOB API |
| `POLYMARKET_GAMMA_URL` | `https://gamma-api.polymarket.com` | Базовый URL Gamma API |
| `ASSETS` | `BTC,ETH` | Список активов через запятую |
| `API_KEY` | *(обязательно)* | Ключ для аутентификации API |
| `LOG_LEVEL` | `INFO` | Уровень логирования |

### Scheduler и сбор данных

| Переменная | Дефолт | Описание |
|---|---|---|
| `COLLECT_INTERVAL_MINUTES` | `60` | Интервал сбора завершённых рынков (минуты) |
| `LIVE_POLL_INTERVAL_SECONDS` | `60` | Интервал поллинга live-цен активных рынков |
| `RETRAIN_INTERVAL_HOURS` | `24` | Интервал переобучения (часы) |
| `COLLECTOR_STALE_HOURS` | `2` | Через сколько часов без сбора — healthcheck = degraded |
| `ALERT_WEBHOOK_URL` | *(опционально)* | URL для отправки алертов (Telegram/Slack/Discord) |

### ML

| Переменная | Дефолт | Описание |
|---|---|---|
| `MIN_SAMPLES_FOR_MODEL` | `500` | Минимум точек для обучения модели |
| `DRIFT_THRESHOLD` | `0.05` | Порог для обнаружения model drift |

### Торговая стратегия (отложена — реализуется после аналитики)

| Переменная | Дефолт | Описание |
|---|---|---|
| `MIN_EDGE` | `0.05` | Минимальный edge для торгового сигнала (action ≠ hold) |
| `BET_FRACTION` | `0.02` | Доля банкролла на одну ставку (2%) |
| `RATE_LIMIT` | `60/minute` | Лимит запросов к API per client |

---

## 8. Безопасность

### API-аутентификация

- Все аналитические и торговые эндпоинты (`/predict`, `/signal`, `/stats/{asset}`, `/backtest`, `/retrain`) защищены заголовком `X-API-Key`
- Дашборд (`/dashboard`, `/api/dashboard-data/{asset}`) доступен без аутентификации — защита на уровне сети/VPN
- `/health`, `/assets` доступны без аутентификации (для Docker healthcheck и service discovery)

### Rate limiting

- `slowapi` с настраиваемым лимитом per-endpoint
- `/predict`, `/signal` — `RATE_LIMIT` (по умолчанию 60/min)
- `/retrain` — 1/hour (защита от случайного спама)
- Дашборд и health — без лимитов

### Хранение секретов

- API-ключ, пароль БД — только через переменные окружения
- `.env` файл добавлен в `.gitignore`
- Никаких хардкод-секретов в коде

---

## 9. Docker-архитектура

```
docker-compose.yml
├── api            (FastAPI + Uvicorn, порт 8001)
│   ├── command: uvicorn polyflip.api.main:app --host 0.0.0.0 --port 8001
│   ├── healthcheck: GET /health (degraded если collector stale > 2h)
│   └── depends_on: db
│
├── scheduler      (отдельный контейнер, тот же образ)
│   ├── command: python -m polyflip.scheduler.run
│   ├── задачи:
│   │   ├── collect_completed: каждый час — завершённые рынки в market_snapshots
│   │   ├── poll_live_markets: каждую минуту — цены активных рынков в live_markets
│   │   ├── retrain: каждые 24h — переобучение моделей
│   │   └── alerting: мониторинг сбоев
│   ├── healthcheck: проверяет, что scheduler loop жив
│   └── depends_on: db
│
└── db             (PostgreSQL 16 Alpine)
    ├── volumes: pgdata (persistent)
    └── порт: 5432 (только внутренний)
```

### Что изменилось vs. первая версия

| Было | Стало | Почему |
|---|---|---|
| Scheduler внутри API-контейнера | Отдельный контейнер `scheduler` | Падение scheduler не роняет API; независимый мониторинг |
| `collector` — одноразовый контейнер `run_once` | CLI-команда `python -m polyflip.cli seed` | Нет асимметрии между seed и регулярным сбором |
| `./models:/app/models` volume для pickle | Модели в БД (`model_registry.model_blob`) | Нет зависимости от filesystem; версионирование; откат |
| CORS middleware | Убран | Дашборд = same-origin (Jinja2), CORS не нужен |

### CLI-команды (вместо отдельного collector-контейнера)

```bash
# Начальная загрузка данных
docker-compose exec api python -m polyflip.cli seed --since 2026-01-01 --max-markets 500

# Ручное переобучение
docker-compose exec api python -m polyflip.cli retrain --asset BTC

# Откат модели
docker-compose exec api python -m polyflip.cli rollback --asset BTC --version 3
```

### Alerting

Scheduler отслеживает:
- **Коллектор не работает** — если `collector_status.run_at` старше `COLLECTOR_STALE_HOURS`, отправляет алерт
- **Polymarket API недоступен** — при 3+ последовательных ошибках
- **Model drift** — при обнаружении drift на retrain
- **Формат алерта** — POST на `ALERT_WEBHOOK_URL` (поддержка Telegram Bot API / Slack Incoming Webhooks / Discord Webhooks)

---

## 10. Trading Signal Layer (strategy/)

> **Принцип:** polyflip = мозг, бот = руки. Вся логика принятия торговых решений живёт в polyflip.
> Бот-исполнитель просто вызывает `GET /signal?asset=BTC` и размещает ордер.

### 10.1 Расчёт edge (`strategy/edge.py`)

```python
def calculate_edge(flip_prob: float, market_price_yes: float) -> dict:
    """
    flip_prob: P(flip) — вероятность, что рынок развернётся
    market_price_yes: текущая цена YES на рынке (0–1)

    Возвращает: { side, edge, ev_per_dollar }
    """
    p_yes_wins = 1 - flip_prob  # P(текущее направление верно)
    p_no_wins = flip_prob       # P(разворот)

    # EV покупки YES
    ev_yes = p_yes_wins * 1.0 - market_price_yes
    # EV покупки NO
    market_price_no = 1 - market_price_yes
    ev_no = p_no_wins * 1.0 - market_price_no

    if ev_yes > ev_no and ev_yes > 0:
        return {"side": "buy_yes", "edge": ev_yes}
    elif ev_no > 0:
        return {"side": "buy_no", "edge": ev_no}
    else:
        return {"side": "hold", "edge": 0.0}
```

### 10.2 Position sizing (`strategy/sizing.py`)

Фиксированный процент банкролла:

```python
def calculate_bet_size(bankroll: float, bet_fraction: float = 0.02) -> float:
    """
    bankroll: текущий банкролл в USD
    bet_fraction: доля банкролла на одну ставку (по умолчанию 2%)
    """
    return round(bankroll * bet_fraction, 2)
```

> **Примечание:** bankroll передаётся ботом в запросе `/signal`. polyflip не знает баланс кошелька напрямую.

### 10.3 Поток принятия решения

```
GET /signal?asset=BTC
        │
        ▼
  live_markets → текущая цена YES, spread, volume
        │
        ▼
  predictor.predict() → flip_prob, confidence
        │
        ▼
  edge.calculate_edge(flip_prob, market_price) → side, edge
        │
        ├── edge < MIN_EDGE (0.05) ──► action = "hold"
        │
        └── edge >= MIN_EDGE ──► action = side (buy_yes / buy_no)
                                  bet_size = bankroll * BET_FRACTION
        │
        ▼
  SignalResponse { action, edge, flip_prob, market_price, ... }
```

### 10.4 Live Price Polling (`collector/live_poller.py`)

Scheduler каждую минуту:
1. Запрашивает активные 15m рынки через Gamma API (фильтр: `active=true`, `tag=crypto`)
2. Для каждого рынка получает текущую цену с CLOB API (mid price из orderbook)
3. Обновляет `live_markets` в БД
4. Удаляет записи для истёкших рынков

Это обеспечивает, что `GET /signal` всегда имеет свежую цену (не старше 1 минуты) без необходимости бота самому ходить в Polymarket API.

Бот подключается к API-контейнеру по имени `polyflip` внутри Docker-сети.
