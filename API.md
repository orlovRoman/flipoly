# PolyFlip — Спецификация REST API

---

## Базовый URL

```
http://localhost:8001
```

## Аутентификация

Все эндпоинты (кроме `/health`, `/assets`, `/dashboard`, `/api/dashboard-data/*`, `/api/settings`) требуют заголовок:

```
X-API-Key: <значение из переменной окружения API_KEY>
```

При отсутствии или неверном ключе — `401 Unauthorized`.

## Rate Limiting

Все аутентифицированные эндпоинты защищены через `slowapi`:

| Эндпоинт | Лимит | Окно |
|---|---|---|
| `POST /predict` | 60 | минута |
| `POST /api/analytics/train/*` | 1 | час |
| `GET /stats/*`, `/api/backtest/*` | 30 | минута |

При превышении — `429 Too Many Requests` с заголовком `Retry-After`.

---

## Эндпоинты

### POST /predict

Основной эндпоинт — предсказание вероятности flip.

**Запрос:**

```json
{
  "asset": "BTC",
  "time_left_min": 5.0,
  "mid_price": 0.72,
  "current_spread": 0.03
}
```

| Поле | Тип | Обязательно | Описание |
|---|---|---|---|
| `asset` | string | ✅ | Код актива: `"BTC"`, `"ETH"` |
| `time_left_min` | float | ✅ | Минуты до закрытия рынка (0–15) |
| `mid_price` | float \| null | ❌ | Текущая цена YES (0–1). Если передана, используется как фича модели |
| `current_spread` | float \| null | ❌ | Текущий спред bid/ask. Если передан, используется как фича модели |

> **Примечание:** `mid_price` и `current_spread` — опциональные фичи реального времени.
> Если не переданы, модель использует средние значения из исторических данных для данного бина.
> `volume_last_5min`, `price_velocity`, `hour_of_day` вычисляются сервисом автоматически
> (volume и velocity — из последних данных коллектора, hour — из текущего времени).

**Ответ (200):**

```json
{
  "asset": "BTC",
  "time_left_min": 5.0,
  "flip_prob": 0.3142,
  "confidence": "high",
  "model_version": 7,
  "model_trained_at": "2026-06-25T12:00:00Z",
  "features_used": ["time_left_min", "mid_price", "spread", "volume_last_5min", "price_velocity", "hour_of_day"],
  "data_points": 1247,
  "drift_detected": false
}
```

| Поле | Тип | Описание |
|---|---|---|
| `asset` | string | Запрошенный актив |
| `time_left_min` | float | Запрошенное время |
| `flip_prob` | float | P(flip), от 0 до 1 |
| `confidence` | string | `"high"` — ML без drift, `"medium"` — ML с drift или empirical ≥30, `"low"` — мало данных |
| `model_version` | int \| null | Номер версии модели (null если empirical fallback) |
| `model_trained_at` | string \| null | ISO дата обучения модели |
| `features_used` | array[string] | Список фичей, использованных для предсказания |
| `data_points` | int | Количество исторических точек для данного бина |
| `drift_detected` | bool | Обнаружен ли model drift |

**Ошибки:**

| Код | Когда |
|---|---|
| 401 | Отсутствует или неверный `X-API-Key` |
| 422 | Невалидные параметры (time_left_min < 0 или > 15, пустой asset) |
| 429 | Превышен rate limit |

---

### GET /stats/{asset}

Статистика по бинам — используется дашбордом и для анализа.

**Параметры:**

| Параметр | Тип | Описание |
|---|---|---|
| `asset` (path) | string | Код актива: `"BTC"`, `"ETH"` |

**Ответ (200):**

```json
{
  "asset": "BTC",
  "bins": [
    {"time_min": 15, "flip_prob": 0.45, "n_points": 312, "avg_mid_price": 0.55},
    {"time_min": 14, "flip_prob": 0.42, "n_points": 298, "avg_mid_price": 0.57},
    ...
    {"time_min": 0, "flip_prob": 0.03, "n_points": 195, "avg_mid_price": 0.89}
  ],
  "model_trained_at": "2026-06-25T12:00:00Z",
  "model_version": 7,
  "model_accuracy": 0.73,
  "baseline_accuracy": 0.61,
  "drift_detected": false,
  "total_markets": 1523,
  "last_collect": "2026-06-25T11:00:00Z"
}
```

| Поле | Тип | Описание |
|---|---|---|
| `asset` | string | Запрошенный актив |
| `bins` | array | Массив бинов по минутам (16 элементов: 0–15) |
| `bins[].time_min` | int | Минута до закрытия |
| `bins[].flip_prob` | float | Эмпирическая P(flip) для этого бина |
| `bins[].n_points` | int | Количество наблюдений в бине |
| `bins[].avg_mid_price` | float | Средняя цена YES в этом бине |
| `model_trained_at` | string \| null | ISO дата обучения модели (null если не обучена) |
| `model_version` | int \| null | Версия текущей активной модели |
| `model_accuracy` | float \| null | Accuracy модели на валидации |
| `baseline_accuracy` | float \| null | Accuracy empirical baseline |
| `drift_detected` | bool | Обнаружен ли model drift |
| `total_markets` | int | Общее кол-во обработанных рынков для актива |
| `last_collect` | string \| null | Когда последний раз собирались данные |

**Ошибки:**

| Код | Когда |
|---|---|
| 401 | Отсутствует или неверный `X-API-Key` |
| 404 | Неизвестный актив (нет в конфиге `ASSETS`) |

---

### GET /assets

Список поддерживаемых активов — доступен без аутентификации.

**Ответ (200):**

```json
{
  "assets": [
    {
      "code": "BTC",
      "model_version": 7,
      "model_trained_at": "2026-06-25T12:00:00Z",
      "total_snapshots": 15230,
      "last_collect": "2026-06-25T11:00:00Z"
    },
    {
      "code": "ETH",
      "model_version": 4,
      "model_trained_at": "2026-06-25T12:00:00Z",
      "total_snapshots": 12150,
      "last_collect": "2026-06-25T11:00:00Z"
    }
  ]
}
```

Клиент больше не должен знать список активов заранее — он запрашивает его через API.

---

### POST /api/analytics/train/{asset}

Ручное переобучение модели — не нужно ждать 24-часовой цикл.

**Параметры:**

| Параметр | Тип | Описание |
|---|---|---|
| `asset` (path) | string | Актив для переобучения |

**Ответ (202 Accepted):**

```json
{
  "status": "accepted",
  "message": "Retrain for BTC queued",
  "estimated_duration_sec": 30
}
```

Переобучение выполняется асинхронно. Результат можно проверить через `GET /stats/{asset}` (поле `model_trained_at` обновится).

**Ошибки:**

| Код | Когда |
|---|---|
| 401 | Отсутствует или неверный `X-API-Key` |
| 429 | Повторный retrain менее чем через час |

---

### GET /api/analytics/summary

Сводка по аналитике моделей.

### GET /api/analytics/models

Список моделей и их статус.

---

### GET /api/slippage

История проскальзываний торгов.

---

## Backtesting API

### GET /backtest
Отдает HTML UI страницу для бэктестинга.

### POST /api/backtest/submit
Запуск нового бэктест-прогона.

### GET /api/backtest/status/{run_id}
Статус текущего прогона бэктеста.

### GET /api/backtest/result/{run_id}
Результаты и метрики завершенного бэктеста.

### GET /api/backtest/history
История запусков бэктестов.

### GET /api/backtest/models
Доступные модели для бэктеста.

### GET /api/backtest/dataset_stats
Статистика по датасету.
---

### GET /health

Healthcheck — доступен без аутентификации. Возвращает `degraded` если коллектор не работал дольше `COLLECTOR_STALE_HOURS`.

**Ответ (200):**

```json
{
  "status": "ok",
  "db_connected": true,
  "models_loaded": ["BTC", "ETH"],
  "last_collect": "2026-06-25T11:00:00Z",
  "collector_stale": false,
  "scheduler_alive": true
}
```

**Ответ (200, degraded):**

```json
{
  "status": "degraded",
  "db_connected": true,
  "models_loaded": ["BTC"],
  "last_collect": "2026-06-25T05:00:00Z",
  "collector_stale": true,
  "scheduler_alive": false
}
```

> `status: "degraded"` — сервис отвечает, но данные устарели. Предсказания всё ещё возвращаются, но confidence может быть ниже.

---

### GET /dashboard

HTML-страница с дашбордом — доступна без аутентификации.

Три вкладки:
1. **Аналитика** — графики P(flip), выбор актива, таблица bins
2. **Статус** — здоровье системы, модели, drift
3. **Настройки** — редактирование всех параметров в реальном времени

Возвращает `text/html` — Jinja2-шаблон с Chart.js. Same-origin, CORS не нужен.

---

### GET /api/dashboard-data/{asset}

JSON-данные для графика дашборда — доступен без аутентификации (дашборд делает fetch к тому же origin).

**Ответ (200):**

```json
{
  "asset": "BTC",
  "labels": [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
  "flip_probs": [0.45, 0.42, 0.39, 0.36, 0.33, 0.30, 0.27, 0.24, 0.21, 0.18, 0.15, 0.12, 0.09, 0.06, 0.03, 0.01],
  "n_points": [312, 298, 305, 287, 301, 295, 310, 303, 289, 296, 308, 291, 297, 284, 289, 195],
  "model_trained_at": "2026-06-25T12:00:00Z",
  "model_accuracy": 0.73,
  "drift_detected": false,
  "total_markets": 1523
}
```

---

### GET /api/settings

Текущие настройки сервиса — доступен без аутентификации (дашборд использует для заполнения форм).

**Ответ (200):**

```json
{
  "settings": {
    "ASSETS": {"value": "BTC,ETH", "source": "env", "type": "string"},
    "COLLECT_INTERVAL_MINUTES": {"value": "60", "source": "default", "type": "int"},
    "LIVE_POLL_INTERVAL_SECONDS": {"value": "60", "source": "default", "type": "int"},
    "RETRAIN_INTERVAL_HOURS": {"value": "24", "source": "default", "type": "int"},
    "MIN_SAMPLES_FOR_MODEL": {"value": "500", "source": "default", "type": "int"},
    "DRIFT_THRESHOLD": {"value": "0.05", "source": "default", "type": "float"},
    "MIN_EDGE": {"value": "0.05", "source": "db", "type": "float"},
    "MAX_EDGE": {"value": "2.0", "source": "db", "type": "float"},
    "TRADE_BET_SIZE_USDC": {"value": "10.0", "source": "db", "type": "float"},
    "MAX_BET_SIZE_USDC": {"value": "50.0", "source": "db", "type": "float"},
    "LIQUIDITY_FRACTION": {"value": "0.1", "source": "db", "type": "float"},
    "BYPASS_BET_SIZE_CHECK": {"value": "false", "source": "db", "type": "bool"},
    "BET_SIZING_MODE": {"value": "linear", "source": "db", "type": "string"},
    "AUTO_DEAD_ZONE": {"value": "true", "source": "db", "type": "bool"},
    "TRADING_MODE": {"value": "PAPER", "source": "db", "type": "string"},
    "MAX_PRICE_DRIFT": {"value": "0.05", "source": "db", "type": "float"},
    "STOP_LOSS_ENABLED": {"value": "false", "source": "db", "type": "bool"},
    "STOP_LOSS_PCT_FAVORITE": {"value": "40.0", "source": "db", "type": "float"},
    "STOP_LOSS_PCT_OUTSIDER": {"value": "60.0", "source": "db", "type": "float"},
    "STOP_LOSS_CHECK_SEC": {"value": "30", "source": "db", "type": "int"},
    "TRADE_FLIP_THRESHOLD": {"value": "0.85", "source": "db", "type": "float"},
    "TRADE_FLIP_THRESHOLD_BTC": {"value": "0.85", "source": "db", "type": "float"},
    "TRADING_ENABLED": {"value": "false", "source": "default", "type": "bool"},
    "ALERT_WEBHOOK_URL": {"value": "", "source": "default", "type": "string"},
    "COLLECTOR_STALE_HOURS": {"value": "2", "source": "default", "type": "int"},
    "RATE_LIMIT": {"value": "60/minute", "source": "default", "type": "string"}
  }
}
```

| Поле | Тип | Описание |
|---|---|---|
| `settings` | object | Словарь параметров |
| `settings[key].value` | string | Текущее значение (всегда строка) |
| `settings[key].source` | string | Откуда взято: `"db"` (runtime_settings), `"env"`, `"default"` |
| `settings[key].type` | string | Тип для валидации на фронте: `"string"`, `"int"`, `"float"`, `"bool"` |

---

### PUT /api/settings

Обновление настроек — доступен без аутентификации (дашборд). Изменения вступают в силу немедленно.

**Запрос:**

```json
{
  "settings": {
    "ASSETS": "BTC,ETH,SOL",
    "MIN_EDGE": "0.07"
  }
}
```

Передаются только изменяемые параметры. Остальные остаются без изменений.

**Ответ (200):**

```json
{
  "updated": ["ASSETS", "MIN_EDGE"],
  "message": "Settings updated successfully"
}
```

**Ошибки:**

| Код | Когда |
|---|---|
| 422 | Невалидное значение (напр. `MIN_EDGE: "abc"`) |
| 400 | Неизвестный ключ настройки |

---

## Коды ответов (сводка)

| Код | Описание |
|---|---|
| 200 | Успешный ответ |
| 202 | Accepted — задача поставлена в очередь (retrain) |
| 401 | Unauthorized — неверный или отсутствующий API-ключ |
| 404 | Not Found — неизвестный актив |
| 422 | Validation Error — невалидные параметры |
| 429 | Too Many Requests — превышен rate limit |
| 500 | Internal Server Error — неожиданная ошибка |

---

## Пример вызова из Python (httpx)

```python
import httpx

async def get_flip_prob(
    asset: str,
    time_left: float,
    mid_price: float | None = None,
    spread: float | None = None,
) -> dict:
    async with httpx.AsyncClient(base_url="http://polyflip:8001") as client:
        payload = {"asset": asset, "time_left_min": time_left}
        if mid_price is not None:
            payload["mid_price"] = mid_price
        if spread is not None:
            payload["current_spread"] = spread

        resp = await client.post(
            "/predict",
            json=payload,
            headers={"X-API-Key": "your-secret-key"},
            timeout=2.0,
        )
        resp.raise_for_status()
        return resp.json()

# Использование:
# result = await get_flip_prob("BTC", 5.0, mid_price=0.72, spread=0.03)
# print(result["flip_prob"], result["confidence"])
```

## Пример вызова из curl

```bash
# Предсказание (с опциональными фичами)
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{"asset": "BTC", "time_left_min": 5.0, "mid_price": 0.72, "current_spread": 0.03}'

# Список активов (без ключа)
curl http://localhost:8001/assets

# Статистика
curl http://localhost:8001/stats/BTC \
  -H "X-API-Key: your-secret-key"

# Список истории бэктестов
curl http://localhost:8001/api/backtest/history \
  -H "X-API-Key: your-secret-key"

# Ручное переобучение
curl -X POST http://localhost:8001/api/analytics/train/BTC \
  -H "X-API-Key: your-secret-key"

# Health
curl http://localhost:8001/health
```
