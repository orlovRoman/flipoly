# PolyFlip — Стратегия Тестирования и Верификации

---

## 1. Обзор

Тестирование PolyFlip охватывает три уровня:

| Уровень | Что тестируем | Инструменты | Когда запускать |
|---|---|---|---|
| Unit | Чистые функции, логика | pytest | При каждом изменении кода |
| Integration | API, БД, взаимодействие слоёв | pytest-asyncio + TestClient | Перед мержем / деплоем |
| E2E / Smoke | Весь сервис в Docker | curl + скрипты | После деплоя |

---

## 2. Unit-тесты

### 2.1 `tests/unit/test_features.py`

Тестирование `features/builder.py` — преобразование сырых рыночных данных в snapshot-записи.

```python
# Что тестируем:
# 1. build_market_timeseries() корректно вычисляет time_to_expiry_min
# 2. flip_vs_final корректно определяется (price > 0.5 → YES, outcome = NO → flip = True)
# 3. Граничные случаи: price = 0.5 (50/50), пустые данные, один trade
# 4. Порядок результатов: от time_left=15 к time_left=0

def test_basic_timeseries():
    """Market с 3 точками → 3 snapshot-записи с корректными полями."""

def test_flip_detection():
    """Price=0.72 + outcome=NO → flip_vs_final=True."""

def test_no_flip():
    """Price=0.72 + outcome=YES → flip_vs_final=False."""

def test_edge_price_50():
    """Price=0.5 → flip_vs_final зависит от outcome (≤0.5 = NO-зона)."""

def test_empty_trades():
    """Пустой список trades → пустой результат, без ошибок."""
```

### 2.2 `tests/unit/test_predictor.py`

Тестирование `models/predictor.py` — инференс и fallback.

```python
# Что тестируем:
# 1. Если модель загружена → используется predict_proba, confidence="high"
# 2. Если модели нет → empirical fallback из БД
# 3. Если данных мало (<10 точек) → fallback 0.25, confidence="low"
# 4. Корректность округления flip_prob до 4 знаков

def test_predict_with_model(mock_model):
    """Мок LogisticRegression → predict_proba возвращает 0.35 → flip_prob=0.35."""

def test_predict_without_model(mock_db_session):
    """Модели нет, в БД 50 точек со средним 0.28 → flip_prob=0.28, confidence="medium"."""

def test_predict_no_data():
    """Модели нет, в БД 0 точек → flip_prob=0.25, confidence="low"."""

def test_get_stats(mock_db_session):
    """get_stats возвращает 16 бинов (0-15 минут) с корректными данными."""
```

### 2.3 `tests/unit/test_trainer.py`

Тестирование `models/trainer.py` — обучение модели.

```python
# Что тестируем:
# 1. train_model() на достаточных данных → создаёт запись в model_registry (BYTEA)
# 2. Недостаточно данных (<MIN_SAMPLES) → модель не создаётся, лог предупреждения
# 3. Валидация: train/test split не смешивается
# 4. Версионирование: повторный retrain создаёт version+1, старая модель is_active=false

def test_train_sufficient_data(mock_db_with_data):
    """•1000 точек → модель обучена, записана в model_registry, is_active=True."""

def test_train_insufficient_data(mock_db_empty):
    """•10 точек → модель НЕ создана, логирование предупреждения."""

def test_retrain_all(mock_db_with_data):
    """•retrain_all() обучает модели для всех активов из конфига."""

def test_retrain_versioning(mock_db_with_data):
    """•Повторный retrain → version=2, старая модель is_active=False."""
```

### 2.4 `tests/unit/test_decision_logic.py`

Тестирование `trading/decision_logic.py` — чистые функции стратегий (без побочных эффектов).

```python
# Что тестируем:
# 1. decide_favorite() корректно вычисляет edge и предлагает токен YES или NO.
# 2. decide_ml_trend() работает с P(flip) и порогами threshold_flip.
# 3. Граничные случаи: вероятности около порогов, отсутствие edge.
```

### 2.5 `tests/unit/test_backtesting.py`

Тестирование логики бэктестинга (Market Replay и метрики).

```python
# Что тестируем:
# 1. runner.py корректно проходит по снапшотам и вызывает стратегии.
# 2. Расчет PnL с учетом slippage и комиссий.
```

### 2.6 Дополнительные тесты моделей и фичей

Тестирование `models/trainer.py` и `models/feature_lags.py` (новые пайплайны логистической регрессии и лаги).

```python
# Что тестируем:
# 1. Pipeline LogisticRegression с учетом гиперпараметров, калибровки, data leakage.
# 2. test_feature_lags.py — корректность расчета динамических лагов.
# 3. test_stop_loss_split.py — миграция и корректное применение FAVORITE/OUTSIDER stop-loss.
```

---

## 3. Integration-тесты

### 3.1 `tests/integration/test_api.py`

Тестирование API-эндпоинтов через FastAPI `TestClient`.

```python
# Фикстуры:
# - test_db: отдельная PostgreSQL БД для тестов (или SQLite in-memory для скорости)
# - test_client: FastAPI TestClient с подменённой БД
# - seeded_db: test_db с предзагруженными данными

def test_health(test_client):
    """GET /health → 200, {"status": "ok"}."""

def test_predict_without_auth(test_client):
    """POST /predict без X-API-Key → 401."""

def test_predict_with_auth(test_client):
    """POST /predict с ключом → 200, корректный PredictResponse."""

def test_predict_invalid_time(test_client):
    """time_left_min=-1 → 422 Validation Error."""

def test_predict_invalid_time_over_15(test_client):
    """time_left_min=20 → 422 Validation Error."""

def test_stats_known_asset(test_client, seeded_db):
    """GET /stats/BTC → 200, bins массив, total_markets > 0."""

def test_stats_unknown_asset(test_client):
    """GET /stats/DOGE → 404."""

def test_dashboard(test_client):
    """GET /dashboard → 200, Content-Type: text/html."""

def test_dashboard_data(test_client, seeded_db):
    """GET /api/dashboard-data/BTC → 200, labels содержит 16 элементов."""
```

### 3.2 `tests/integration/test_settings.py`

Тестирование `GET/PUT /api/settings` — управление настройками через дашборд.

```python
def test_get_settings(test_client):
    """GET /api/settings → 200, содержит все известные ключи с value/source/type."""

def test_get_settings_default_source(test_client):
    """Без записей в runtime_settings → source='default' или 'env'."""

def test_put_settings(test_client):
    """PUT /api/settings {ASSETS: 'BTC,ETH,SOL'} → 200, значение сохранено в БД."""

def test_put_settings_persists(test_client):
    """PUT → GET: source='db', значение совпадает с отправленным."""

def test_put_settings_invalid_key(test_client):
    """PUT {UNKNOWN_KEY: '...'} → 400."""

def test_put_settings_invalid_value(test_client):
    """PUT {MIN_EDGE: 'abc'} → 422."""

def test_settings_priority(test_client, monkeypatch):
    """Значение в runtime_settings имеет приоритет над env var."""
```

### 3.3 `tests/integration/test_backtest_api.py`

Тестирование эндпоинтов `/api/backtest/*`.

```python
def test_submit_backtest(test_client, seeded_db):
    """POST /api/backtest/submit → 200, run_id возвращается."""

def test_backtest_history(test_client):
    """GET /api/backtest/history → 200, массив запусков."""
```

---

## 4. E2E / Smoke-тесты

Запускаются после деплоя Docker-контейнеров. Скрипт `tests/smoke.sh`:

```bash
#!/bin/bash
set -e

BASE_URL="${1:-http://localhost:8001}"
API_KEY="${2:-test-key}"

echo "=== PolyFlip Smoke Tests ==="

# 1. Health
echo -n "Health check... "
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/health")
[ "$STATUS" = "200" ] && echo "✅ OK" || (echo "❌ FAIL ($STATUS)" && exit 1)

# 2. Auth
echo -n "Auth required... "
STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"asset":"BTC","time_left_min":5}')
[ "$STATUS" = "401" ] && echo "✅ OK" || (echo "❌ FAIL ($STATUS)" && exit 1)

# 3. Predict
echo -n "Predict endpoint... "
RESP=$(curl -s -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"asset":"BTC","time_left_min":5}')
echo "$RESP" | python -c "import sys,json; d=json.load(sys.stdin); assert 0<=d['flip_prob']<=1" \
  && echo "✅ OK" || (echo "❌ FAIL" && exit 1)

# 4. Stats
echo -n "Stats endpoint... "
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/stats/BTC" \
  -H "X-API-Key: $API_KEY")
[ "$STATUS" = "200" ] && echo "✅ OK" || (echo "❌ FAIL ($STATUS)" && exit 1)

# 5. Dashboard
echo -n "Dashboard page... "
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/dashboard")
[ "$STATUS" = "200" ] && echo "✅ OK" || (echo "❌ FAIL ($STATUS)" && exit 1)

# 6. Settings (GET)
echo -n "Settings GET... "
RESP=$(curl -s "$BASE_URL/api/settings")
echo "$RESP" | python -c "import sys,json; d=json.load(sys.stdin); assert 'settings' in d" \
  && echo "✅ OK" || (echo "❌ FAIL" && exit 1)

# 7. Settings (PUT + verify)
echo -n "Settings PUT... "
curl -s -X PUT "$BASE_URL/api/settings" \
  -H "Content-Type: application/json" \
  -d '{"settings": {"COLLECT_INTERVAL_MINUTES": "30"}}' | \
  python -c "import sys,json; d=json.load(sys.stdin); assert 'COLLECT_INTERVAL_MINUTES' in d.get('updated',[])" \
  && echo "✅ OK" || (echo "❌ FAIL" && exit 1)

echo "=== All smoke tests passed ✅ ==="
```

---

## 5. Тестовые фикстуры (`tests/conftest.py`)

```python
# Ключевые фикстуры:

@pytest.fixture
def settings_override():
    """Переопределение настроек для тестов (тестовая БД, тестовый API-ключ)."""

@pytest.fixture
async def test_db(settings_override):
    """Создание и миграция тестовой БД. Очистка после теста."""

@pytest.fixture
def test_client(test_db):
    """FastAPI TestClient с подменённой БД и настройками."""

@pytest.fixture
async def seeded_db(test_db):
    """Тестовая БД с предзагруженными данными (100 snapshot-записей)."""

@pytest.fixture
def mock_model():
    """Мок LogisticRegression с предопределённым predict_proba."""
```

---

## 6. Что НЕ тестируем автоматически

| Что | Почему | Как проверяем |
|---|---|---|
| Реальные запросы к Polymarket API | Rate limits, нестабильность | Мокаем httpx-ответы |
| Визуальное качество дашборда | Субъективно | Ручная проверка в браузере |
| Производительность под нагрузкой | Не критично для MVP | При необходимости — locust |
| Реальное переобучение модели | Долго, зависит от данных | Тестируем на маленьких данных |

---

## 7. Запуск тестов

```bash
# Все тесты
poetry run pytest tests/ -v

# Только unit (быстро, без БД)
poetry run pytest tests/unit/ -v

# Только integration (нужна PostgreSQL)
poetry run pytest tests/integration/ -v

# С покрытием
poetry run pytest tests/ --cov=polyflip --cov-report=term-missing

# Smoke-тесты (после docker-compose up)
bash tests/smoke.sh http://localhost:8001 your-api-key
```

---

## 8. CI-конвейер (будущее)

```yaml
# .github/workflows/test.yml (ориентир)
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
          POSTGRES_DB: polyflip_test
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install poetry && poetry install
      - run: poetry run pytest tests/ -v --cov=polyflip
```

---

## 9. Критерии приёмки (Definition of Done)

Сервис считается готовым к продуктивному использованию, когда:

- [ ] Все unit-тесты проходят зелёным
- [ ] Все integration-тесты проходят зелёным
- [ ] Smoke-тесты проходят на запущенном Docker
- [ ] Дашборд отображает график P(flip) с реальными данными
- [ ] Настройки сохраняются через дашборд и вступают в силу без перезапуска
- [ ] API возвращает осмысленные предсказания (не дефолтные 0.25) для BTC и ETH
- [ ] Scheduler автоматически собирает данные и переобучает модели
- [ ] Модели хранятся в model_registry (БД), не в файловой системе
- [ ] Логи structlog записываются в JSON-формате
- [ ] `.env.example` задокументирован
- [ ] README содержит инструкции по запуску
