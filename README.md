# PolyFlip — Аналитический Микросервис для Polymarket 15m Рынков

> Самодостаточный сервис, который собирает исторические данные по 15-минутным рынкам Polymarket,
> обучает модели P(flip | features, asset) и отдаёт предсказания через REST API + встроенный дашборд.

> **Стратегия:** сначала аналитика — собрать данные, обучить модели, посмотреть есть ли edge.
> Торговый слой (бот-исполнитель, расчёт EV, position sizing) реализуется **только если** аналитика покажет, что рынок неэффективен.

---

## Что такое "flip"?

**Flip** — это ситуация, когда финальный исход рынка отличается от того, что показывала цена в момент наблюдения.

Пример: за 5 минут до закрытия рынка цена YES = 0.72 (рынок "уверен" в YES).
Рынок разрешился в NO → **произошёл flip**.

Модель предсказывает вероятность такого разворота на основе 6 признаков:
`time_left_min`, `mid_price`, `spread`, `volume_last_5min`, `price_velocity`, `hour_of_day`.

---

## Основные возможности

| Возможность | Описание |
|---|---|
| 📊 **Сбор данных** | Автоматический сбор завершённых + live-мониторинг активных 15m рынков |
| 🤖 **ML-модели** | LogisticRegression (6 фичей) для каждого актива, переобучение каждые 24 часа |
| 🔮 **REST API** | `POST /predict`, `GET /stats/{asset}`, `GET /backtest/{asset}`, `POST /retrain` |
| 📈 **Дашборд** | Графики P(flip), статус системы, **панель настроек** — всё в одном месте |
| ⚙️ **Live Settings** | Все параметры редактируемы через дашборд без перезапуска |
| 🔐 **API-ключ** | Базовая защита через `X-API-Key` заголовок |
| 🐳 **Docker** | Полная контейнеризация: API + Scheduler + PostgreSQL |
| 📡 **Live-цены** | Поллинг текущих цен активных рынков каждую минуту |

---

## Быстрый старт

### Предварительные требования

- Docker + docker-compose
- (Опционально) Python 3.12+ и Poetry для локальной разработки

### Запуск через Docker

```bash
# 1. Клонировать репозиторий
git clone <repo-url> && cd polyflip

# 2. Создать .env файл
cp .env.example .env
# Отредактировать .env: задать API_KEY и другие параметры

# 3. Запустить
docker-compose up -d

# 4. Проверить работоспособность
curl http://localhost:8001/health

# 5. Открыть дашборд
# http://localhost:8001/dashboard
```

### Локальная разработка

```bash
# 1. Установить зависимости
poetry install

# 2. Поднять PostgreSQL (можно через Docker)
docker-compose up -d db

# 3. Применить миграции
poetry run alembic upgrade head

# 4. Запустить сервис
poetry run uvicorn polyflip.api.main:app --host 0.0.0.0 --port 8001 --reload
```

---

## Поддерживаемые активы (v1)

- **BTC** — Bitcoin 15-minute markets
- **ETH** — Ethereum 15-minute markets

Новые активы добавляются через дашборд (вкладка «Настройки») или переменную окружения `ASSETS=BTC,ETH,SOL`.

---

## Дашборд

Три вкладки:

1. **📈 Аналитика** — график P(flip) по time_left, выбор актива, таблица bins
2. **🔧 Статус** — здоровье системы, модели, drift, последний сбор
3. **⚙️ Настройки** — все параметры сервиса, редактируемые в реальном времени

---

## Будущее: торговый бот

> Торговый слой (strategy/, `GET /signal`, расчёт edge, position sizing) задокументирован
> в ARCHITECTURE.md, но реализуется **только после** того, как аналитика покажет наличие edge.

Когда торговля будет включена, бот-исполнитель будет вызывать один эндпоинт:

```python
# Будущий бот — тупой исполнитель ордеров
async def check_and_trade(asset: str) -> None:
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"http://polyflip:8001/signal?asset={asset}",
            headers={"X-API-Key": API_KEY},
            timeout=2.0,
        )
    signal = resp.json()
    if signal["action"] != "hold":
        place_order(signal["action"], signal["bet_size"])
```

---

## Документация

| Документ | Описание |
|---|---|
| [ARCHITECTURE.md](./ARCHITECTURE.md) | Архитектура системы, слои, потоки данных |
| [DEVELOPMENT.md](./DEVELOPMENT.md) | Порядок разработки, фазы, чек-листы |
| [API.md](./API.md) | Спецификация REST API |
| [TESTING.md](./TESTING.md) | Стратегия тестирования и верификации |

---

## Лицензия

Private / Internal use only.
