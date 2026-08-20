# План внедрения rule-based market regime filter

## Назначение

Этот документ предназначен для агента-оркестратора, который не знаком с проектом
PolyFlip и должен организовать внедрение фильтра режима рынка для 15-минутных
Polymarket-сделок.

Цель первой версии — не обучать новую LightGBM-модель, а сделать проверяемый
rule-based фильтр на закрытых Binance 15m-свечах. Фильтр должен различать:

- `TREND_UP`;
- `TREND_DOWN`;
- `SIDEWAYS`;
- `HIGH_VOL_CHOP`;
- `MIXED`/`UNKNOWN`, если признаки противоречат друг другу или истории
  недостаточно.

Фильтр используется как защитный gate для `OUTSIDER` и как подтверждение
направления для `ML_TREND FAVORITE`. Сначала он работает только в `SHADOW`,
затем включается отдельно для PAPER и LIVE после backtest и ручного одобрения.

## Контекст проекта, который нужно проверить в начале

Предварительно обнаружены следующие компоненты, но перед изменением кода агент
обязан подтвердить их состояние в текущем checkout:

- `polyflip/crypto/feature_builder.py` — признаки по Binance-свечам;
- `polyflip/models/sequence_features.py` — короткие sequence/trend-признаки;
- `polyflip/crypto/predictor.py` — текущий LightGBM direction engine;
- `polyflip/crypto/market_direction_service.py` — фиксация сигнала на время
  Polymarket-рынка;
- `polyflip/trading/decision_runners.py`, `decision_logic.py` и
  `combined_voting.py` — принятие торгового решения;
- `polyflip/trading/trade_recorder.py` и funnel logger — аудит решения;
- `polyflip/db/models.py` и `alembic/versions/` — модели и миграции;
- `tests/` — pytest suite;
- `pyproject.toml` и `.github/workflows/ci.yml` — команды проверки.

В текущем проекте `CryptoSignal.regime` означает режим волатильности
(`low_vol`/`mid_vol`/`high_vol`), вычисленный через `vol_6 / vol_24`. Его нельзя
без дополнительной логики трактовать как боковик или направленный рынок.

## Жёсткие ограничения

1. Работа ведётся только в отдельной ветке, например
   `codex/market-regime-filter-implementation`; `main` не изменяется.
2. Нельзя выполнять `git reset --hard`, `git clean`, удалять untracked-файлы или
   перезаписывать чужие изменения.
3. Перед началом и перед каждым заданием нужно сохранять baseline:
   `git status --short --branch`. Existing untracked-файлы считаются чужими.
4. Любое чтение production PostgreSQL — только read-only SQL. Миграции,
   backfill и изменение runtime settings на боевой БД запрещены.
5. Все timestamps нормализуются в UTC. Для расчёта на момент `t` разрешены
   только полностью закрытые свечи с `close_time <= t`.
6. Нельзя использовать будущие цены, итог сделки, PnL или будущий Polymarket
   outcome как признак текущего режима.
7. Не менять существующие LightGBM/LogReg target, feature schema и пороги без
   отдельного задания. Новая LightGBM-модель режима в этот этап не входит.
8. `ACTIVE` не является значением по умолчанию. При недоступном фильтре безопасное
   поведение: не увеличивать размер ставки; для `OUTSIDER` в active-режиме
   выбирать консервативный multiplier, согласованный в спецификации.

## Протокол оркестратора и agy

Оркестратор не реализует подзадания сам. Каждое задание выполняется отдельным
вызовом Antigravity через настроенный интерфейс `agy`.

Логический формат вызова:

```text
agy(
  task_id="MRF-Txx",
  objective="...",
  repository="/home/orlovrp/flipoly",
  branch="codex/market-regime-filter-implementation",
  allowed_files=[...],
  required_checks=[...],
  non_goals=[...],
  previous_artifacts=[...]
)
```

Если в среде нет рабочего `agy`-интерфейса, оркестратор останавливается до
реализации и сообщает блокирующую причину. Он не подменяет вызов самостоятельной
реализацией.

После каждого вызова `agy` оркестратор обязан:

1. проверить, что агент изменял только разрешённые файлы;
2. выполнить `git diff --check`;
3. просмотреть diff и проверить отсутствие lookahead, unsafe SQL и изменения
   production-конфигурации;
4. выполнить целевые тесты задания;
5. при изменении публичного контура выполнить `python -m compileall -q
   polyflip` или эквивалентную compile-проверку;
6. только после успешной проверки зафиксировать отдельный commit задания.

Если проверка не пройдена, оркестратор не исправляет код сам. Он вызывает `agy`
повторно с corrective task, включающим исходный `task_id`, точную команду,
полный релевантный вывод ошибки, ожидаемое поведение и ограничение «исправить
только эту проблему».

Разрешено максимум три corrective calls на одно задание. После третьей неудачи
задание помечается `BLOCKED`, изменения не вливаются в следующую часть плана, а
в финальном отчёте указывается причина.

Каждый commit должен иметь формат:

```text
MRF-Txx: <краткое описание завершённого задания>
```

## Контракт результата

К концу плана должен существовать единый объект, например
`MarketRegimeSnapshot`, содержащий минимум:

```text
asof_close_time_utc
feature_version
global_regime
global_score
global_direction
asset_regimes
asset_scores
features_used
history_ready
reason_codes
filter_mode
```

Объект должен быть детерминированным: одинаковые закрытые свечи, конфигурация и
время дают одинаковый результат.

Для каждого актива и горизонта `h` используются log-return:

```text
r_asset_h(t) = log(close_asset(t) / close_asset(t-h))
```

Для 15m-свечей базовые горизонты:

```text
4h  = 16 свечей
12h = 48 свечей
24h = 96 свечей
```

Trend efficiency:

```text
efficiency_h = abs(sum(last_h_returns)) /
                (sum(abs(last_h_returns)) + epsilon)
```

Для общего рынка строятся basket-признаки по активам, реально включённым в
конфигурацию:

```text
market_return_h = median(asset_return_h)
breadth_up_h = share(asset_return_h > 0)
breadth_down_h = share(asset_return_h < 0)
cross_asset_dispersion_h = std(asset_return_h)
market_efficiency_h = median(asset_efficiency_h)
```

Нельзя без проверки зашивать BTC, ETH, SOL, XRP и DOGE как единственный список:
активы должны определяться из project settings. В тестах допустим фиксированный
набор synthetic assets.

Начальные правила должны быть конфигурируемыми, а не разбросанными по коду.
Порог `+1%` для одного дня может быть стартовой гипотезой для анализа, но не
финальным универсальным порогом: он должен проверяться относительно исторической
волатильности и на нескольких неделях данных.

## Атомарные задания

### MRF-T00 — создать ветку и зафиксировать baseline

**Задание agy:** проверить текущую ветку, статус, remote и список untracked;
создать feature-ветку только если она ещё не создана; ничего не удалять и не
редактировать.

**Готово, если:** текущая ветка не `main`, baseline сохранён в отчёте
оркестратора, список чужих untracked-файлов известен.

**Проверка:** `git status --short --branch`, `git branch --show-current`.

**Correction:** только исправить ветку/состояние, не делать cleanup.

### MRF-T01 — разведка фактической архитектуры

**Задание agy:** прочитать README, `pyproject.toml`, CI и фактический call chain
от загрузки свечей до `decision_runners`; найти модель `MarketDirectionSignal`,
таблицы свечей, стратегии `OUTSIDER` и `ML_TREND`, текущие режимы `OFF/SHADOW/ACTIVE`.
Изменения кода не делать.

**Артефакт:** `docs/design/market-regime-filter-recon.md` с путями файлов,
функциями входа/выхода, тестовыми командами и найденными ограничениями.

**Проверка оркестратора:** все названные symbols/functions существуют; нет
утверждений, не подтверждённых исходниками.

### MRF-T02 — утвердить спецификацию фильтра

**Задание agy:** на основе T01 написать design specification с контрактом
`MarketRegimeSnapshot`, UTC/as-of boundary, горизонтами 4h/12h/24h, per-asset и
global basket features, режимами `TREND_UP`, `TREND_DOWN`, `SIDEWAYS`,
`HIGH_VOL_CHOP`, `MIXED`, минимальной историей, reason codes, fail-safe поведением,
policy для `OUTSIDER`/`ML_TREND` и режимами `OFF`/`SHADOW`/`ACTIVE`.

**Артефакт:** `docs/design/market-regime-filter.md`.

**Проверка:** нет будущих признаков, volatility regime не смешан с directional
regime, новая LightGBM-модель не добавлена.

### MRF-T03 — подтвердить контракт исторических свечей

**Задание agy:** проверить ORM/repository и fixtures: open/high/low/close,
`close_time`, `is_closed`, timezone, сортировку, дубликаты и пропуски. Если нужно,
добавить read-only диагностический тест.

**Готово, если:** однозначно определено получение последних закрытых свечей на
границе решения и отбрасывание incomplete/future rows.

**Проверка:** тест с naive/UTC datetime, незакрытой свечой, дубликатом и
пропуском; production DB не изменяется.

### MRF-T04 — реализовать чистый feature builder

**Задание agy:** создать изолированный pure module, например
`polyflip/crypto/market_regime.py`, и реализовать per-asset returns 4h/12h/24h,
volatility, trend efficiency, basket median returns, breadth up/down,
cross-asset dispersion, market efficiency, `history_ready` и reason codes.

Функция не читает БД, не меняет global state и не зависит от будущего
Polymarket outcome. Пропавший актив не превращается в нулевую доходность без
явной маркировки coverage.

**Проверка:** type/shape contract, finite values, стабильная сортировка,
неизменяемый input, `git diff --check`, targeted compile.

### MRF-T05 — unit-тесты feature builder

**Задание agy:** добавить synthetic tests для монотонного роста и падения всех
активов, чередующихся свечей с малым net return, одного актива против рынка,
недостаточной истории, incomplete candle, UTC/naive timestamp, отсутствующего
актива, flat price и аномального volume.

**Проверка:** targeted pytest проверяет конкретные значения и as-of boundary,
а не только отсутствие исключения.

### MRF-T06 — реализовать rule-based classifier

**Задание agy:** добавить чистую классификацию global и asset regime. Правила
конфигурируемы и прозрачны:

- `TREND_UP`: положительное движение на нескольких горизонтах, высокий
  `breadth_up`, достаточная efficiency;
- `TREND_DOWN`: симметрично;
- `SIDEWAYS`: небольшой basket return, смешанный breadth, низкая efficiency;
- `HIGH_VOL_CHOP`: высокая volatility при низкой efficiency;
- `MIXED`/`UNKNOWN`: противоречивые признаки или недостаточная история.

Вернуть score, direction и reason codes. `p_up == 0.5` LightGBM не считать
доказательством боковика.

**Проверка:** synthetic scenarios, симметрия UP/DOWN, отсутствие переключения на
неизвестные значения, пороги в одной config-структуре.

### MRF-T07 — реализовать стратегическую policy-функцию

**Задание agy:** создать отдельную функцию, которая получает snapshot, стратегию
и режим фильтра и возвращает `allow`, `stake_multiplier`, `reason`.

Начальная policy, окончательно проверяемая backtest:

- `OUTSIDER`: normal в `SIDEWAYS`, reduced/blocked в TREND;
- `ML_TREND`: разрешать направление, совпадающее с global и local trend;
- `HIGH_VOL_CHOP`/`MIXED`: conservative multiplier;
- `UNKNOWN`: не увеличивать риск.

Не смешивать policy с feature builder.

**Проверка:** таблица regime × strategy × mode; `SHADOW` не меняет финальное
решение; `ACTIVE` меняет только заявленный контур.

### MRF-T08 — интегрировать snapshot на границе решения

**Задание agy:** встроить вычисление режима в фактическую границу решения,
используя только candles до `recorded_at`/market opening boundary. Если
`market_direction_service` фиксирует сигнал на lifetime рынка, режим тоже
фиксируется для этого рынка и не пересчитывается будущими свечами.

В `SHADOW` действующее решение не менять; сохранить snapshot в decision result.

**Проверка:** integration test на отсутствие lookahead, стабильность повторного
вызова одного market и безопасное поведение при отсутствии истории.

### MRF-T09 — добавить конфигурацию и feature flag

**Задание agy:** выбрать существующий settings-механизм и добавить минимальные
параметры:

```text
MARKET_REGIME_FILTER_MODE=OFF|SHADOW|ACTIVE
MARKET_REGIME_FILTER_VERSION
MARKET_REGIME_MIN_HISTORY
MARKET_REGIME_OUTSIDER_TREND_MULTIPLIER
MARKET_REGIME_UNKNOWN_MULTIPLIER
MARKET_REGIME_BREADTH_THRESHOLD
MARKET_REGIME_EFFICIENCY_THRESHOLD
```

Default — `SHADOW` или `OFF`, но не `ACTIVE`. Не применять миграции и не менять
runtime settings на production DB.

**Проверка:** settings parse tests, invalid values, backward-compatible defaults,
startup/compile.

### MRF-T10 — добавить аудит и telemetry

**Задание agy:** записывать в decision funnel или metadata namespace:
filter mode/version, as-of timestamp, global/asset regime, компактный feature
summary, score/reason codes, `allow`, multiplier, applied/not-applied flag и
failure reason. Не ломать существующую семантику `lgbm_metadata`.

**Проверка:** serialization tests, backward compatibility без namespace, явная
маркировка `SHADOW` как `not_applied`.

### MRF-T11 — подключить policy к OUTSIDER и ML_TREND

**Задание agy:** подключить T07 к реальному месту выбора/сайзинга стратегии,
найденному в T01. В `SHADOW` только рассчитывать и логировать; в `ACTIVE`
применять policy только к заявленным стратегиям.

Не допустить влияния на entry-model, funding veto, stop-loss, settlement и
дневные лимиты.

**Проверка:** decision tests для SIDEWAYS/TREND_UP/TREND_DOWN/UNKNOWN и проверка
неизменности unrelated strategies.

### MRF-T12 — воспроизводимый offline backtest

**Задание agy:** использовать существующий backtester/analytics или создать
read-only evaluator. Сравнить baseline без фильтра, shadow-классификацию и
active policy на диапазоне минимум 30–60 дней, если данные доступны.

Посчитать total PnL, `OUTSIDER` PnL, `ML_TREND` PnL, WR, число сделок, avoided
losses, blocked winners, regime coverage, max drawdown, результаты по активам и
UTC-дням. Включить 29.07 и 02.08 как smoke examples, но не оптимизировать только
по ним.

**Проверка:** нет записи в production DB, результаты воспроизводимы, версия
config сохранена, есть тест на отсутствие lookahead.

### MRF-T13 — выбрать пороги по истории

**Задание agy:** по T12 предложить пороги и multiplier, разделив development и
validation по времени. Предпочитать устойчивые пороги для нескольких активов и
недель. Подготовить trade-off: предотвращённые убытки OUTSIDER против
заблокированных прибыльных возможностей.

`ACTIVE` автоматически не включать.

**Проверка:** повторный backtest с замороженными параметрами; код не меняется
без явного согласования конфигурации.

### MRF-T14 — shadow rollout в PAPER

**Задание agy:** включить `SHADOW` только в PAPER, не меняя исполнение, и добавить
ежедневный отчёт по режимам, coverage, policy и расхождениям с фактическими
решениями. Проверить startup, collector, decision worker и settlement при
отсутствующем snapshot.

**Проверка:** paper smoke test, version/as-of в логах, restart test, отсутствие
live order calls.

### MRF-T15 — документация и runbook включения

**Задание agy:** описать режимы, актуальность свечей, reason codes, telemetry,
выключение через `OFF`, rollback, критерии перед `ACTIVE` и признаки stale data.
Не обещать прибыльность и не считать два дня достаточной статистикой.

### MRF-T16 — финальная независимая проверка

**Задание agy:** провести review diff текущей ветки на отсутствие main changes,
удаления чужих файлов, lookahead, нарушения UTC/as-of, default `ACTIVE`,
production DB writes и backward compatibility. Проверить unit/integration/
backtest tests и CI commands.

**Проверка оркестратора:**

```bash
poetry run python -m compileall -q polyflip
poetry run pytest tests/ -m "not live" --strict-config --strict-markers -q
poetry run black --check <изменённые Python-файлы>
poetry run flake8 <изменённые Python-каталоги> --max-line-length=100
git diff --check
```

Если полный pytest требует недоступной инфраструктуры, это фиксируется как
environment limitation с результатом targeted tests; молча считать его пройденным
нельзя.

## Критерий завершения всей задачи

Задача выполнена только если все обязательные задания T00–T16 завершены либо
явно помечены `BLOCKED`; фильтр детерминирован на закрытых UTC-свечах; есть
per-asset и global regime; `OUTSIDER` и `ML_TREND` имеют policy; `SHADOW` не
меняет торговлю; есть baseline-vs-filter backtest; сохраняются telemetry/version/
reason codes; `ACTIVE` не включён автоматически; чужие untracked-файлы не попали
в commit; итоговый отчёт содержит commit hash, тесты, backtest, ограничения и
рекомендацию по включению.

## Что сознательно не входит в этот этап

- новая LightGBM regime-модель;
- переобучение текущих direction-моделей;
- изменение target `FLIP_VS_FINAL_OUTCOME`;
- изменение production database;
- автоматическое включение LIVE;
- вывод о прибыльности фильтра только по 29.07 и 02.08.

После устойчивого shadow/backtest результата отдельным планом можно обучить
LightGBM на тех же market-regime features и сравнить её с rule-based baseline.
