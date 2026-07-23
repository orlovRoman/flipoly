# 🏆 Отчёт об устранении крайних ошибок в торговой логике

Успешно исправлены 3 узких места и скрытые рантайм-ошибки в торговом движке.

---

## 🛠️ Детали исправлений:

### 1. **Баг 1 — Устранение TypeError в `decide_favorite` (NO-side)**
* В `polyflip/trading/decision_logic.py` на стороне NO вызов `_resolve_final_bet(edge, signal.no_ask)` содержал 2 аргумента вместо 3.
* Исправлено на `_resolve_final_bet(edge, signal.volume_5min, config)`. Устранена угроза падения процесса при выборе NO-сторон фаворита.

### 2. **Баг 2 — Порядок проверки `dead zone` в `decide_outsider`**
* Проверка `is_in_dead_zone(signal.mid_price, dead_zone)` перенесена в начало функции `decide_outsider` до проверки `p_flip_calibrated < flip_thresh`.
* Приведено к единому стандарту с `decide_ml_trend`.

### 3. **Баг 3 — Заполнение `applied_lower` и `applied_upper` в `DecisionResult`**
* В `polyflip/trading/decision_runners.py` при создании `DecisionResult` в `decide_ml_mode` переданы вычисленные значения порогов `applied_lower=lower` и `applied_upper=upper`.
* Теперь воронка `log_funnel` гарантированно регистрирует реальные пороги в логах и дашборде.

---

## 🧪 Деплой
* Все тесты самопроверки пройдены.
* Коммит и пуш отправлены в Git (`fix: resolve TypeError in decide_favorite NO-side, fix dead_zone order, populate applied_lower/upper`).
* Приложение пересобрано и перезапущено на сервере `34.50.54.183`.
