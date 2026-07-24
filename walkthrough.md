# 🏆 Отчёт об устранении 4 технических недочётов

Все 4 обнаруженных бага успешно исправлены и проверены.

---

## 🛠️ Что сделано:

### 1. **Баг 1 — Проверка существования модели при активации (`activate_model`)**
* Восстановлена проверка `if result.rowcount == 0: raise HTTPException(status_code=404, detail="Model not found")`.
* Попытка активировать несуществующую версию отдаёт 404 вместо ложного 200 OK.

### 2. **Баг 2 — Исключение Race Condition в кэше моделей (`list_models`)**
* Внедрен паттерн **Double-Check Locking** с использованием `asyncio.Lock()`:
  ```python
  async with _models_cache_lock:
      if "data" in _models_cache and time.time() - _models_cache.get("time", 0) < _MODELS_CACHE_TTL:
          return _models_cache["data"]
  ```
* Защищает от Thundering Herd (одновременного пробоя кэша сотней параллельных запросов при его очистке).

### 3. **Баг 3 — Защита эндпоинта коэффициентов авторизацией (`get_model_coefficients`)**
* К эндпоинту `GET /analytics/models/{asset}/{version}/coefficients` добавлена зависимость авторизации `dependencies=[Depends(verify_api_key)]`.
* Предотвращена утечка торговой логики и весов моделей неопознанным внешним наблюдателям.

### 4. **Баг 4 — Проверка успешности ответа `!res.ok` в JS (`app.js`)**
* В `renderSelectedModelWeights` добавлена проверка статуса `if (!res.ok)`.
* При ответе с ошибкой (например `401 Unauthorized` или `404 Not Found`) клиент обрабатывает ошибку без слома GUI.

---

## 🧪 Деплой
* Все изменения синтаксически проверены и отправлены в Git (`fix: resolve 4 API bugs (activate 404, asyncio.Lock cache, verify_api_key on coefs, res.ok check)`).
* Контейнеры на сервере `34.50.54.183` перезапущены.
