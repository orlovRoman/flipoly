# 🏆 Отчёт о реализации удаления моделей в API

Устранена ошибка «Ошибка сети при удалении» при клике на иконку 🗑️ в интерфейсе моделей.

---

## 🛠️ Что сделано:

1. **Реализация эндпоинта `DELETE /api/analytics/models/{asset}/{version}` (`polyflip/api/analytics.py`)**:
   * Добавлен ранее отсутствовавший серверный эндпоинт удаления моделей.
   * **Защита активных моделей**: если модель является текущей активной (`is_active == True`), бэкенд блокирует удаление с ошибкой `400 Bad Request: Cannot delete active model. Activate another model first.`.
   * Для архивных неактивных моделей операция удаляет запись из базы данных `ModelRegistry` и очищает кэш `invalidate_models_cache()`.

---

## 🧪 Деплой
* Изменения проверены и отправлены в Git (`feat(api): add DELETE /analytics/models/{asset}/{version} endpoint with active model protection`).
* Контейнеры на сервере `34.50.54.183` перезапущены.
