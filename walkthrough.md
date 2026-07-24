# 🏆 Отчёт о детализации параметров при применении пресета

Во всплывающее уведомление (alert) добавлен детальный список всех изменённых параметров с их установленными значениями.

---

## 🛠️ Что сделано:

1. **Расширение ответа API (`polyflip/services/preset_service.py` и `polyflip/api/presets.py`)**:
   - Метод `restore_preset` теперь возвращает `(changed_count, updated_params)`, где `updated_params` — словарь вида `{ "MIN_EDGE": "0.1", "FLIP_THRESHOLD": "0.6" }`.
   - Эндпоинт `POST /api/presets/{preset_id}/restore` отдаёт ключ `updated_params` клиенту.

2. **Форматирование списка в UI (`polyflip/static/js/trading.js`)**:
   - При успешном применении пресета выводится алерт с перечнем всех измененных ключей:
     ```
     ✅ Пресет "1" успешно применен!

     Изменённые параметры (4):
     • MIN_EDGE: 0.10
     • FLIP_THRESHOLD: 0.60
     • FAVORITE_THRESHOLD: 0.55
     • NO_MIN_EDGE: 0.05
     ```

---

## 🧪 Деплой
* Изменения проверены и отправлены в Git (`feat(presets): include detailed updated parameter names and values in restore alert`).
* Контейнеры на сервере `34.50.54.183` перезапущены.
