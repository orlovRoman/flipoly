# 🏆 Отчёт об удалении параметра MAX_BET_EDGE

Параметр **`MAX_BET_EDGE`** («Максимальный Edge для ставки») полностью удалён из пользовательского интерфейса, API и торгового движка.

---

## 🛠️ Что сделано:

1. **Веб-интерфейс (`polyflip/templates/trading.html` & `trading.js`)**:
   * Поле `Максимальный Edge для ставки (%)` (`MAX_BET_EDGE`) удалено из HTML-формы настроек.
   * Удалены обработчики чтения и сохранения `MAX_BET_EDGE` в JavaScript.

2. **Реестр настроек & Конфигурация (`polyflip/settings_registry.py` & `config.py`)**:
   * `MAX_BET_EDGE` удалён из списка допустимых редактируемых ключей и дефолтных параметров системы.

3. **API настроек (`polyflip/api/settings.py`)**:
   * Удалены алиасы, нормализации и логика кросс-валидации (`MAX_BET_EDGE >= MIN_EDGE`).

4. **Торговый движок & Бэктестинг (`polyflip/trading/trading_config.py`, `decision_runners.py`, `runner.py`)**:
   * Удалены привязки к `max_bet_edge`.

---

## 🧪 Деплой
* Изменения проверены на синтаксис Python и JavaScript (`node -c`).
* Все коммиты отправлены в Git (`feat: remove MAX_BET_EDGE parameter from UI and codebase`) и развернуты на сервере `34.50.54.183`.
