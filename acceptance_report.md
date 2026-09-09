# Итоговый отчёт приёмки (Final Acceptance Report)

## Этап 1: Исправление данных и расчётов

| Пункт | Изменение | Файл / Функция | Проверка | Результат | Ограничение |
|-------|-----------|----------------|----------|-----------|-------------|
| 1. Фиксация точки аудита | Сохранение коммита a2bc8af, хэши | `artifacts/research/reproducible_data_manifest.json` | Хэши стабильны и не меняются без причины | PASS | Нет |
| 2. Происхождение результатов | Динамический расчёт git, dirty и SHA256 файлов | `polyflip/research/stage2_regime_study.py:run_stage2_study` | Метаданные содержат актуальные значения | PASS | Нет |
| 3. Явные статусы результатов | Добавлены статусы COMPUTED, BLOCKED_DATA | `polyflip/research/stage2_regime_study.py` | Варианты без сделок имеют null PnL и статус | PASS | Нет |
| 4. Удаление искусственной глубины | Исключена автоматическая генерация уровней | `polyflip/collector/orderbook_depth.py` | BLOCKED_DATA вместо исполнения на глубине | PASS | Глубина пока недоступна в CSV |
| 5. Отделение сценариев ликвидности | Сценарные стаканы вынесены в SCENARIO_ONLY | `polyflip/research/orderbook_execution.py` | Сценарии не попадают в историческую таблицу | PASS | Сценарии генерируются отдельно |
| 6. Метки коллектора | Отделено время запроса и ответа (received_at) | `polyflip/collector/client.py` | Временные метки не смешиваются | PASS | Точность зависит от сети |
| 7. Односторонний стакан | Обработка случаев KeyError при пустом стакане | `polyflip/collector/client.py` | Ошибки не прерывают сбор данных | PASS | Spread не может быть рассчитан |
| 8. Missing != Zero | Отсутствующий размер вызывает ValueError, а не 0 | `polyflip/collector/orderbook_depth.py` | Уровни без размера отвергаются | PASS | Нет |
| 9. Интеграция стакана | Тестирование клиента -> парсера -> БД | `tests/research/test_comprehensive_plan.py` | Независимые YES и NO корректно пишутся | PASS | Нет |
| 10. Фактическое покрытие глубины | Вычисление доступности и возраста стакана | `polyflip/collector/orderbook_depth.py:compute_orderbook_completeness_report` | Истинное покрытие измеряется | PASS | Глубина ещё не собрана |
| 11/12. CS_short источник | Введены timestamps, окно 10м (300с+ покрытие) | `polyflip/research/stage2_regime_study.py` | Ошибка при отсутствии временных меток | PASS | Нет |
| 13. Причинность underlying | Строгое окно до decision_at (causal alignment) | `polyflip/research/regime_features.py` | Future leaks отсутствуют | PASS | Нет |
| 14. CT и CS неизменны | Раздельные варианты для старых и новых фильтров | `polyflip/research/stage2_regime_study.py` | Старые фильтры сохранены | PASS | Нет |
| 15. Единицы strike | Использование underlying local_mean вместо token | `polyflip/research/regime_features.py:compute_strike_context` | Оценки возврата underlying корректны | PASS | Нет |
| 16. Canonical strike | Вычисляется, если неизвестен - пропускается | `polyflip/collector/client.py` | Прокси и канонический оцениваются отдельно | PASS | Исторически не собран |
| 17. Устойчиво дешёвый | Минимальное окно 300с, макс разрыв 300с | `polyflip/research/stage2_regime_study.py` | Быстрые 3 снапшота отвергаются | PASS | Нет |
| 18. Границы корзин | 0.40 включён в последнюю корзину (<=) | `polyflip/research/stage2_regime_study.py` | Все сделки распределены без потерь | PASS | Нет |
| 19. Bootstrap expectancy | Пересэмплирование числа сделок в репликах | `polyflip/research/stage2_regime_study.py` | Реплики без сделок не вызывают деление на 0 | PASS | Нет |
| 20. Задержка исполнения | Отделено latency от "исторического исполнения" | `polyflip/research/orderbook_execution.py` | Слишком старый стакан даёт STALE_BOOK | PASS | Нет |
| 21. Повторное потребление | ExecutionVolumeTracker вычитает объёмы | `polyflip/research/orderbook_execution.py` | Объём не дублируется | PASS | Нет |
| 22. Параметризация бюджета | Убран хардкод 1 USDC, используется budget_usdc | `polyflip/research/orderbook_execution.py` | Расчёты корректны при $1, $5, $10 | PASS | Нет |

## Этап 2: Пересчёт исследования

| Пункт | Изменение | Файл / Функция | Проверка | Результат | Ограничение |
|-------|-----------|----------------|----------|-----------|-------------|
| 23. Реестр возможностей | Сохранение причин исключения в ledger | `polyflip/research/stage2_regime_study.py` | Все варианты записаны | PASS | Нет |
| 24. Пересчёт на сопоставимых данных | Оценка CT, CS, CS_short на одном пуле | `polyflip/research/stage2_regime_study.py` | Общий common_opportunity_ledger | PASS | Нет |
| 25. Исполнение по вариантам | Расчёт таблиц для каждого варианта отдельно | `polyflip/research/stage2_regime_study.py` | Блокируется из-за отсутствия глубины | BLOCKED_DATA | Ждём данные стаканов |
| 26. Когорты и концентрация | CT-only, CS-only, Both, Neither разбивка | `polyflip/research/stage2_regime_study.py` | Данные разложены без потерь | PASS | Нет |
| 27. Анализ фаворитов | Формализованы группы траекторий | `polyflip/research/stage2_regime_study.py` | Группы взаимоисключающие | PASS | Нет |
| 28. Исследовательская разбивка | Переименовано из holdout | `polyflip/research/stage2_regime_study.py` | Exploratory period оценивается | PASS | Нет |
| 29. Удаление выводов | Матрица принятия решений на лету | `polyflip/research/stage2_regime_study.py` | Авто-вывод вместо текста | PASS | Нет |
| 30. Содержательные тесты | if exists -> assert, интеграционные тесты | `tests/research/test_comprehensive_plan.py` | Ошибки падают вместо пропуска | PASS | Нет |

