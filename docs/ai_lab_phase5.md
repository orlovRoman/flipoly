# AI Lab Phase 5 — безопасные LightGBM-адаптеры

Фаза 5 подключает реальные offline-операции к уже существующему AI Lab executor.

## Что выполняется

polyflip/ai_lab/lgbm_adapters.py регистрирует только три действия:

- TRAIN_MODEL — вызывает канонический CryptoModelTrainer с save_settings=False и activate_after_train=False. Кандидаты сохраняются в ModelRegistry неактивными. После обучения создаётся content-addressed AIModelArtifact с идентификаторами всех сохранённых режимов.
- RUN_OOT_BACKTEST — читает метрики сохранённых строк ModelRegistry; повторного обучения нет.
- RUN_POLYMARKET_OOT — читает только метаданные target_source=POLYMARKET_FINAL_OUTCOME и backtest_pnl_mode=POLYMARKET_OOF, затем агрегирует сохранённые варианты OUTSIDER_ONLY, FAVORITE_ONLY или COMBINED. Binance-return и legacy PnL отбрасываются.

polyflip/ai_lab/lgbm_worker.py — ограниченная точка запуска:

    outcomes = await execute_lgbm_steps(session, run_id, max_steps=1)

Один вызов выполняет не более указанного числа очередных шагов и сохраняет ExperimentResult и аудит через общий executor. Подключение worker не меняет активные модели, RuntimeSettings, live-политику или ордера.

## Безопасность

- Допустимы только модельные семьи LGBM, LIGHTGBM, CRYPTO_LGBM.
- Активная строка ModelRegistry после AI Lab-тренировки считается нарушением безопасности и приводит к ошибке.
- AI-конфигурация не записывается в FK старой таблицы LGBMExperimentConfig; её хэш и состав сохраняются в AIModelArtifact.metadata.
- Polymarket-OOT не подменяет отсутствие котировки проигрышем: покрытие и причины остаются в метриках.
- Адаптеры не импортируют gateway, worker live-исполнения или RuntimeSettings.

## Порядок запуска

1. Создать AI Lab AIExperimentConfig и план из TRAIN_MODEL, RUN_OOT_BACKTEST, RUN_POLYMARKET_OOT.
2. Worker вызывает execute_lgbm_steps ограниченными батчами.
3. Проверить /api/ai-lab/runs/{id}: результаты содержат TRAIN, OOT, POLYMARKET_OOT, а аудиты — ошибки и причины.
4. Только после отдельного SHADOW-периода человек принимает решение об активации; фаза 5 активацию не выполняет.
