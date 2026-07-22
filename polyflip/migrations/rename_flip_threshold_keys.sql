BEGIN;

-- 1. Обновляем значение FLIP_THRESHOLD значением из TRADE_FLIP_THRESHOLD (0.8)
UPDATE runtime_settings
SET value = (SELECT value FROM runtime_settings WHERE key = 'TRADE_FLIP_THRESHOLD')
WHERE key = 'FLIP_THRESHOLD'
  AND EXISTS (SELECT 1 FROM runtime_settings WHERE key = 'TRADE_FLIP_THRESHOLD' AND value IS NOT NULL AND value != '');

-- 2. Удаляем старый глобальный ключ TRADE_FLIP_THRESHOLD
DELETE FROM runtime_settings WHERE key = 'TRADE_FLIP_THRESHOLD';

-- 3. Для per-asset ключей: переименовываем TRADE_FLIP_THRESHOLD_* в FLIP_THRESHOLD_*
DELETE FROM runtime_settings WHERE key IN ('FLIP_THRESHOLD_BTC', 'FLIP_THRESHOLD_ETH', 'FLIP_THRESHOLD_XRP', 'FLIP_THRESHOLD_SOL', 'FLIP_THRESHOLD_DOGE');

UPDATE runtime_settings
SET key = REPLACE(key, 'TRADE_FLIP_THRESHOLD_', 'FLIP_THRESHOLD_')
WHERE key LIKE 'TRADE_FLIP_THRESHOLD_%';

-- 4. Проверка — должно быть 0 строк с TRADE_FLIP_THRESHOLD
SELECT COUNT(*) AS must_be_zero
FROM runtime_settings
WHERE key LIKE 'TRADE_FLIP_THRESHOLD%';

-- 5. Итоговая картина
SELECT key, value
FROM runtime_settings
WHERE key LIKE '%FLIP_THRESHOLD%'
ORDER BY key;

COMMIT;
