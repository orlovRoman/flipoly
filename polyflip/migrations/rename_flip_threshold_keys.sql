BEGIN;

-- 1. Обновляем глобальный FLIP_THRESHOLD значением из TRADE_FLIP_THRESHOLD
UPDATE runtime_settings
SET value = (SELECT value FROM runtime_settings WHERE key = 'TRADE_FLIP_THRESHOLD')
WHERE key = 'FLIP_THRESHOLD'
  AND EXISTS (SELECT 1 FROM runtime_settings WHERE key = 'TRADE_FLIP_THRESHOLD'
              AND value IS NOT NULL AND value != '');

-- 2. Удаляем старый глобальный ключ
DELETE FROM runtime_settings WHERE key = 'TRADE_FLIP_THRESHOLD';

-- 3. Переименовываем per-asset ключи ТОЛЬКО если целевой ключ НЕ существует
UPDATE runtime_settings AS old
SET key = REPLACE(old.key, 'TRADE_FLIP_THRESHOLD_', 'FLIP_THRESHOLD_')
WHERE old.key LIKE 'TRADE_FLIP_THRESHOLD_%'
  AND NOT EXISTS (
      SELECT 1 FROM runtime_settings AS existing
      WHERE existing.key = REPLACE(old.key, 'TRADE_FLIP_THRESHOLD_', 'FLIP_THRESHOLD_')
  );

-- 4. Удаляем оставшиеся старые per-asset ключи
DELETE FROM runtime_settings
WHERE key LIKE 'TRADE_FLIP_THRESHOLD_%';

-- 5. Проверка — должно быть 0 строк с TRADE_FLIP_THRESHOLD
SELECT COUNT(*) AS must_be_zero FROM runtime_settings
WHERE key LIKE 'TRADE_FLIP_THRESHOLD%';

-- 6. Итоговая картина
SELECT key, value FROM runtime_settings
WHERE key LIKE '%FLIP_THRESHOLD%' ORDER BY key;

COMMIT;
