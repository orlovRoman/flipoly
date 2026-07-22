BEGIN;

-- 1. Снижаем NO_MIN_EDGE с 0.3 (30%) до 0.04 (4%) для адекватного фильтра аутсайдеров
UPDATE runtime_settings
SET value = '0.04'
WHERE key = 'NO_MIN_EDGE';

-- 2. Корректируем OUTSIDER_MAX_PRICE до 0.48 (для создания зазора [0.48, 0.50])
UPDATE runtime_settings
SET value = '0.48'
WHERE key = 'OUTSIDER_MAX_PRICE';

SELECT key, value
FROM runtime_settings
WHERE key IN ('NO_MIN_EDGE', 'OUTSIDER_MAX_PRICE', 'FAVORITE_MIN_PRICE', 'MIN_EDGE')
ORDER BY key;

COMMIT;
