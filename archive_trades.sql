-- Шаг 1: Создаём архивную таблицу и переносим старые данные
CREATE TABLE trade_history_legacy AS 
SELECT * FROM trade_history 
WHERE created_at < '2026-07-28 00:00:00+00';

-- Шаг 2: Проверяем сколько записей перенесено
SELECT COUNT(*) as archived FROM trade_history_legacy;

-- Шаг 3: Проверяем сколько останется в основной
SELECT COUNT(*) as will_remain FROM trade_history 
WHERE created_at >= '2026-07-28 00:00:00+00';
