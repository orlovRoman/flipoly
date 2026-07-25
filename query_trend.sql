SELECT key, value FROM runtime_settings 
WHERE key LIKE '%FAVORITE%' 
   OR key LIKE '%FLIP%' 
   OR key LIKE '%TREND%' 
   OR key LIKE '%MODE%' 
   OR key LIKE '%MARGIN%'
   OR key LIKE '%TIME%';
