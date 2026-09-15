# REVIEW_FINDINGS — 2026-09-15

## Статус
- direction CLOSEOUT: INVALIDATED_PENDING_RERUN
- feature CLOSEOUT: INVALIDATED_PENDING_RERUN
- shared_dataset: CONDITIONAL_PASS (построение причинное, gate формальный)

## Дефекты E4-E6 (подтверждено кодом)
- E4: `D:\lgbm-audit-v1\code\direction\e4_mapping.py:1` — 3 строки вручную, без `era_voting` (`combined_voting_arch0803.py:73` etc.)
- E5: `e5_paired.py:10` `R2_side = "SKIP"`, `delta = 0.0` — все политики `SKIP`
- E6: `e6_economics.py:12` `means.append(0.0)` → `CI [0,0]` — не парный расчет

## Дефекты F1-F7
- F1 `FEATURE_CATALOG.csv:1` — 10 строк вместо 80
- F3 `FEATURE_CORRELATIONS.parquet` — 0 bytes, не открывается PyArrow
- F gate `FEATURE_VERDICTS.json:1` — только `ret_1`
- Все F-файлы по 1 строке

## Ограничения E1-E3
- 1 497 818 строк / 21 757 рынков (медиана 56, макс 288, 1 496 905 повторяющихся прогнозов)
- E1: 73 версии `DONE` при 16 x 0 пар, 35 x 0% match, 1010 `MISSING_FEATURES`
- E2: `legacy_native_for()` hardcode 15m
- synthetic_no 99.97% (диагностика, не исполнение)
- Gate `leakage_zero_by_construction = true` — формальность
- Git: direction только E1-E3, feature совпадает с shared

## Решение
Отклонить E4-E6/F/CLOSEOUT, сохранить shared_dataset как основу, перереализовать по шагам 1-24. Текущие заглушки не публиковать.
