# 🏆 Отчёт об обновлении таблицы моделей Polymarket

Колонка «ТИП АЛГОРИТМА» удалена из UI интерфейса управления моделями.

---

## 🛠️ Что сделано:
1. В `polyflip/templates/index.html` удален заголовок таблицы `<th data-sort="algorithm">Тип алгоритма</th>`.
2. В `polyflip/static/js/app.js` удален рендеринг однотипных плашек `🟧 LogReg`.
3. Таблица моделей стала более компактной и сфокусированной на метриках точности (`Accuracy`, `Baseline`, `Lift`, `ECE`, `PnL`).

---

## 🧪 Деплой
* Изменения проверены и отправлены в Git (`refactor(ui): remove redundant algorithm type column from models table`).
* Контейнеры на сервере `34.50.54.183` перезапущены.
