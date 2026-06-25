import asyncio
import pandas as pd
from sqlalchemy import select
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

from polyflip.db.connection import async_session
from polyflip.db.models import MarketSnapshot, LiveMarket

async def main():
    async with async_session() as db:
        print("--- СБОР ДАННЫХ ДЛЯ ОБУЧЕНИЯ (BTC) ---")
        stmt = select(MarketSnapshot).where(
            MarketSnapshot.asset == "BTC",
            MarketSnapshot.final_outcome != "PENDING"
        )
        result = await db.execute(stmt)
        snapshots = result.scalars().all()
        
        print(f"Найдено {len(snapshots)} завершенных рынков (снепшотов) для BTC.")
        if len(snapshots) < 10:
            print("Слишком мало данных для теста!")
            return

        data = []
        for s in snapshots:
            data.append({
                "time_left_min": float(s.time_left_min),
                "mid_price": float(s.mid_price),
                "spread": float(s.spread),
                "volume_5min": float(s.volume_5min),
                "target": 1 if s.flip_vs_final else 0
            })
            
        df = pd.DataFrame(data)
        X = df[["time_left_min", "mid_price", "spread", "volume_5min"]]
        y = df["target"]
        
        print(f"Баланс классов (Флип = 1): \n{y.value_counts()}")
        
        if len(y.unique()) < 2:
            print("В данных только один класс (например, вообще не было флипов или 100% флипов). Модель не обучится.")
            return

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression(class_weight="balanced", random_state=42)
        model.fit(X_train, y_train)
        
        val_preds = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds)
        print(f"Модель обучена! Точность на отложенной выборке: {val_acc:.2f}")
        
        print("\n--- ТЕСТОВАЯ ПРЕДСКАЗАТЕЛЬНАЯ СИЛА НА ЖИВЫХ РЫНКАХ ---")
        live_stmt = select(LiveMarket).where(LiveMarket.asset == "BTC")
        live_result = await db.execute(live_stmt)
        live_markets = live_result.scalars().all()
        
        if not live_markets:
            print("Нет активных рынков по BTC для теста.")
            return
            
        for lm in live_markets[:3]: # Возьмем первые 3 рынка
            import datetime
            time_left_sec = (lm.end_time_est.replace(tzinfo=datetime.timezone.utc) - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
            time_left_min = max(0, time_left_sec / 60.0)
            
            X_test = pd.DataFrame([{
                "time_left_min": float(time_left_min),
                "mid_price": float(lm.current_yes_price),
                "spread": float(lm.current_spread),
                "volume_5min": float(lm.volume_5min)
            }])
            
            prob = model.predict_proba(X_test)[0][1] # Вероятность класса 1 (ФЛИП)
            
            print(f"\nРынок: {lm.question}")
            print(f"Осталось мин: {time_left_min:.1f}, Текущая цена (mid): {lm.current_yes_price}")
            print(f"Вероятность флипа (по мнению модели): {prob * 100:.2f}%")

if __name__ == "__main__":
    asyncio.run(main())
