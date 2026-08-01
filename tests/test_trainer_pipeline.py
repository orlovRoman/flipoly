import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def test_trainer_pipeline_serialization():
    # Эмулируем данные с разными масштабами
    X = pd.DataFrame({
        "time_left_min": [5.0, 2.0, 10.0],
        "mid_price":     [0.7, 0.3, 0.5],
        "volume_5min":   [8000.0, 100.0, 4000.0],
    })
    y = pd.Series([1, 0, 1])

    # Создаем и обучаем pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()), 
        ("model", LogisticRegression(class_weight="balanced", random_state=42))
    ])
    pipeline.fit(X, y)

    # Проверяем сериализацию
    model_bytes = pickle.dumps(pipeline)
    loaded = pickle.loads(model_bytes)
    assert isinstance(loaded, Pipeline), "Сохранён не Pipeline!"

    # Проверяем, что predict_proba работает без ручного скейлинга
    proba = loaded.predict_proba(X)
    assert proba.shape == (3, 2), "Неверная форма вывода predict_proba"
    assert all(0 <= p <= 1 for row in proba for p in row), "Вероятности вне [0,1]"
