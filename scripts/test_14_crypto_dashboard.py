import os
import asyncio
# Задаем DATABASE_URL до импорта config/app
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"

from httpx import AsyncClient, ASGITransport
from polyflip.api.main import app
from polyflip.db.connection import engine
from polyflip.db.models import Base

async def test_endpoints():
    os.environ["API_KEY"] = "test-key"
    headers = {"X-API-Key": "test-key"}
    
    # Инициализируем БД
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Тест страницы дашборда
        resp = await client.get("/crypto")
        assert resp.status_code == 200, f"Dashboard page: {resp.status_code}"
        assert "Crypto LightGBM Dashboard" in resp.text
        print("✅ GET /crypto OK")

        # Тест статуса модели (без ключа должен вернуть 401)
        resp = await client.get("/crypto/api/crypto/status")
        assert resp.status_code == 401, f"Status without auth: {resp.status_code}"
        print("✅ Auth protection OK")

        # Тест статуса модели с авторизацией
        resp = await client.get("/crypto/api/crypto/status", headers=headers)
        assert resp.status_code == 200, f"Status with auth: {resp.status_code}"
        data = resp.json()
        assert "models" in data
        assert "symbols" in data
        assert "settings" in data
        print("✅ GET /crypto/api/crypto/status OK")

        # Тест сохранения настроек
        payload = {
            "n_estimators": 150,
            "learning_rate": 0.08
        }
        resp = await client.post("/crypto/api/crypto/settings", json=payload, headers=headers)
        assert resp.status_code == 200, f"Save settings: {resp.status_code}"
        print("✅ POST /crypto/api/crypto/settings OK")

        # Тест бэктеста
        resp = await client.get("/crypto/api/crypto/backtest?symbol=BTCUSDT", headers=headers)
        # Бэктест может вернуть ошибку, если нет свечей в БД, но сам эндпоинт должен обработать это корректно (код 200)
        assert resp.status_code == 200, f"Backtest status: {resp.status_code}"
        bt_data = resp.json()
        assert "error" in bt_data or "n_trades" in bt_data
        print("✅ GET /crypto/api/crypto/backtest OK")

    print("🎉 Все новые эндпоинты дашборда работают и авторизованы корректно!")

asyncio.run(test_endpoints())
