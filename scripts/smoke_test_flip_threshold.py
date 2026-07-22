"""
Smoke-тест после деплоя Варианта B:
  python scripts/smoke_test_flip_threshold.py
"""
import asyncio
from polyflip.db.connection import async_session
from polyflip.trading.settings_loader import load_trading_settings
from polyflip.trading.decision_runners import _get_float_setting


async def main():
    async with async_session() as db:
        raw = await load_trading_settings(db)
        
        # Проверка 1: старый ключ отсутствует
        old_val = raw.get("TRADE_FLIP_THRESHOLD")
        assert old_val is None, f"❌ Старый ключ TRADE_FLIP_THRESHOLD ещё в raw_settings: {old_val}"
        print("✅ 1. TRADE_FLIP_THRESHOLD отсутствует в raw_settings")

        # Проверка 2: новый ключ читается
        new_val = _get_float_setting(raw, "FLIP_THRESHOLD")
        assert new_val is not None, "❌ FLIP_THRESHOLD не найден в raw_settings (нет в БД?)"
        assert 0.0 < new_val <= 1.0, f"❌ FLIP_THRESHOLD={new_val} вне диапазона [0, 1]"
        print(f"✅ 2. FLIP_THRESHOLD={new_val} читается корректно")

        # Проверка 3: деления на 100 нет
        raw_test = {"X": "0.8"}
        assert _get_float_setting(raw_test, "X") == 0.8, "❌ _get_float_setting делит на 100"
        print("✅ 3. _get_float_setting не делит на 100")

        # Проверка 4: per-asset ключи читаются если есть
        for key in raw:
            if key.startswith("FLIP_THRESHOLD_") and not key.startswith("AUTO_"):
                val = _get_float_setting(raw, key)
                print(f"✅ 4. Per-asset {key}={val}")

        print("\n✅ Все проверки пройдены. Вариант B активен.")


if __name__ == "__main__":
    asyncio.run(main())
