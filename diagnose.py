import asyncio
from polyflip.db.connection import async_session
from polyflip.crypto.predictor import CryptoPredictor
from polyflip.crypto.candle_repository import get_recent_candles
from polyflip.crypto.feature_builder import build_crypto_features, CRYPTO_FEATURE_COLUMNS

async def diagnose():
    async with async_session() as session:
        predictor = CryptoPredictor()
        for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
            print(f"\n=== Диагностика {symbol} ===")
            ok = await predictor.load(session, symbol)
            if not ok:
                print(f"Не удалось загрузить модели для {symbol}")
                continue
            
            # Режимы волатильности и пороги
            regimes = ["low_vol", "mid_vol", "high_vol"]
            print("Пороги по режимам:")
            for reg in regimes:
                th = predictor._thresholds.get(symbol, {}).get(reg, (None, None))
                ver = predictor._model_versions.get(symbol, {}).get(reg, None)
                print(f"  {reg:8s} (v{ver}): up={th[0]}, down={th[1]}")
            print(f"Границы волатильности: p33={predictor._vol_p33s.get(symbol):.4f}, p67={predictor._vol_p67s.get(symbol):.4f}")

            # Последние свечи и предсказание
            candles = await get_recent_candles(session, symbol, "15m", limit=150)
            if not candles:
                print("Свечи отсутствуют в БД!")
                continue
            
            print(f"Загружено свечей: {len(candles)}. Последняя закрытая свеча: open_time={candles[-1].open_time}, close={candles[-1].close}")
            
            # Посмотрим на фичи последнего шага
            feat_vec = build_crypto_features(candles)
            if not feat_vec.valid:
                print("Фичи невалидны!")
                continue
                
            fv_dict = dict(zip(CRYPTO_FEATURE_COLUMNS, feat_vec.features[0]))
            print("Основные фичи:")
            print(f"  vol_ratio: {fv_dict.get('vol_ratio'):.4f}")
            print(f"  ret_1:     {fv_dict.get('ret_1'):.4f}")
            print(f"  ret_12:    {fv_dict.get('ret_12'):.4f}")
            print(f"  rsi_14:    {fv_dict.get('rsi_14'):.2f}")
            
            # Определим текущий режим
            vol_ratio = fv_dict.get("vol_ratio", 1.0)
            vol_p33 = predictor._vol_p33s.get(symbol, 0.5)
            vol_p67 = predictor._vol_p67s.get(symbol, 1.5)
            if vol_ratio <= vol_p33:
                curr_reg = "low_vol"
            elif vol_ratio <= vol_p67:
                curr_reg = "mid_vol"
            else:
                curr_reg = "high_vol"
            print(f"Текущий режим волатильности: {curr_reg} (vol_ratio={vol_ratio:.4f})")
            
            # Предсказание
            signal = predictor.predict(candles, symbol)
            print(f"Результат инференса:")
            print(f"  p_up:      {signal.p_up:.4f}")
            print(f"  direction: {signal.direction}")
            print(f"  edge:      {signal.edge:.4f}")
            print(f"  features_ok: {signal.features_ok}")
            
            # Посмотрим историю p_up на последних 10 свечах (сдвигаем окно)
            print("История предсказаний (последние 5 свечей):")
            for i in range(-4, 1):
                idx = len(candles) + i
                if idx > 110:
                    sub_candles = candles[:idx]
                    sig = predictor.predict(sub_candles, symbol)
                    print(f"  time={sub_candles[-1].open_time}  close={sub_candles[-1].close:.2f}  p_up={sig.p_up:.4f}  dir={sig.direction}")

asyncio.run(diagnose())
