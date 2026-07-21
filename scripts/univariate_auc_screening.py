"""
Унивариантный скрининг AUC (Univariate AUC Screening) для всех криптофичей по всем символам.
"""
import asyncio
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from polyflip.db.connection import async_session
from polyflip.crypto.candle_repository import get_recent_candles
from polyflip.crypto.feature_builder import build_features, CRYPTO_FEATURE_COLUMNS
from polyflip.crypto.trainer import _build_target, CRYPTO_FEATURES

SYMBOLS = ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "XRPUSDT", "SOLUSDT"]

async def run_univariate_auc():
    results = {feat: {} for feat in CRYPTO_FEATURE_COLUMNS}
    
    async with async_session() as session:
        for symbol in SYMBOLS:
            candles = await get_recent_candles(session, symbol, "15m", limit=10_000)
            if len(candles) < 500:
                print(f"⚠️ Недостаточно свечей для {symbol}: {len(candles)}")
                continue

            df = build_features(candles)
            df_filtered = _build_target(df)

            y = df_filtered["target"].values
            
            print(f"\n📊 Символ: {symbol} (всего объектов: {len(df_filtered)}, позитивных: {y.sum()})")
            
            for feat in CRYPTO_FEATURE_COLUMNS:
                if feat not in df_filtered.columns:
                    results[feat][symbol] = 0.5
                    continue

                vals = df_filtered[feat].values
                # Проверка на константность или NaN
                if np.isnan(vals).any() or len(np.unique(vals)) <= 1:
                    auc = 0.5
                else:
                    try:
                        auc_raw = roc_auc_score(y, vals)
                        auc = max(auc_raw, 1.0 - auc_raw)
                    except Exception as e:
                        auc = 0.5
                
                results[feat][symbol] = auc

    print("\n" + "=" * 90)
    print(" 🚀 РЕЗУЛЬТАТЫ СКУПЕР-СКРИНИНГА UNIVARIATE AUCДЛЯ ВСЕХ ФИЧ 🚀")
    print("=" * 90)
    
    summary = []
    for feat in CRYPTO_FEATURE_COLUMNS:
        scores = [results[feat].get(s, 0.5) for s in SYMBOLS]
        mean_auc = np.mean(scores)
        min_auc = np.min(scores)
        max_auc = np.max(scores)
        summary.append({
            "feature": feat,
            "mean_auc": mean_auc,
            "min_auc": min_auc,
            "max_auc": max_auc,
            **{s: results[feat].get(s, 0.5) for s in SYMBOLS}
        })
    
    summary_df = pd.DataFrame(summary).sort_values("mean_auc", ascending=False)
    
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 1000)
    
    print("\nПолная таблица результатов AUC (нормализованный max(auc, 1-auc)):")
    print(summary_df.to_string(index=False, formatters={"mean_auc": "{:.4f}".format, "min_auc": "{:.4f}".format, "max_auc": "{:.4f}".format, **{s: "{:.4f}".format for s in SYMBOLS}}))

    cut_features = summary_df[summary_df["mean_auc"] < 0.505]["feature"].tolist()
    keep_features = summary_df[summary_df["mean_auc"] >= 0.505]["feature"].tolist()

    print("\n" + "-" * 90)
    print(f"❌ К КАНДИДАТАМ НА УДАЛЕНИЕ (Mean AUC < 0.505) [{len(cut_features)} фич]:")
    for f in cut_features:
        row = summary_df[summary_df["feature"] == f].iloc[0]
        print(f"  - {f:<20} | Mean AUC: {row['mean_auc']:.4f} | Min: {row['min_auc']:.4f} | Max: {row['max_auc']:.4f}")

    print("\n" + "-" * 90)
    print(f"✅ ОСТАЮТСЯ В МОДЕЛИ (Mean AUC >= 0.505) [{len(keep_features)} фич]:")
    for f in keep_features:
        row = summary_df[summary_df["feature"] == f].iloc[0]
        print(f"  + {f:<20} | Mean AUC: {row['mean_auc']:.4f} | Min: {row['min_auc']:.4f} | Max: {row['max_auc']:.4f}")

if __name__ == "__main__":
    asyncio.run(run_univariate_auc())
