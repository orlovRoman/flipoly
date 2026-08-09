"""
polyflip/scripts/compare_target_models.py

Скрипт прямого A/B сравнения моделей LightGBM:
  - Контрольная модель A: Binance target (shift(-1) close > open)
  - Новая модель B: Polymarket target (LiveMarket/MarketSnapshot final_outcome, alignment pm_window_v1)

Оценка выполняется строго на едином Out-Of-Time (OOT) периоде (последние 20% рынков) против реальных исходов Polymarket.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import structlog
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.db.connection import async_session
from polyflip.crypto.dataset import build_polymarket_training_dataset
from polyflip.crypto.trainer import CRYPTO_FEATURES
from polyflip.constants import ASSET_TO_BINANCE_SYMBOL

logger = structlog.get_logger(__name__)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        idx = binids == i
        if np.any(idx):
            bin_acc = np.mean(y_true[idx])
            bin_conf = np.mean(y_prob[idx])
            bin_size = np.sum(idx)
            ece += (bin_size / n) * np.abs(bin_acc - bin_conf)
    return float(ece)


def evaluate_confidence_buckets(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, dict[str, float]]:
    max_conf = np.maximum(y_prob, 1.0 - y_prob)
    pred_class = (y_prob >= 0.5).astype(int)
    is_correct = (pred_class == y_true).astype(int)
    
    buckets = {
        "50–52%": (0.50, 0.52),
        "52–55%": (0.52, 0.55),
        "55–60%": (0.55, 0.60),
        "60–65%": (0.60, 0.65),
        ">65%":   (0.65, 1.01),
    }
    res = {}
    for b_name, (lo, hi) in buckets.items():
        idx = (max_conf >= lo) & (max_conf < hi)
        cnt = int(np.sum(idx))
        if cnt > 0:
            acc = float(np.mean(is_correct[idx]) * 100.0)
        else:
            acc = 0.0
        res[b_name] = {"count": cnt, "accuracy_pct": round(acc, 2)}
    return res


async def run_ab_comparison_for_asset(db: AsyncSession, symbol: str):
    print(f"\n==================================================================")
    print(f"=== RUNNING A/B TARGET COMPARISON FOR {symbol} ===")
    print(f"==================================================================")

    # 1. Загрузка чистого датасета (pm_window_v1)
    dataset = await build_polymarket_training_dataset(db, symbol)
    if dataset.empty or len(dataset) < 300:
        print(f"⚠️ Недостаточно данных для A/B тестирования по {symbol}: {len(dataset)} рынков")
        return

    # Сортировка по времени
    dataset = dataset.sort_values("market_start").reset_index(drop=True)
    
    # 2. Разделение на Train (80%) и Out-Of-Time Test (20%)
    split_idx = int(len(dataset) * 0.8)
    train_df = dataset.iloc[:split_idx].copy()
    test_df = dataset.iloc[split_idx:].copy()
    
    print(f"Всего уникальных рынков: {len(dataset)} | Train: {len(train_df)} | OOT Test: {len(test_df)}")

    X_train = train_df[CRYPTO_FEATURES].fillna(0.0).values
    y_train_poly = train_df["target"].values

    X_test = test_df[CRYPTO_FEATURES].fillna(0.0).values
    y_test_poly = test_df["target"].values  # Истинный таргет Polymarket

    # 3. Модель A (Контроль - предсказание Binance ret_1 > 0)
    # Для контрольной модели вычисляем следующий ret_1 Binance на train
    if "ret_1" in train_df.columns:
        y_train_binance = (train_df["ret_1"].shift(-1) > 0).astype(int).fillna(0).values
    else:
        y_train_binance = y_train_poly

    model_a = LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, num_leaves=15, random_state=42, verbose=-1)
    model_a.fit(X_train, y_train_binance)
    y_prob_a = model_a.predict_proba(X_test)[:, 1]

    # 4. Модель B (Новая - предсказание Polymarket final_outcome pm_window_v1)
    model_b = LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, num_leaves=15, random_state=42, verbose=-1)
    model_b.fit(X_train, y_train_poly)
    y_prob_b = model_b.predict_proba(X_test)[:, 1]

    # 5. Метрики для обеих моделей ПРОТИВ POLYMARKET OUTCOME на OOT тесте
    maj_baseline = float(np.maximum(np.mean(y_test_poly), 1.0 - np.mean(y_test_poly)) * 100.0)

    # Model A Metrics
    acc_a = float(accuracy_score(y_test_poly, (y_prob_a >= 0.5).astype(int)) * 100.0)
    auc_a = float(roc_auc_score(y_test_poly, y_prob_a))
    brier_a = float(brier_score_loss(y_test_poly, y_prob_a))
    ece_a = compute_ece(y_test_poly, y_prob_a)
    buckets_a = evaluate_confidence_buckets(y_test_poly, y_prob_a)

    # Model B Metrics
    acc_b = float(accuracy_score(y_test_poly, (y_prob_b >= 0.5).astype(int)) * 100.0)
    auc_b = float(roc_auc_score(y_test_poly, y_prob_b))
    brier_b = float(brier_score_loss(y_test_poly, y_prob_b))
    ece_b = compute_ece(y_test_poly, y_prob_b)
    buckets_b = evaluate_confidence_buckets(y_test_poly, y_prob_b)

    print(f"\nMajority Baseline Accuracy: {maj_baseline:.2f}%")
    print("-" * 65)
    print(f"{'МЕТРИКА':<22} | {'МОДЕЛЬ A (BINANCE TARGET)':<24} | {'МОДЕЛЬ B (POLYMARKET TARGET)':<24}")
    print("-" * 75)
    print(f"{'OOT Accuracy %':<22} | {acc_a:.2f}%{'':<18} | {acc_b:.2f}%")
    print(f"{'OOT AUC':<22} | {auc_a:.4f}{'':<18} | {auc_b:.4f}")
    print(f"{'OOT Brier Score':<22} | {brier_a:.4f}{'':<18} | {brier_b:.4f}")
    print(f"{'OOT ECE (Calibration)':<22} | {ece_a:.4f}{'':<18} | {ece_b:.4f}")
    
    print("\n--- ТОЧНОСТЬ ПО ДИАПАЗОНАМ УВЕРЕННОСТИ (OOT TEST) ---")
    print(f"{'УВЕРЕННОСТЬ':<15} | {'МОДЕЛЬ A (Count / Acc %)':<25} | {'МОДЕЛЬ B (Count / Acc %)':<25}")
    print("-" * 70)
    for b_name in buckets_a:
        ca, aa = buckets_a[b_name]["count"], buckets_a[b_name]["accuracy_pct"]
        cb, ab = buckets_b[b_name]["count"], buckets_b[b_name]["accuracy_pct"]
        print(f"{b_name:<15} | {ca:<5} / {aa:.2f}%{'':<11} | {cb:<5} / {ab:.2f}%")


async def main():
    async with async_session() as session:
        for asset, symbol in ASSET_TO_BINANCE_SYMBOL.items():
            await run_ab_comparison_for_asset(session, symbol)

if __name__ == "__main__":
    asyncio.run(main())
