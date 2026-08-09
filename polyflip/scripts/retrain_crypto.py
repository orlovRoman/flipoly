"""
polyflip/scripts/retrain_crypto.py — Последовательное переобучение моделей в фоновом режиме
с низким приоритетом и паузами между символами для предотвращения нагрузки на ЦП.
"""
import asyncio
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("background_retrain")

from polyflip.db.connection import async_session
from polyflip.crypto.trainer import CryptoModelTrainer

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]

async def run_sequential_retrain():
    logger.info("Starting background retraining for symbols: %s", SYMBOLS)
    results = {}
    
    for symbol in SYMBOLS:
        logger.info(">>> Beginning retrain for %s...", symbol)
        try:
            async with async_session() as session:
                trainer = CryptoModelTrainer(session)
                ok = await trainer.train(symbol, interval="15m")
                results[symbol] = "SUCCESS" if ok else "QUALITY_GATE_FAILED_OR_EMPTY"
                logger.info(">>> Retrain finished for %s: status=%s", symbol, results[symbol])
        except Exception as exc:
            logger.exception("Error retraining %s: %s", symbol, str(exc))
            results[symbol] = f"ERROR: {str(exc)}"
        
        # Небольшая пауза между символами для снижения пиковой нагрузки
        await asyncio.sleep(3)

    logger.info("=== RETRAINING SUMMARY ===")
    for sym, res in results.items():
        logger.info("  %-6s: %s", sym, res)

if __name__ == "__main__":
    asyncio.run(run_sequential_retrain())
