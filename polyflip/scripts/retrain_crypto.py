"""
Sequential background retraining with pauses between symbols to limit peak CPU load.
"""
import asyncio
import logging
from collections.abc import Sequence

from polyflip.crypto.trainer import CryptoModelTrainer
from polyflip.db.connection import async_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("background_retrain")

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT")


async def run_sequential_retrain(
    symbols: Sequence[str] = SYMBOLS,
    pause_seconds: float = 3,
) -> dict[str, str]:
    symbols = tuple(symbols)
    logger.info("Starting background retraining for symbols: %s", symbols)
    results: dict[str, str] = {}

    for index, symbol in enumerate(symbols):
        logger.info(">>> Beginning retrain for %s...", symbol)
        try:
            async with async_session() as session:
                trainer = CryptoModelTrainer(session)
                ok = await trainer.train(
                    symbol, interval="15m", activate_after_train=True
                )
                results[symbol] = "COMPLETED" if ok else "FAILED_OR_EMPTY"
                logger.info(
                    ">>> Retrain finished for %s: status=%s",
                    symbol,
                    results[symbol],
                )
        except Exception as exc:
            logger.exception("Error retraining %s: %s", symbol, str(exc))
            results[symbol] = f"ERROR: {exc}"

        if pause_seconds > 0 and index < len(symbols) - 1:
            await asyncio.sleep(pause_seconds)

    logger.info("=== RETRAINING SUMMARY ===")
    for symbol, result in results.items():
        logger.info("  %-8s: %s", symbol, result)
    return results


def exit_code_for(results: dict[str, str]) -> int:
    """Return a non-zero process status when any symbol did not complete."""
    return 0 if results and all(value == "COMPLETED" for value in results.values()) else 1


def main() -> int:
    return exit_code_for(asyncio.run(run_sequential_retrain()))


if __name__ == "__main__":
    raise SystemExit(main())
