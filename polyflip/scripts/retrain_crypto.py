"""Sequential candidate retraining with pauses between symbols.

Candidate retraining is intentionally inactive by default.  Live activation
requires a separate explicit action after the saved OOT report is reviewed.
"""

import argparse
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
    *,
    activate_after_train: bool = False,
) -> dict[str, str]:
    symbols = tuple(symbols)
    logger.info(
        "Starting background retraining for symbols: %s (activate_after_train=%s)",
        symbols,
        activate_after_train,
    )
    results: dict[str, str] = {}

    for index, symbol in enumerate(symbols):
        logger.info(">>> Beginning retrain for %s...", symbol)
        try:
            async with async_session() as session:
                trainer = CryptoModelTrainer(session)
                ok = await trainer.train(
                    symbol, interval="15m", activate_after_train=activate_after_train
                )
                results[symbol] = "COMPLETED" if ok else "FAILED_OR_EMPTY"
                logger.info(">>> Retrain finished for %s: status=%s", symbol, results[symbol])
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
    return 0 if results and all(value == "COMPLETED" for value in results.values()) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="*", default=SYMBOLS)
    parser.add_argument("--pause-seconds", type=float, default=3)
    parser.add_argument(
        "--activate-after-train",
        action="store_true",
        help="Explicitly allow Quality-Gate-passing models to replace active models",
    )
    args = parser.parse_args()
    return exit_code_for(asyncio.run(run_sequential_retrain(
        args.symbols,
        args.pause_seconds,
        activate_after_train=args.activate_after_train,
    )))


if __name__ == "__main__":
    raise SystemExit(main())
