"""Container entrypoint for the independent AI research agent."""
from __future__ import annotations

import asyncio

from runner import main


if __name__ == "__main__":
    asyncio.run(main())
