import asyncio
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, Request
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import structlog
import os

from polyflip.api.auth import verify_api_key
from polyflip.api.analytics import router as analytics_router
from polyflip.api.dashboard import router as dashboard_router
from polyflip.api.trading_dashboard import router as trading_dashboard_router
from polyflip.api.settings import router as settings_router
from polyflip.api.slippage import router as slippage_router
from polyflip.api.backtest_api import router as backtest_router
from polyflip.api.crypto_dashboard import router as crypto_router
from polyflip.api.crypto_backtest_api import router as crypto_backtest_router
from polyflip.config import settings

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

class SimpleRateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limit: int = 60, window: int = 60):
        super().__init__(app)
        self.limit = limit
        self.window = window
        self.requests = defaultdict(list)
        self._lock = asyncio.Lock()
        self.request_count = 0

    async def dispatch(self, request: Request, call_next):
        # Пропускаем статические файлы и проверку здоровья /health
        if request.url.path.startswith("/static") or request.url.path == "/health":
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        async with self._lock:
            # Периодическая полная очистка словаря от старых IP (каждые 1000 запросов)
            self.request_count += 1
            if self.request_count % 1000 == 0:
                for ip in list(self.requests.keys()):
                    self.requests[ip] = [t for t in self.requests[ip] if now - t < self.window]
                    if not self.requests[ip]:
                        self.requests.pop(ip, None)

            # Очищаем запросы за пределами окна
            self.requests[client_ip] = [t for t in self.requests[client_ip] if now - t < self.window]

            if len(self.requests[client_ip]) >= self.limit:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Too many requests. Rate limit exceeded."}
                )

            self.requests[client_ip].append(now)

        response = await call_next(request)
        return response

from polyflip.db.connection import async_session
from polyflip.db.init_runtime_settings import seed_runtime_settings, migrate_auto_dead_zone_width, migrate_stop_loss_pct

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("application_startup")
    if settings.API_KEY == "test-key":
        logger.warning("API key is set to insecure default 'test-key'. Please change it in production.")
    
    async with async_session() as session:
        # Сначала миграция (переименование AUTO_DEAD_ZONE_WIDTH → DEAD_ZONE_WIDTH)
        await migrate_auto_dead_zone_width(session)
        # Миграция: STOP_LOSS_PCT → STOP_LOSS_PCT_FAVORITE + STOP_LOSS_PCT_OUTSIDER
        await migrate_stop_loss_pct(session)
        # Потом посев дефолтов для новых ключей
        await seed_runtime_settings(session)
        
    yield

app = FastAPI(title="PolyFlip API", version="0.1.0", lifespan=lifespan)

# Подключаем middleware ограничения частоты запросов
app.add_middleware(SimpleRateLimitMiddleware, limit=200, window=60)

app.include_router(analytics_router)
app.include_router(dashboard_router)
app.include_router(trading_dashboard_router)
app.include_router(settings_router)
app.include_router(slippage_router)
app.include_router(backtest_router)
app.include_router(crypto_router)
app.include_router(crypto_backtest_router)

# Подключение статических файлов
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app.mount("/static", StaticFiles(directory=os.path.join(base_dir, "static")), name="static")

@app.get("/health")
async def health_check():
    """Health check endpoint, open to public."""
    return {"status": "ok"}

@app.get("/stats/{asset}", dependencies=[Depends(verify_api_key)])
async def get_stats(asset: str):
    """Stub for stats endpoint."""
    return {
        "asset": asset,
        "bins": []
    }
