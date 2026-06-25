from fastapi import FastAPI, Depends
from fastapi.staticfiles import StaticFiles
import structlog
import os
from polyflip.api.auth import verify_api_key
from polyflip.api.analytics import router as analytics_router
from polyflip.api.dashboard import router as dashboard_router

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("application_startup")
    yield
    # Cleanup here if needed

app = FastAPI(title="PolyFlip API", version="0.1.0", lifespan=lifespan)
app.include_router(analytics_router)
app.include_router(dashboard_router)

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
