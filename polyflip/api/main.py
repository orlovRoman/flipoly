from fastapi import FastAPI, Depends
import structlog
from polyflip.api.auth import verify_api_key

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

app = FastAPI(title="PolyFlip API", version="0.1.0")

@app.on_event("startup")
async def startup_event():
    logger.info("application_startup")

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
