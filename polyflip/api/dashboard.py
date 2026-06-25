import os
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter(tags=["Dashboard"])

# Получаем абсолютный путь до папки templates, так как uvicorn может запускаться из разных мест
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(base_dir, "templates"))

@router.get("/dashboard")
async def get_dashboard(request: Request):
    """Отдает главную страницу дашборда"""
    return templates.TemplateResponse("index.html", {"request": request})
