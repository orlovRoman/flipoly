import sys
from polyflip.api.main import app

for route in app.routes:
    print(getattr(route, "methods", None), getattr(route, "path", route.name))
