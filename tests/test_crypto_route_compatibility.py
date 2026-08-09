"""
tests/test_crypto_route_compatibility.py

Тесты сохранения метода, заголовков и пути при запросах /crypto/* -> /lightgbm/*
через CryptoRouteRewriteMiddleware.
"""
import pytest
from polyflip.api.main import CryptoRouteRewriteMiddleware


@pytest.mark.asyncio
async def test_old_post_crypto_train_preserves_method_and_replaces_path():
    called_scope = {}

    async def dummy_app(scope, receive, send):
        nonlocal called_scope
        called_scope = scope

    middleware = CryptoRouteRewriteMiddleware(dummy_app)
    scope = {"type": "http", "method": "POST", "path": "/crypto/api/train"}

    await middleware(scope, None, None)

    assert called_scope["method"] == "POST"
    assert called_scope["path"] == "/lightgbm/api/train"
