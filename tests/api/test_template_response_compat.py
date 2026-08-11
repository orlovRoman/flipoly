import pytest
from starlette.datastructures import URLPath
from starlette.requests import Request

from polyflip.api.backtest_api import backtest_page
from polyflip.api.dashboard import get_dashboard, get_execution_dashboard
from polyflip.api.trading_dashboard import get_trading_dashboard


class _Router:
    def url_path_for(self, name, **params):
        assert name == "static"
        return URLPath(f"/{params['path'].lstrip('/')}")


def _request(path: str) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "root_path": "",
        "scheme": "http",
        "server": ("testserver", 80),
        "client": ("testclient", 1234),
        "headers": [],
        "query_string": b"",
        "router": _Router(),
    }
    return Request(scope)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("handler", "path", "template"),
    [
        (get_dashboard, "/dashboard", "index.html"),
        (get_execution_dashboard, "/execution", "execution.html"),
        (backtest_page, "/backtest", "backtest.html"),
        (get_trading_dashboard, "/trading", "trading.html"),
    ],
)
async def test_template_routes_use_current_starlette_signature(handler, path, template):
    response = await handler(_request(path))
    assert response.template.name == template
