import re

with open('tests/test_live_session.py', encoding='utf-8') as f:
    content = f.read()

new_content = content.replace(
    '@patch("polyflip.execution.worker.build_execution_gateway")\n    async def test_fak_no_liquidity_releases_reservation(mock_build_gateway, db_session):',
    '@patch("polyflip.execution.worker.PolymarketClient")\n    @patch("polyflip.execution.worker.build_execution_gateway")\n    async def test_fak_no_liquidity_releases_reservation(mock_build_gateway, mock_client_class, db_session):\n        mock_client = mock_client_class.return_value\n        mock_client.get_market_prices.return_value = {"best_ask": 0.5, "best_bid": 0.5}'
)

new_content = new_content.replace(
    '@patch("polyflip.execution.worker.build_execution_gateway")\n    async def test_live_execution_happy_path(mock_build_gateway, db_session):',
    '@patch("polyflip.execution.worker.PolymarketClient")\n    @patch("polyflip.execution.worker.build_execution_gateway")\n    async def test_live_execution_happy_path(mock_build_gateway, mock_client_class, db_session):\n        mock_client = mock_client_class.return_value\n        mock_client.get_market_prices.return_value = {"best_ask": 0.5, "best_bid": 0.5}'
)

new_content = new_content.replace(
    '@patch("polyflip.execution.worker.build_execution_gateway")\n    async def test_live_execution_fails_with_gateway_error(mock_build_gateway, db_session):',
    '@patch("polyflip.execution.worker.PolymarketClient")\n    @patch("polyflip.execution.worker.build_execution_gateway")\n    async def test_live_execution_fails_with_gateway_error(mock_build_gateway, mock_client_class, db_session):\n        mock_client = mock_client_class.return_value\n        mock_client.get_market_prices.return_value = {"best_ask": 0.5, "best_bid": 0.5}'
)

if new_content != content:
    with open('tests/test_live_session.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Patched test_live_session.py")
