import re

with open('tests/test_live_session.py', encoding='utf-8') as f:
    content = f.read()

new_content = content.replace(
    'mock_client.get_market_prices.return_value = {"best_ask": 0.5, "best_bid": 0.5}',
    'from unittest.mock import AsyncMock\n    mock_client.get_market_prices = AsyncMock(return_value={"best_ask": 0.5, "best_bid": 0.5})'
)

if new_content != content:
    with open('tests/test_live_session.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Patched test_live_session.py yet again")
