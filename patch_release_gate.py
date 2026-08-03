import re

with open('tests/test_release_gate.py', encoding='utf-8') as f:
    content = f.read()

# Add the import for patch and AsyncMock if not there
if 'from unittest.mock import patch' not in content:
    content = "from unittest.mock import patch, AsyncMock\n" + content

# we need to replace all async def test_ with a patched version
def repl(match):
    test_name = match.group(1)
    args = match.group(2)
    # If it's already patched, don't patch again
    if 'mock_client_class' in args:
        return match.group(0)
    
    return f'''@pytest.mark.asyncio
@patch('polyflip.collector.client.PolymarketClient')
async def {test_name}(mock_client_class, {args}):
    mock_client = mock_client_class.return_value
    mock_client.get_market_prices = AsyncMock(return_value={{"best_ask": 0.5, "best_bid": 0.5}})'''

new_content = re.sub(r'@pytest\.mark\.asyncio\nasync def (test_[a-zA-Z0-9_]+)\((.*?)\):', repl, content)

if new_content != content:
    with open('tests/test_release_gate.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print('Patched test_release_gate.py')
