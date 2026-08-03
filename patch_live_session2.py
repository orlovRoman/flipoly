import re

with open('tests/test_live_session.py', encoding='utf-8') as f:
    content = f.read()

# Replace any @patch("polyflip.execution.worker.build_execution_gateway")
# async def some_test(mock_build_gateway, db_session):
# with the two patches and the mock_client lines.

def repl(match):
    func_def = match.group(0)
    # Extract function name and arguments
    # func_def looks like: @patch("polyflip.execution.worker.build_execution_gateway")\nasync def test_name(mock_build_gateway, db_session):
    m = re.match(r'@patch\("polyflip\.execution\.worker\.build_execution_gateway"\)\s*async def (test_[a-zA-Z0-9_]+)\(mock_build_gateway,\s*db_session\):', func_def)
    if not m:
        return func_def
    
    test_name = m.group(1)
    return f'''@patch("polyflip.execution.worker.PolymarketClient")
@patch("polyflip.execution.worker.build_execution_gateway")
async def {test_name}(mock_build_gateway, mock_client_class, db_session):
    mock_client = mock_client_class.return_value
    mock_client.get_market_prices.return_value = {{"best_ask": 0.5, "best_bid": 0.5}}'''

new_content = re.sub(r'@patch\("polyflip\.execution\.worker\.build_execution_gateway"\)\s*async def test_[a-zA-Z0-9_]+\(mock_build_gateway,\s*db_session\):', repl, content)

if new_content != content:
    with open('tests/test_live_session.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Patched test_live_session.py")
