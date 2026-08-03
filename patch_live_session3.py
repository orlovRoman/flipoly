import re

with open('tests/test_live_session.py', encoding='utf-8') as f:
    content = f.read()

def repl(match):
    func_def = match.group(0)
    m = re.match(r'@patch\("polyflip\.execution\.worker\.PolymarketClient"\)\n@patch\("polyflip\.execution\.worker\.build_execution_gateway"\)\nasync def (test_[a-zA-Z0-9_]+)\(mock_build_gateway,\s*mock_client_class,\s*db_session\):', func_def)
    if not m:
        return func_def
    
    test_name = m.group(1)
    return f'''@patch("polyflip.integrations.polymarket.client.PolymarketClient")
@patch("polyflip.execution.worker.build_execution_gateway")
async def {test_name}(mock_build_gateway, mock_client_class, db_session):'''

new_content = re.sub(r'@patch\("polyflip\.execution\.worker\.PolymarketClient"\)\n@patch\("polyflip\.execution\.worker\.build_execution_gateway"\)\nasync def test_[a-zA-Z0-9_]+\(mock_build_gateway,\s*mock_client_class,\s*db_session\):', repl, content)

if new_content != content:
    with open('tests/test_live_session.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Patched test_live_session.py again")
