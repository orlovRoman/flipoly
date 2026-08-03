import re

with open('tests/test_live_session.py', encoding='utf-8') as f:
    content = f.read()

new_content = content.replace(
    '@patch("polyflip.integrations.polymarket.client.PolymarketClient")',
    '@patch("polyflip.collector.client.PolymarketClient")'
)

if new_content != content:
    with open('tests/test_live_session.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Patched test_live_session.py again")
