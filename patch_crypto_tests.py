import re

files = [
    'tests/test_engine_crypto_sizing.py',
    'tests/test_crypto_integration.py',
    'tests/crypto/test_risk_guard.py',
    'tests/crypto/test_feature_importance_guard.py'
]

for f in files:
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    
    content = re.sub(r'assert len\(CRYPTO_FEATURES\) == 23', 'assert len(CRYPTO_FEATURES) == 22', content)
    content = re.sub(r'Ожидалось 23 фичи', 'Ожидалось 22 фичи', content)
    content = re.sub(r'np\.array\(\[0\.01\]\*23\)', 'np.array([0.01]*22)', content)
    
    with open(f, 'w', encoding='utf-8') as file:
        file.write(content)
print('Done!')
