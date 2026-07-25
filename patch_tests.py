import re

files = [
    'tests/test_derived_features.py',
    'tests/test_feature_lags.py',
    'tests/test_interaction_features.py',
    'tests/models/test_c_grid_robustness.py',
    'tests/models/test_logistic_regression_pipeline.py'
]

for f in files:
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Remove specific assertions that fail due to deleted features
    content = re.sub(r'(?m)^\s*assert r\.loc\[\d+, \"deviation_x_time\"\].*$\n?', '', content)
    content = re.sub(r'(?m)^\s*assert r\.loc\[\d+, \"price_deviation_sq\"\].*$\n?', '', content)
    content = re.sub(r'(?m)^\s*assert r\.iloc\[\d+\]\[\"price_velocity_lag1\"\].*$\n?', '', content)
    content = re.sub(r'(?m)^\s*assert not first_rows\[\"price_velocity_lag1\"\].isna\(\).any\(\).*$\n?', '', content)
    
    # In test_interaction_features.py
    content = re.sub(r'\"velocity_x_phase\", \"dev_sq_x_phase\"', '', content)
    
    # In test_c_grid_robustness.py
    content = re.sub(r'(?m)^\s*\"deviation_x_time\": 0\.0,.*$\n?', '', content)
    content = re.sub(r'(?m)^\s*\"price_deviation_sq\": 0\.0,.*$\n?', '', content)
    content = re.sub(r'(?m)^\s*\"price_velocity_lag1\": 0\.0,.*$\n?', '', content)
    
    # In test_logistic_regression_pipeline.py
    content = content.replace('\"deviation_x_time\", \"price_deviation_sq\",', '')
    content = content.replace('\"time_phase\"', '')
    
    with open(f, 'w', encoding='utf-8') as file:
        file.write(content)
print('Done!')
