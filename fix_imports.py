files = [
    'tests/models/test_c_grid_robustness.py',
    'tests/test_decision_logic_max_edge.py'
]

for filepath in files:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    if 'import pytest' not in content:
        content = 'import pytest\n' + content
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
