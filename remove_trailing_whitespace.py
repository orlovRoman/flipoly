def strip_whitespace(files):
    for filename in files:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        with open(filename, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(line.rstrip(' \t\r\n') + '\n')

if __name__ == "__main__":
    files = [
        'polyflip/api/execution_api.py',
        'polyflip/execution/release_gate.py',
        'tests/test_live_execution_hotfixes.py',
        'tests/test_manual_review.py',
        'tests/test_release_gate.py',
    ]
    strip_whitespace(files)
