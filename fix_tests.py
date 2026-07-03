import glob

for filepath in glob.glob('tests/*.py'):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if 'MAX_EDGE' in content:
        # replace MAX_EDGE with MAX_BET_EDGE in strings and dict keys
        content = content.replace("'MAX_EDGE'", "'MAX_BET_EDGE'")
        content = content.replace('"MAX_EDGE"', '"MAX_BET_EDGE"')
        # for test_init_settings.py
        content = content.replace('from polyflip.constants import MAX_EDGE', 'from polyflip.constants import MAX_EDGE_SCALING as MAX_BET_EDGE')
        content = content.replace('from polyflip.constants import MAX_EDGE, MIN_EDGE', 'from polyflip.constants import MAX_EDGE_SCALING as MAX_BET_EDGE, MIN_EDGE')
        content = content.replace('constants.MAX_EDGE', 'constants.MAX_EDGE_SCALING')
        content = content.replace('MAX_EDGE', 'MAX_BET_EDGE')
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
