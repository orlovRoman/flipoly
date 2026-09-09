import sys

content = open('tests/research/test_comprehensive_plan.py').read()
lines = content.split('\n')
out = []
in_if = False
for line in lines:
    if line.strip() == 'if res_path.exists():':
        out.append('    assert res_path.exists(), "Results file missing"')
        in_if = True
    else:
        if in_if and line.startswith('        '):
            out.append(line[4:])
        elif in_if and line.strip() == '':
            out.append(line)
        else:
            in_if = False
            out.append(line)

open('tests/research/test_comprehensive_plan.py', 'w').write('\n'.join(out))
