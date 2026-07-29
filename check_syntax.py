import ast, sys
src = sys.stdin.read()
ast.parse(src)
print("Syntax OK")
