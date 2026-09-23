import json
import ast

with open('ASR_FULL_BENCHMARK_CLOUD_v4.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
code_cell_idx = 0
for i, cell in enumerate(nb['cells']):
    ctype = cell['cell_type']
    src = cell['source']
    first_line = src[0].strip() if src else ""
    if ctype == 'code':
        code_str = "".join(src)
        try:
            ast.parse(code_str)
            status = "AST_OK"
        except SyntaxError as e:
            status = f"AST_FAIL: {e}"
        print(f"Cell {i:02d} [CODE #{code_cell_idx:02d} | {status} | {len(src)} lines]: {first_line[:70]}")
        code_cell_idx += 1
    else:
        print(f"Cell {i:02d} [MD                | {len(src)} lines]: {first_line[:70]}")
