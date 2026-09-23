import json
import ast

nb = json.load(open('ASR_STAGE0_BUILD_FROZEN_MANIFESTS.ipynb', 'r', encoding='utf-8'))
print(f"Total cells: {len(nb['cells'])}")

code_count = 0
for i, c in enumerate(nb['cells']):
    ctype = c['cell_type']
    first_line = c['source'][0].strip() if c['source'] else ''
    if ctype == 'code':
        code_str = "".join(c['source'])
        clean = "\n".join([l for l in code_str.split("\n") if not l.strip().startswith("!")])
        try:
            ast.parse(clean)
            st = "AST_OK"
        except Exception as e:
            st = f"AST_ERR: {e}"
        print(f"Cell {i:02d} [CODE #{code_count:02d} | {st} | {len(c['source']):3d} lines]: {first_line[:65]}")
        code_count += 1
    else:
        print(f"Cell {i:02d} [MD          | {len(c['source']):3d} lines]: {first_line[:65]}")
