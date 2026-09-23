import re
import ast

with open('scratch/generate_stage0_notebook.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace any print("\n or print(f"\n with print("\\n or print(f"\\n
# Specifically find all unescaped \n inside quotes
fixed_content = content.replace('print("\\n', 'print("\\\\n')
fixed_content = fixed_content.replace('print(f"\\n', 'print(f"\\\\n')
fixed_content = fixed_content.replace('print(f"\\n', 'print(f"\\\\n')

with open('scratch/generate_stage0_notebook.py', 'w', encoding='utf-8') as f:
    f.write(fixed_content)

print("Saved fixed generator.")
