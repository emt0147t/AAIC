with open("scratch/update_v4_generator.py", "r", encoding="utf-8") as f:
    code = f.read()

code = code.replace('print(f"\\n[Auditing', 'print(f"\\\\\\\\n[Auditing')
code = code.replace('print(f"\\nSaved duration', 'print(f"\\\\\\\\nSaved duration')
code = code.replace('print(f"\\n[Frozen Dynamic', 'print(f"\\\\\\\\n[Frozen Dynamic')

with open("scratch/update_v4_generator.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Updated update_v4_generator.py with escaped newlines.")
