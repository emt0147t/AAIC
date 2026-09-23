with open("scratch/generate_cloud_notebook_v4.py", "r", encoding="utf-8") as f:
    text = f.read()

# Replace any print(f"\n or print("\n inside Section 5
text = text.replace('print(f"\\n[Auditing', 'print("")\\n    print(f"[Auditing')
text = text.replace('print(f"\\nSaved duration', 'print("")\\nprint(f"Saved duration')
text = text.replace('print(f"\\n[Frozen Dynamic', 'print("")\\nprint(f"[Frozen Dynamic')

# Also in raw text if present as literal newline
text = text.replace('print(f"\\n', 'print("")\\n    print(f"')
text = text.replace('print("\\n', 'print("")\\n    print("')

with open("scratch/generate_cloud_notebook_v4.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Replaced all print(\"\\n with print(\"\") in generate_cloud_notebook_v4.py")
