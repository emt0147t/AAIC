import json

nb = json.load(open("ASR_FULL_BENCHMARK_CLOUD_v4.ipynb", "r", encoding="utf-8"))

def get_lines(cell_idx, start, end):
    lines = nb["cells"][cell_idx]["source"][start:end]
    return "".join(lines)

print("=== CHECK A1: Canonical ID Mapping (Cell 10) ===")
print(get_lines(10, 40, 75))

print("\n=== CHECK A2: Training Accounting & Bidirectional Equality (Cell 32) ===")
print(get_lines(32, 275, 305))

print("\n=== CHECK A3: Validation Bidirectional Equality (Cell 26) ===")
print(get_lines(26, 45, 65))

print("\n=== CHECK A4: Runtime Manifest Verification (Cell 14) ===")
print(get_lines(14, 0, 35))

print("\n=== CHECK B1/B3/B4: Duration Audit & Exclusion (Cell 12) ===")
print(get_lines(12, 60, 135))

print("\n=== FINAL PRE-RUN GATE (Cell 30) ===")
print(get_lines(30, 0, 41))
