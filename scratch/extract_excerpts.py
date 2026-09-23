# Excerpt extractor for audit report
import json

nb = json.load(open("ASR_FULL_BENCHMARK_CLOUD_v3.ipynb", "r", encoding="utf-8"))

def print_cell_snippet(cell_idx, start_line=0, end_line=None):
    lines = nb["cells"][cell_idx]["source"]
    if end_line is None:
        end_line = len(lines)
    snippet = "".join(lines[start_line:end_line])
    print(f"--- Cell {cell_idx} (lines {start_line+1} to {end_line}) ---")
    print(snippet)
    print("-" * 50)

print("B1 Excerpt:")
print_cell_snippet(10, 0, 30)
print_cell_snippet(24, 0, 25)

print("\nB2 Excerpt:")
print_cell_snippet(10, 35, 55)
print_cell_snippet(20, 30, 50)
print_cell_snippet(24, 15, 35)

print("\nB3 Excerpt:")
print_cell_snippet(12, 50, 85)
print_cell_snippet(24, 10, 30)
print_cell_snippet(32, 20, 45)

print("\nB4 Excerpt:")
print_cell_snippet(26, 40, 80)

print("\nB5 Excerpt:")
print_cell_snippet(8, 0, 20)

print("\nB6 Excerpt:")
print_cell_snippet(22, 0, 35)
print_cell_snippet(30, 20, 50)

print("\nB7 Excerpt:")
print_cell_snippet(30, 90, 130)

print("\nB8 Excerpt:")
print_cell_snippet(30, 185, 235)

print("\nB9 Excerpt:")
print_cell_snippet(6, 0, 30)

print("\nB10 Excerpt:")
print_cell_snippet(32, 55, 85)
print_cell_snippet(34, 45, 75)
