# Script to perform comprehensive static validation of ASR_FULL_BENCHMARK_CLOUD_v3.ipynb
import json
import ast
import re
import sys

def verify_v3():
    nb_path = "ASR_FULL_BENCHMARK_CLOUD_v3.ipynb"
    print(f"Loading {nb_path}...")
    nb = json.load(open(nb_path, "r", encoding="utf-8"))
    
    cells = nb["cells"]
    code_cells = [c for c in cells if c["cell_type"] == "code"]
    print(f"Total cells: {len(cells)} | Code cells: {len(code_cells)}")
    
    # 1. AST Syntax Validation
    print("\n[Check 1] AST Syntax Validation on every Python cell...")
    ast_pass = True
    for idx, cell in enumerate(code_cells):
        code_str = "".join(cell["source"])
        try:
            ast.parse(code_str)
        except SyntaxError as e:
            print(f"  FAIL: Code cell {idx} syntax error: {e}")
            ast_pass = False
    if ast_pass:
        print("  -> AST SYNTAX VALIDATION: PASS")
    else:
        print("  -> AST SYNTAX VALIDATION: FAIL")
        
    # 2. Static Reference / Variable Validation
    print("\n[Check 2] Static Reference / Variable Validation...")
    ref_pass = True
    # Check that split="validation" does NOT appear anywhere in code cells
    for idx, cell in enumerate(code_cells):
        code_str = "".join(cell["source"])
        if 'split="validation"' in code_str or "split='validation'" in code_str:
            print(f"  FAIL: Code cell {idx} still contains split='validation'")
            ref_pass = False
            
    # Check that ViMD transcript access does NOT use "transcription"
    for idx, cell in enumerate(code_cells):
        code_str = "".join(cell["source"])
        if 'vimd' in code_str.lower() and '["transcription"]' in code_str:
            print(f"  FAIL: Code cell {idx} accesses ViMD via ['transcription']")
            ref_pass = False
            
    # Check that seed 42 is initialized
    seed_found = False
    for cell in code_cells:
        code_str = "".join(cell["source"])
        if 'set_seed(SEED)' in code_str or 'set_seed(42)' in code_str:
            seed_found = True
    if not seed_found:
        print("  FAIL: set_seed(42) not found in code cells")
        ref_pass = False
        
    if ref_pass:
        print("  -> STATIC REFERENCE / VARIABLE VALIDATION: PASS")
    else:
        print("  -> STATIC REFERENCE / VARIABLE VALIDATION: FAIL")
        
    # 3. Manifest Logic Inspection
    print("\n[Check 3] Manifest Logic Inspection...")
    manifest_pass = True
    manifest_checks = [
        "EXACT MANIFEST = BLOCKED",
        "manifest_runtime_verification.json",
        "len(evaluated_ids) == 2026",
        "set(evaluated_ids) == test_manifest_ids",
        "len(vivos_evaluated_ids) == 760",
        "set(vivos_evaluated_ids) == vivos_manifest_ids"
    ]
    all_code = "\n".join(["".join(c["source"]) for c in code_cells])
    for mc in manifest_checks:
        if mc not in all_code:
            print(f"  FAIL: Manifest check '{mc}' not found in notebook code")
            manifest_pass = False
            
    if manifest_pass:
        print("  -> MANIFEST LOGIC INSPECTION: PASS")
    else:
        print("  -> MANIFEST LOGIC INSPECTION: FAIL")
        
    # 4. Checkpoint Resume Logic Inspection
    print("\n[Check 4] Checkpoint Resume Logic Inspection...")
    ckpt_pass = True
    ckpt_checks = [
        "COMPREHENSIVE CHECKPOINT RESUME DRY-RUN: PASS",
        "CHECKPOINT PROTOCOL MISMATCH = BLOCKED",
        "INTERRUPTED",
        "COMPLETED",
        "checkpoint_epoch_",
        "interrupt_checkpoint"
    ]
    for cc in ckpt_checks:
        if cc not in all_code:
            print(f"  FAIL: Checkpoint check '{cc}' not found in notebook code")
            ckpt_pass = False
            
    if ckpt_pass:
        print("  -> CHECKPOINT RESUME LOGIC INSPECTION: PASS")
    else:
        print("  -> CHECKPOINT RESUME LOGIC INSPECTION: FAIL")

    overall = ast_pass and ref_pass and manifest_pass and ckpt_pass
    print(f"\nOVERALL VERIFICATION: {'PASS' if overall else 'FAIL'}")
    return overall

if __name__ == "__main__":
    success = verify_v3()
    sys.exit(0 if success else 1)
