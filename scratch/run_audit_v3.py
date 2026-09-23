# Complete audit inspector for ASR_FULL_BENCHMARK_CLOUD_v3.ipynb
import json
import re

nb = json.load(open("ASR_FULL_BENCHMARK_CLOUD_v3.ipynb", "r", encoding="utf-8"))

def inspect_all():
    print(f"Total cells: {len(nb['cells'])}")
    code_cells = [(i, c) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']
    print(f"Total Python code cells: {len(code_cells)}")
    
    # Check B1: split="valid" vs split="validation"
    print("\n=== B1: ViMD Split Name ===")
    b1_found_valid = []
    b1_found_invalid = []
    for i, c in code_cells:
        src = "".join(c['source'])
        if re.search(r'split\s*=\s*["\']valid["\']', src):
            b1_found_valid.append(i)
        if re.search(r'split\s*=\s*["\']validation["\']', src):
            b1_found_invalid.append(i)
    print("Cells with split='valid':", b1_found_valid)
    print("Cells with split='validation':", b1_found_invalid)
    
    # Check B2: ViMD text field
    print("\n=== B2: ViMD Transcript Field ===")
    b2_transcription = []
    b2_text = []
    for i, c in code_cells:
        src = "".join(c['source'])
        if 'transcription' in src:
            b2_transcription.append(i)
        if 'item["text"]' in src or "item['text']" in src:
            b2_text.append(i)
    print("Cells with 'transcription':", b2_transcription)
    print("Cells with item['text']:", b2_text)
    
    # Check B3: Manifest Authoritative
    print("\n=== B3: Manifest Authoritative ===")
    for i, c in code_cells:
        src = "".join(c['source'])
        if "manifest_runtime_verification.json" in src:
            print(f"Cell {i} saves manifest_runtime_verification.json")
        if "val_manifest_ids" in src:
            print(f"Cell {i} filters via val_manifest_ids")
        if "test_manifest_ids" in src:
            print(f"Cell {i} filters via test_manifest_ids")
            
    # Check B4: Resume test
    print("\n=== B4: Checkpoint Resume Dry-Run ===")
    for i, c in code_cells:
        src = "".join(c['source'])
        if "COMPREHENSIVE CHECKPOINT RESUME DRY-RUN: PASS" in src:
            print(f"Cell {i} contains resume dry-run pass assertion")
            
    # Check B5: Seed
    print("\n=== B5: Seed Initialization ===")
    for i, c in code_cells:
        src = "".join(c['source'])
        if "set_seed(" in src:
            print(f"Cell {i} executes set_seed")
            
    # Check B6: Fingerprint & Resume States
    print("\n=== B6: Resume States & Fingerprint ===")
    for i, c in code_cells:
        src = "".join(c['source'])
        if "compute_protocol_fingerprint" in src:
            print(f"Cell {i} defines/uses compute_protocol_fingerprint")
        if "CHECKPOINT PROTOCOL MISMATCH = BLOCKED" in src:
            print(f"Cell {i} asserts fingerprint match")
            
    # Check B7: Interrupted Semantics
    print("\n=== B7: Interrupted Checkpoint Semantics ===")
    for i, c in code_cells:
        src = "".join(c['source'])
        if "INTERRUPTED" in src:
            print(f"Cell {i} contains INTERRUPTED logic")
            
    # Check B8: Gradient Accumulation
    print("\n=== B8: Gradient Accumulation ===")
    for i, c in code_cells:
        src = "".join(c['source'])
        if "micro_step % GRAD_ACCUM" in src:
            print(f"Cell {i} handles micro_step % GRAD_ACCUM")
        if "correction_factor" in src:
            print(f"Cell {i} handles partial group rescale")
            
    # Check B9: Dependencies
    print("\n=== B9: Dependency Pinning ===")
    for i, c in code_cells:
        src = "".join(c['source'])
        if "EXACT_PINNED_PACKAGES" in src:
            print(f"Cell {i} defines EXACT_PINNED_PACKAGES")
        if "import evaluate" in src:
            print(f"Cell {i} imports evaluate")
            
    # Check B10: Test IDs Equality
    print("\n=== B10: Test IDs Verification ===")
    for i, c in code_cells:
        src = "".join(c['source'])
        if "len(evaluated_ids) == 2026" in src:
            print(f"Cell {i} verifies ViMD evaluated_ids count")
        if "set(evaluated_ids) == test_manifest_ids" in src:
            print(f"Cell {i} verifies ViMD evaluated_ids set equality")
        if "len(vivos_evaluated_ids) == 760" in src:
            print(f"Cell {i} verifies VIVOS evaluated_ids count")

if __name__ == "__main__":
    inspect_all()
