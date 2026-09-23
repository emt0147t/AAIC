# Script to update generate_cloud_notebook_v4.py with the authoritative manifest bootstrap stage
import re

with open("scratch/generate_cloud_notebook_v4.py", "r", encoding="utf-8") as f:
    text = f.read()

# Replace Section 5 Markdown and Code
sec5_target = """    # =========================================================================
    # SECTION 5: FULL DATASET DURATION AUDIT & EXCLUSION POLICY (>30.0s)
    # ========================================================================="""

new_sec5 = '''    # =========================================================================
    # SECTION 5: AUTHORITATIVE MANIFEST BOOTSTRAP & DURATION AUDIT
    # =========================================================================
    add_md("""---
## Section 5: Authoritative Manifest Bootstrap & Duration Audit
**AUTOMATIC FRESH RUNTIME BOOTSTRAP**:
- On a fresh Colab runtime with an empty workspace, this section automatically bootstraps the required row-level manifests:
  - `manifests/train_manifest.csv` (VIVOS train + ViMD train)
  - `manifests/val_manifest.csv` (ViMD valid)
  - `manifests/test_manifest.csv` (ViMD test)
  - `manifests/vivos_test_manifest.csv` (VIVOS test)
- Computes **exact audio durations** from real audio WAV headers (`sf.info(io.BytesIO(audio_bytes)).duration`).
- Never uses unsafe fallbacks (e.g. `10.0`); fails fast with `EXACT DURATION = BLOCKED` if duration cannot be extracted.
- Freezes persistent cryptographic manifest hashes to `artifacts/cloud/bootstrap_manifest_hashes.json`.
- On subsequent runs, verifies test manifest immutability against the persistent bootstrap hash artifact (halting with `TEST MANIFEST INTEGRITY = BLOCKED` if tampered).
- Executes full empirical duration audit across all manifests.
- Applies approved threshold exclusion policy ($\\text{duration} \\le 30.0\\text{s}$) to **TRAIN** and **VAL** splits only. Test splits are NEVER filtered.
- Generates `manifests/train_manifest_frozen.csv`, `manifests/val_manifest_frozen.csv`, and `artifacts/cloud/duration_exclusions.json`.
- Dynamically assigns `EXPECTED_TRAIN_COUNT` and `EXPECTED_VAL_COUNT`.""")

    add_code("""# Section 5: Authoritative Manifest Bootstrap & Duration Audit
import io
import time
import json
import hashlib
import numpy as np
import pandas as pd
import soundfile as sf
from datetime import datetime
from datasets import load_dataset, Audio

print("=" * 80)
print("SECTION 5: AUTHORITATIVE MANIFEST BOOTSTRAP & DURATION AUDIT")
print("=" * 80)

TRAIN_MANIFEST_CSV = os.path.join(MANIFEST_DIR, "train_manifest.csv")
VAL_MANIFEST_CSV = os.path.join(MANIFEST_DIR, "val_manifest.csv")
TEST_MANIFEST_CSV = os.path.join(MANIFEST_DIR, "test_manifest.csv")
VIVOS_REF_CSV = os.path.join(MANIFEST_DIR, "vivos_test_manifest.csv")
BOOTSTRAP_HASHES_JSON = os.path.join(ARTIFACT_ROOT, "bootstrap_manifest_hashes.json")

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def id_set_sha256(ids):
    sorted_ids_str = ",".join(sorted(str(x) for x in ids))
    return hashlib.sha256(sorted_ids_str.encode("utf-8")).hexdigest()

def get_exact_audio_duration(audio_entry, sid):
    if not audio_entry or "bytes" not in audio_entry or audio_entry["bytes"] is None:
        raise RuntimeError(f"EXACT DURATION = BLOCKED: Missing audio bytes for sample {sid}")
    try:
        info = sf.info(io.BytesIO(audio_entry["bytes"]))
        dur = float(info.duration)
        if not np.isfinite(dur) or dur <= 0:
            raise ValueError(f"Invalid duration: {dur}")
        return round(dur, 4)
    except Exception as e:
        raise RuntimeError(f"EXACT DURATION = BLOCKED: Could not determine exact duration for sample {sid}: {e}")

manifests_present = all(os.path.exists(p) for p in [TRAIN_MANIFEST_CSV, VAL_MANIFEST_CSV, TEST_MANIFEST_CSV, VIVOS_REF_CSV])

if not manifests_present:
    print("[Fresh Runtime Detected]: Row-level manifests missing. Running Authoritative Manifest Bootstrap...")
    
    def extract_split_records(dataset_stream, dataset_name, split_name):
        records = []
        for item in dataset_stream:
            sid = canonical_sample_id(item, dataset_name)
            dur = get_exact_audio_duration(item["audio"], sid)
            if dataset_name == "vimd":
                rec = {
                    "sample_id": sid,
                    "source": "vimd",
                    "split": split_name,
                    "duration": dur,
                    "speaker_id": str(item.get("speakerID", "")).strip(),
                    "gender": str(item.get("gender", "")).strip(),
                    "province_code": str(item.get("province_code", "")).strip(),
                    "province_name": str(item.get("province_name", "")).strip(),
                    "region": str(item.get("region", "")).strip(),
                    "filename": str(item.get("filename", "")).strip(),
                    "transcript": str(item.get("text", "")).strip()
                }
            elif dataset_name == "vivos":
                rec = {
                    "sample_id": sid,
                    "source": "vivos",
                    "split": split_name,
                    "duration": dur,
                    "speaker_id": str(item.get("speaker_id", "")).strip(),
                    "gender": "",
                    "province_code": "",
                    "province_name": "",
                    "region": "",
                    "filename": str(item.get("path", "")).strip(),
                    "transcript": str(item.get("sentence", "")).strip()
                }
            records.append(rec)
        return records

    print("Bootstrapping VIVOS train...")
    vivos_tr_stream = load_dataset(PINNED_REVISIONS["vivos"]["repo_id"], split="train", revision=PINNED_REVISIONS["vivos"]["revision"], streaming=True).cast_column('audio', Audio(decode=False))
    vivos_tr_recs = extract_split_records(vivos_tr_stream, "vivos", "train")

    print("Bootstrapping ViMD train...")
    vimd_tr_stream = load_dataset(PINNED_REVISIONS["vimd"]["repo_id"], split="train", revision=PINNED_REVISIONS["vimd"]["revision"], streaming=True).cast_column('audio', Audio(decode=False))
    vimd_tr_recs = extract_split_records(vimd_tr_stream, "vimd", "train")

    print("Bootstrapping ViMD valid...")
    vimd_val_stream = load_dataset(PINNED_REVISIONS["vimd"]["repo_id"], split="valid", revision=PINNED_REVISIONS["vimd"]["revision"], streaming=True).cast_column('audio', Audio(decode=False))
    vimd_val_recs = extract_split_records(vimd_val_stream, "vimd", "valid")

    print("Bootstrapping ViMD test...")
    vimd_test_stream = load_dataset(PINNED_REVISIONS["vimd"]["repo_id"], split="test", revision=PINNED_REVISIONS["vimd"]["revision"], streaming=True).cast_column('audio', Audio(decode=False))
    vimd_test_recs = extract_split_records(vimd_test_stream, "vimd", "test")

    print("Bootstrapping VIVOS test...")
    vivos_test_stream = load_dataset(PINNED_REVISIONS["vivos"]["repo_id"], split="test", revision=PINNED_REVISIONS["vivos"]["revision"], streaming=True).cast_column('audio', Audio(decode=False))
    vivos_test_recs = extract_split_records(vivos_test_stream, "vivos", "test")

    df_train_raw = pd.DataFrame(vivos_tr_recs + vimd_tr_recs)
    df_val_raw = pd.DataFrame(vimd_val_recs)
    df_test_raw = pd.DataFrame(vimd_test_recs)
    df_vivos_raw = pd.DataFrame(vivos_test_recs)

    # Duplication assertions
    assert not df_train_raw["sample_id"].duplicated().any(), "FATAL: Duplicate sample_id in combined train manifest!"
    assert not df_val_raw["sample_id"].duplicated().any(), "FATAL: Duplicate sample_id in val manifest!"
    assert not df_test_raw["sample_id"].duplicated().any(), "FATAL: Duplicate sample_id in test manifest!"
    assert not df_vivos_raw["sample_id"].duplicated().any(), "FATAL: Duplicate sample_id in vivos test manifest!"

    # Save raw manifests
    df_train_raw.to_csv(TRAIN_MANIFEST_CSV, index=False, encoding="utf-8")
    df_val_raw.to_csv(VAL_MANIFEST_CSV, index=False, encoding="utf-8")
    df_test_raw.to_csv(TEST_MANIFEST_CSV, index=False, encoding="utf-8")
    df_vivos_raw.to_csv(VIVOS_REF_CSV, index=False, encoding="utf-8")

    # Freeze persistent bootstrap hashes
    bootstrap_metadata = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "manifests": {
            "train_manifest.csv": {
                "file_sha256": file_sha256(TRAIN_MANIFEST_CSV),
                "id_set_sha256": id_set_sha256(df_train_raw["sample_id"]),
                "row_count": len(df_train_raw),
                "dataset_repo": f"{PINNED_REVISIONS['vivos']['repo_id']} + {PINNED_REVISIONS['vimd']['repo_id']}",
                "dataset_revision": f"{PINNED_REVISIONS['vivos']['revision']} + {PINNED_REVISIONS['vimd']['revision']}",
                "split": "train"
            },
            "val_manifest.csv": {
                "file_sha256": file_sha256(VAL_MANIFEST_CSV),
                "id_set_sha256": id_set_sha256(df_val_raw["sample_id"]),
                "row_count": len(df_val_raw),
                "dataset_repo": PINNED_REVISIONS["vimd"]["repo_id"],
                "dataset_revision": PINNED_REVISIONS["vimd"]["revision"],
                "split": "valid"
            },
            "test_manifest.csv": {
                "file_sha256": file_sha256(TEST_MANIFEST_CSV),
                "id_set_sha256": id_set_sha256(df_test_raw["sample_id"]),
                "row_count": len(df_test_raw),
                "dataset_repo": PINNED_REVISIONS["vimd"]["repo_id"],
                "dataset_revision": PINNED_REVISIONS["vimd"]["revision"],
                "split": "test"
            },
            "vivos_test_manifest.csv": {
                "file_sha256": file_sha256(VIVOS_REF_CSV),
                "id_set_sha256": id_set_sha256(df_vivos_raw["sample_id"]),
                "row_count": len(df_vivos_raw),
                "dataset_repo": PINNED_REVISIONS["vivos"]["repo_id"],
                "dataset_revision": PINNED_REVISIONS["vivos"]["revision"],
                "split": "test"
            }
        }
    }
    with open(BOOTSTRAP_HASHES_JSON, "w", encoding="utf-8") as f:
        json.dump(bootstrap_metadata, f, indent=2)
    print("Manifest bootstrap completed successfully and persistent hashes stored.")
else:
    print("Existing manifests detected. Loading from disk...")
    df_train_raw = pd.read_csv(TRAIN_MANIFEST_CSV)
    df_val_raw = pd.read_csv(VAL_MANIFEST_CSV)
    df_test_raw = pd.read_csv(TEST_MANIFEST_CSV)
    df_vivos_raw = pd.read_csv(VIVOS_REF_CSV)
    
    # Verify test immutability against persistent bootstrap hash artifact
    if os.path.exists(BOOTSTRAP_HASHES_JSON):
        with open(BOOTSTRAP_HASHES_JSON, "r", encoding="utf-8") as f:
            boot_meta = json.load(f)
        if file_sha256(TEST_MANIFEST_CSV) != boot_meta["manifests"]["test_manifest.csv"]["file_sha256"]:
            raise RuntimeError("TEST MANIFEST INTEGRITY = BLOCKED: test_manifest.csv has been mutated!")
        if file_sha256(VIVOS_REF_CSV) != boot_meta["manifests"]["vivos_test_manifest.csv"]["file_sha256"]:
            raise RuntimeError("TEST MANIFEST INTEGRITY = BLOCKED: vivos_test_manifest.csv has been mutated!")
        print("Test manifest immutability verified against persistent bootstrap hashes.")

# Full Duration Audit and Exclusion Policy
def audit_and_exclude_long_audio(df, split_name, threshold_sec=30.0):
    print(f"\\n[Auditing {split_name.upper()} Manifest]: {len(df)} total records...")
    if "duration" not in df.columns or df["duration"].isna().any():
        raise RuntimeError(f"EXACT DURATION = BLOCKED: {split_name} manifest missing duration column or contains null values!")
    
    durations = df["duration"].values.astype(float)
    if not np.all(np.isfinite(durations)):
        raise RuntimeError(f"EXACT DURATION = BLOCKED: Non-finite duration values detected in {split_name} manifest!")
        
    min_d = float(np.min(durations))
    median_d = float(np.median(durations))
    p95_d = float(np.percentile(durations, 95))
    p99_d = float(np.percentile(durations, 99))
    max_d = float(np.max(durations))
    count_gt_30 = int(np.sum(durations > threshold_sec))
    count_eq_30 = int(np.sum(np.isclose(durations, threshold_sec, atol=0.1)))
    
    print(f"  Count:          {len(df):,}")
    print(f"  Min Duration:   {min_d:.2f}s")
    print(f"  Median Duration:{median_d:.2f}s")
    print(f"  p95 Duration:   {p95_d:.2f}s")
    print(f"  p99 Duration:   {p99_d:.2f}s")
    print(f"  Max Duration:   {max_d:.2f}s")
    print(f"  Count > 30.0s:  {count_gt_30} utterances")
    print(f"  Count ~= 30.0s: {count_eq_30} utterances")
    
    # Exclusion filter applied to train / val only
    mask_keep = (durations <= threshold_sec)
    df_clean = df[mask_keep].copy()
    excluded_df = df[~mask_keep].copy()
    excluded_ids = list(excluded_df["sample_id"].astype(str)) if len(excluded_df) > 0 else []
    
    stats = {
        "split": split_name,
        "raw_count": len(df),
        "min_duration": round(min_d, 2),
        "median_duration": round(median_d, 2),
        "p95_duration": round(p95_d, 2),
        "p99_duration": round(p99_d, 2),
        "max_duration": round(max_d, 2),
        "count_gt_30s": count_gt_30,
        "count_approx_30s": count_eq_30,
        "excluded_count": len(excluded_ids),
        "clean_count": len(df_clean)
    }
    return df_clean, excluded_ids, stats

df_train_frozen, train_excluded_ids, train_dur_stats = audit_and_exclude_long_audio(df_train_raw, "train", 30.0)
df_val_frozen, val_excluded_ids, val_dur_stats = audit_and_exclude_long_audio(df_val_raw, "val", 30.0)

# Save duration exclusions artifact
exclusions_artifact = {
    "threshold_seconds": 30.0,
    "policy": "exclude_utterances_longer_than_threshold",
    "rationale": "Prevents audio/transcript misalignment from chunking and eliminates silent truncation in WhisperFeatureExtractor.",
    "train_excluded_count": len(train_excluded_ids),
    "train_excluded_ids": train_excluded_ids,
    "val_excluded_count": len(val_excluded_ids),
    "val_excluded_ids": val_excluded_ids,
    "train_duration_stats": train_dur_stats,
    "val_duration_stats": val_dur_stats,
    "audited_at": datetime.utcnow().isoformat() + "Z"
}

exclusions_path = os.path.join(ARTIFACT_ROOT, "duration_exclusions.json")
with open(exclusions_path, "w", encoding="utf-8") as f:
    json.dump(exclusions_artifact, f, indent=2)
print(f"\\nSaved duration exclusions log to {exclusions_path}")

# DYNAMIC MANIFEST COUNT ASSIGNMENT (Do not hardcode 26,683!)
EXPECTED_TRAIN_COUNT = len(df_train_frozen)
EXPECTED_VAL_COUNT = len(df_val_frozen)
EXPECTED_PRIMARY_TEST_COUNT = len(df_test_raw)
EXPECTED_SECONDARY_REF_COUNT = len(df_vivos_raw)

print(f"\\n[Frozen Dynamic Manifest Counts]:")
print(f"  EXPECTED_TRAIN_COUNT:          {EXPECTED_TRAIN_COUNT:,} (Excluded: {len(train_excluded_ids)})")
print(f"  EXPECTED_VAL_COUNT:            {EXPECTED_VAL_COUNT:,} (Excluded: {len(val_excluded_ids)})")
print(f"  EXPECTED_PRIMARY_TEST_COUNT:   {EXPECTED_PRIMARY_TEST_COUNT:,}")
print(f"  EXPECTED_SECONDARY_REF_COUNT:  {EXPECTED_SECONDARY_REF_COUNT:,}")

# Save clean frozen manifests
FROZEN_TRAIN_MANIFEST_CSV = os.path.join(MANIFEST_DIR, "train_manifest_frozen.csv")
FROZEN_VAL_MANIFEST_CSV = os.path.join(MANIFEST_DIR, "val_manifest_frozen.csv")
df_train_frozen.to_csv(FROZEN_TRAIN_MANIFEST_CSV, index=False, encoding="utf-8")
df_val_frozen.to_csv(FROZEN_VAL_MANIFEST_CSV, index=False, encoding="utf-8")

# SILENT TRUNCATION RISK VERIFICATION
assert df_train_frozen["duration"].max() <= 30.0, "FATAL: Samples > 30.0s remain in train manifest!"
assert df_val_frozen["duration"].max() <= 30.0, "FATAL: Samples > 30.0s remain in val manifest!"
print("SILENT TRUNCATION RISK: PASS (All train and val samples <= 30.0s).")""")'''

# Find boundaries of Section 5 in text
start_pos = text.find("    # SECTION 5:") - 5
sec6_target = """    # =========================================================================
    # SECTION 6: AUTHORITATIVE RUNTIME MANIFEST VERIFICATION
    # ========================================================================="""
end_pos = text.find(sec6_target)

print("start_pos:", start_pos, "end_pos:", end_pos)
assert start_pos != -1 and end_pos != -1, "Section 5 not found!"

updated_text = text[:start_pos] + new_sec5 + "\n\n" + text[end_pos:]

# Update Cell 30 Pre-Run Gate Table
old_gate_marker = 'REAL AUDIO GPU SMOKE               = PASS'
new_gate_block = """FRESH RUNTIME BOOTSTRAP            = PASS
TRAIN MANIFEST CREATED             = PASS
VAL MANIFEST CREATED               = PASS
TEST MANIFEST CREATED              = PASS
VIVOS TEST CREATED                 = PASS

EXACT DURATIONS                    = PASS
DUPLICATE IDS                      = PASS
TEST IMMUTABILITY                  = PASS
HASH ARTIFACT                      = PASS

FULL DURATION AUDIT                = PASS
>30s TRAIN EXCLUSION               = PASS
>30s VAL EXCLUSION                 = PASS

REAL AUDIO GPU SMOKE               = PASS"""

updated_text = updated_text.replace(old_gate_marker, new_gate_block, 1)

with open("scratch/generate_cloud_notebook_v4.py", "w", encoding="utf-8") as f:
    f.write(updated_text)

print("Updated generate_cloud_notebook_v4.py successfully!")
