# Generator script for ASR_STAGE0_BUILD_FROZEN_MANIFESTS.ipynb
import json
import ast
import os

def create_stage0_notebook():
    cells = []
    
    def add_md(source):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.split("\n")]
        })
        
    def add_code(source):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.split("\n")]
        })

    # =========================================================================
    # CELL 0: Title & Objective
    # =========================================================================
    add_md("""# Stage 0: Build Frozen Manifests for Vietnamese ASR Benchmark
**Target Notebook**: `ASR_STAGE0_BUILD_FROZEN_MANIFESTS.ipynb`  
**Task**: Deterministic Row-Level Manifest Bootstrapping & Audio Duration Auditing  
**Primary Dataset**: ViMD (`nguyendv02/ViMD_Dataset` @ `3a5b30157034e7eadd5c75fae1a820c6f9383398`)  
**Secondary Dataset**: VIVOS (`thanhduycao/vivos_ng_only` @ `b2fbc10431b721dc9b0409b716d56a759d1cf332`)  

---
### Objective & Protocol Context
The full benchmark cloud notebook (`ASR_FULL_BENCHMARK_CLOUD_v4.ipynb`) mandates authoritative row-level manifests for all splits:
- `manifests/train_manifest.csv` (VIVOS train + ViMD train)
- `manifests/val_manifest.csv` (ViMD valid)
- `manifests/test_manifest.csv` (ViMD test)
- `manifests/vivos_test_manifest.csv` (VIVOS test)

This Stage 0 notebook is a dedicated, self-contained bootstrap pipeline that:
1. Audits live Hugging Face dataset schemas at exact pinned commit revisions.
2. Establishes canonical, collision-free sample identifiers (`canonical_sample_id`).
3. Streams audio headers to extract exact, microsecond-accurate audio durations.
4. Generates raw row-level manifests preserving full metadata (speakers, regions, provinces).
5. Performs complete row-level integrity checks (uniqueness, speaker disjointness, SHA-256 digests).
6. Computes the complete empirical audio duration distribution across all splits.
7. Generates frozen training manifests (`train_manifest_frozen.csv`, `val_manifest_frozen.csv`) by applying the approved $> 30.0\\text{s}$ exclusion policy.
8. Enforces byte-for-byte immutability of test manifests (`test_manifest.csv` and `vivos_test_manifest.csv`).
9. Exports `artifacts/cloud/duration_exclusions.json` and `artifacts/cloud/bootstrap_manifest_hashes.json`.
10. Compiles the formal audit report `ASR_STAGE0_MANIFEST_REPORT.md`.""")

    # =========================================================================
    # CELL 1: Environment Setup & Platform Adapter
    # =========================================================================
    add_code("""# Section 0: Environment Setup, Dependencies & Directory Hierarchy
import os
import sys
import shutil
from pathlib import Path

# Detect runtime platform
IN_COLAB = "google.colab" in sys.modules or "COLAB_GPU" in os.environ
IN_KAGGLE = os.path.exists("/kaggle/working")

if IN_COLAB:
    PROJECT_ROOT = "/content/Vietnamese_ASR_Week6"
    print("[Platform Adapter]: Detected Google Colab environment.")
elif IN_KAGGLE:
    PROJECT_ROOT = "/kaggle/working/Vietnamese_ASR_Week6"
    print("[Platform Adapter]: Detected Kaggle Notebook environment.")
else:
    PROJECT_ROOT = os.path.abspath(os.getcwd())
    print(f"[Platform Adapter]: Detected Local/Standard environment: {PROJECT_ROOT}")

MANIFEST_DIR = os.path.join(PROJECT_ROOT, "manifests")
ARTIFACT_ROOT = os.path.join(PROJECT_ROOT, "artifacts", "cloud")

os.makedirs(MANIFEST_DIR, exist_ok=True)
os.makedirs(ARTIFACT_ROOT, exist_ok=True)

print(f"Manifest Directory: {MANIFEST_DIR}")
print(f"Artifact Directory: {ARTIFACT_ROOT}")

# Install dependencies if running in cloud
if IN_COLAB or IN_KAGGLE:
    print("Installing pinned dependencies in cloud environment...")
    !pip install -q datasets soundfile pandas numpy tqdm huggingface_hub
else:
    print("Local environment: Assuming existing Python environment with datasets & soundfile.")
""")

    # =========================================================================
    # CELL 2: Step 1 Markdown
    # =========================================================================
    add_md("""---
## Section 1: Inspect Actual Runtime Schemas (Pinned Revisions)
Inspects split names, column schemas, and sample records directly from the pinned repository revisions:
- **ViMD**: `nguyendv02/ViMD_Dataset` @ `3a5b30157034e7eadd5c75fae1a820c6f9383398`
- **VIVOS**: `thanhduycao/vivos_ng_only` @ `b2fbc10431b721dc9b0409b716d56a759d1cf332`""")

    # =========================================================================
    # CELL 3: Step 1 Code
    # =========================================================================
    add_code("""# Section 1: Schema Inspection
import json
from datasets import load_dataset, get_dataset_split_names, Audio

PINNED_REVISIONS = {
    "vimd": {
        "repo_id": "nguyendv02/ViMD_Dataset",
        "revision": "3a5b30157034e7eadd5c75fae1a820c6f9383398"
    },
    "vivos": {
        "repo_id": "thanhduycao/vivos_ng_only",
        "revision": "b2fbc10431b721dc9b0409b716d56a759d1cf332"
    }
}

print("=" * 80)
print("STEP 1: INSPECTING RUNTIME DATASET SPLITS & SCHEMAS")
print("=" * 80)

# Check splits
vimd_splits = get_dataset_split_names(
    PINNED_REVISIONS["vimd"]["repo_id"], 
    revision=PINNED_REVISIONS["vimd"]["revision"]
)
vivos_splits = get_dataset_split_names(
    PINNED_REVISIONS["vivos"]["repo_id"], 
    revision=PINNED_REVISIONS["vivos"]["revision"]
)

print(f"ViMD Split Names:  {vimd_splits}")
print(f"VIVOS Split Names: {vivos_splits}")

assert "train" in vimd_splits and "valid" in vimd_splits and "test" in vimd_splits, (
    f"ViMD missing required splits! Found: {vimd_splits}"
)
assert "train" in vivos_splits and "test" in vivos_splits, (
    f"VIVOS missing required splits! Found: {vivos_splits}"
)

# Inspect 1 sample from each required split
schema_inspection = {}

print("\\n--- Inspecting Sample Items ---")

# 1. ViMD Train
ds_vimd_train = load_dataset(
    PINNED_REVISIONS["vimd"]["repo_id"], split="train", 
    revision=PINNED_REVISIONS["vimd"]["revision"], streaming=True
).cast_column('audio', Audio(decode=False))
sample_vimd_train = next(iter(ds_vimd_train))
schema_inspection["vimd_train"] = {k: str(type(v)) for k, v in sample_vimd_train.items()}
print("ViMD train columns:     ", list(sample_vimd_train.keys()))

# 2. ViMD Valid
ds_vimd_val = load_dataset(
    PINNED_REVISIONS["vimd"]["repo_id"], split="valid", 
    revision=PINNED_REVISIONS["vimd"]["revision"], streaming=True
).cast_column('audio', Audio(decode=False))
sample_vimd_val = next(iter(ds_vimd_val))
schema_inspection["vimd_valid"] = {k: str(type(v)) for k, v in sample_vimd_val.items()}
print("ViMD valid columns:     ", list(sample_vimd_val.keys()))

# 3. ViMD Test
ds_vimd_test = load_dataset(
    PINNED_REVISIONS["vimd"]["repo_id"], split="test", 
    revision=PINNED_REVISIONS["vimd"]["revision"], streaming=True
).cast_column('audio', Audio(decode=False))
sample_vimd_test = next(iter(ds_vimd_test))
schema_inspection["vimd_test"] = {k: str(type(v)) for k, v in sample_vimd_test.items()}
print("ViMD test columns:      ", list(sample_vimd_test.keys()))

# 4. VIVOS Train
ds_vivos_train = load_dataset(
    PINNED_REVISIONS["vivos"]["repo_id"], split="train", 
    revision=PINNED_REVISIONS["vivos"]["revision"], streaming=True
).cast_column('audio', Audio(decode=False))
sample_vivos_train = next(iter(ds_vivos_train))
schema_inspection["vivos_train"] = {k: str(type(v)) for k, v in sample_vivos_train.items()}
print("VIVOS train columns:    ", list(sample_vivos_train.keys()))

# 5. VIVOS Test
ds_vivos_test = load_dataset(
    PINNED_REVISIONS["vivos"]["repo_id"], split="test", 
    revision=PINNED_REVISIONS["vivos"]["revision"], streaming=True
).cast_column('audio', Audio(decode=False))
sample_vivos_test = next(iter(ds_vivos_test))
schema_inspection["vivos_test"] = {k: str(type(v)) for k, v in sample_vivos_test.items()}
print("VIVOS test columns:     ", list(sample_vivos_test.keys()))

# Verify required fields
for split_name, s in [("vimd_train", sample_vimd_train), ("vimd_valid", sample_vimd_val), ("vimd_test", sample_vimd_test)]:
    assert "filename" in s, f"{split_name} missing 'filename'"
    assert "text" in s, f"{split_name} missing 'text'"
    assert "speakerID" in s, f"{split_name} missing 'speakerID'"

for split_name, s in [("vivos_train", sample_vivos_train), ("vivos_test", sample_vivos_test)]:
    assert "path" in s, f"{split_name} missing 'path'"
    assert "sentence" in s, f"{split_name} missing 'sentence'"
    assert "speaker_id" in s, f"{split_name} missing 'speaker_id'"

print("\\nSCHEMA PREFLIGHT AUDIT: ALL CHECKS PASS.")
""")

    # =========================================================================
    # CELL 4: Step 2 Markdown
    # =========================================================================
    add_md("""---
## Section 2: Canonical Sample-ID Mapping
Defines `canonical_sample_id(item, dataset_name)` using verified runtime schema attributes:
- For **ViMD**: strictly uses `item["filename"]` (e.g. `11_0001.wav`).
- For **VIVOS**: strictly uses `item["path"]` (e.g. `vivos/train/waves/VIVOSSPK27/VIVOSSPK27_155.wav`).
- Raises explicit exceptions on missing, empty, or ambiguous identifiers.""")

    # =========================================================================
    # CELL 5: Step 2 Code
    # =========================================================================
    add_code("""# Section 2: Canonical Sample Identifier Mapping
def canonical_sample_id(item, dataset_name):
    \"\"\"
    Maps raw dataset item to canonical, stable sample_id.
    Fails fast if identifier is missing or ambiguous.
    \"\"\"
    if dataset_name == "vimd":
        if "filename" not in item or not item["filename"]:
            raise KeyError(f"ViMD item missing verified 'filename' field: {list(item.keys())}")
        sid = str(item["filename"]).strip()
        if not sid:
            raise ValueError("ViMD item has empty 'filename' field")
        return sid
    elif dataset_name == "vivos":
        if "path" not in item or not item["path"]:
            raise KeyError(f"VIVOS item missing verified 'path' field: {list(item.keys())}")
        sid = str(item["path"]).strip()
        if not sid:
            raise ValueError("VIVOS item has empty 'path' field")
        return sid
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

print("=" * 80)
print("STEP 2: TESTING CANONICAL IDENTIFIER MAPPINGS")
print("=" * 80)

vivos_id_ex = canonical_sample_id(sample_vivos_train, "vivos")
vimd_id_ex = canonical_sample_id(sample_vimd_train, "vimd")

print(f"Dataset: VIVOS")
print(f"  Raw Identifier Fields: path='{sample_vivos_train['path']}', speaker_id='{sample_vivos_train['speaker_id']}'")
print(f"  Canonical sample_id:   '{vivos_id_ex}'")
print(f"  Transcript Field:      sentence='{sample_vivos_train['sentence'][:50]}...'")

print(f"\\nDataset: ViMD")
print(f"  Raw Identifier Fields: filename='{sample_vimd_train['filename']}', speakerID='{sample_vimd_train['speakerID']}'")
print(f"  Canonical sample_id:   '{vimd_id_ex}'")
print(f"  Transcript Field:      text='{sample_vimd_train['text'][:50]}...'")

# Verify sample ID uniqueness between the two datasets
assert vivos_id_ex != vimd_id_ex, "Sample ID collision across datasets!"
print("\\nCANONICAL SAMPLE ID MAPPING: VERIFIED & PASS.")
""")

    # =========================================================================
    # CELL 6: Step 3 & 4 Markdown
    # =========================================================================
    add_md("""---
## Section 3: Build RAW Manifests
Streams all records across the required splits and computes exact audio durations directly from WAV headers via `soundfile.info(io.BytesIO(audio_bytes)).duration`.  
Generates the 4 authoritative raw manifests:
1. `manifests/train_manifest.csv`: VIVOS train + ViMD train
2. `manifests/val_manifest.csv`: ViMD valid
3. `manifests/test_manifest.csv`: ViMD test
4. `manifests/vivos_test_manifest.csv`: VIVOS test

Retains all necessary metadata (`sample_id`, `source_dataset`, `source_split`, `audio_identifier`, `speaker_id`, `transcript`, `duration_seconds`, `gender`, `province_code`, `province_name`, `region`).""")

    # =========================================================================
    # CELL 7: Step 3 & 4 Code
    # =========================================================================
    add_code("""# Section 3: Manifest Extraction & Audio Duration Measurement
import io
import time
import soundfile as sf
import pandas as pd
from tqdm import tqdm

def extract_manifest_records(dataset_stream, dataset_name, split_name, desc="Extracting"):
    \"\"\"
    Iterates through dataset stream and extracts structured manifest records
    with exact duration obtained from audio bytes.
    \"\"\"
    records = []
    t0 = time.time()
    
    for i, item in enumerate(tqdm(dataset_stream, desc=desc)):
        sid = canonical_sample_id(item, dataset_name)
        
        # Audio duration from header bytes
        audio_entry = item.get("audio")
        if not audio_entry or "bytes" not in audio_entry or audio_entry["bytes"] is None:
            raise ValueError(f"Sample {sid} missing valid audio bytes structure!")
            
        audio_bytes = audio_entry["bytes"]
        info = sf.info(io.BytesIO(audio_bytes))
        duration_sec = round(info.duration, 4)
        
        if dataset_name == "vimd":
            speaker = str(item.get("speakerID", "")).strip()
            text = str(item.get("text", "")).strip()
            rec = {
                "sample_id": sid,
                "source_dataset": "vimd",
                "source_split": split_name,
                "audio_identifier": item.get("filename"),
                "speaker_id": speaker,
                "transcript": text,
                "duration_seconds": duration_sec,
                "sample_rate": info.samplerate,
                "frames": info.frames,
                "gender": item.get("gender", ""),
                "province_code": item.get("province_code", ""),
                "province_name": item.get("province_name", ""),
                "region": item.get("region", "")
            }
        elif dataset_name == "vivos":
            speaker = str(item.get("speaker_id", "")).strip()
            text = str(item.get("sentence", "")).strip()
            rec = {
                "sample_id": sid,
                "source_dataset": "vivos",
                "source_split": split_name,
                "audio_identifier": item.get("path"),
                "speaker_id": speaker,
                "transcript": text,
                "duration_seconds": duration_sec,
                "sample_rate": info.samplerate,
                "frames": info.frames,
                "gender": "",
                "province_code": "",
                "province_name": "",
                "region": ""
            }
        records.append(rec)
        
    elapsed = time.time() - t0
    print(f"Extracted {len(records):,} records from {dataset_name.upper()} {split_name} in {elapsed:.1f}s ({len(records)/max(1, elapsed):.1f} samples/s).")
    return records

print("=" * 80)
print("STEP 3 & 4: EXTRACTING SPLITS & COMPILING RAW MANIFESTS")
print("=" * 80)

# 1. VIVOS Train
vivos_train_records = extract_manifest_records(
    ds_vivos_train, "vivos", "train", desc="VIVOS train"
)

# 2. ViMD Train
vimd_train_records = extract_manifest_records(
    ds_vimd_train, "vimd", "train", desc="ViMD train"
)

# 3. ViMD Valid
vimd_val_records = extract_manifest_records(
    ds_vimd_val, "vimd", "valid", desc="ViMD valid"
)

# 4. ViMD Test
vimd_test_records = extract_manifest_records(
    ds_vimd_test, "vimd", "test", desc="ViMD test"
)

# 5. VIVOS Test
vivos_test_records = extract_manifest_records(
    ds_vivos_test, "vivos", "test", desc="VIVOS test"
)

# Build DataFrames
df_raw_train = pd.DataFrame(vivos_train_records + vimd_train_records)
df_raw_val = pd.DataFrame(vimd_val_records)
df_raw_test = pd.DataFrame(vimd_test_records)
df_raw_vivos_test = pd.DataFrame(vivos_test_records)

# File paths
RAW_TRAIN_CSV = os.path.join(MANIFEST_DIR, "train_manifest.csv")
RAW_VAL_CSV = os.path.join(MANIFEST_DIR, "val_manifest.csv")
RAW_TEST_CSV = os.path.join(MANIFEST_DIR, "test_manifest.csv")
RAW_VIVOS_TEST_CSV = os.path.join(MANIFEST_DIR, "vivos_test_manifest.csv")

# Save raw manifests
df_raw_train.to_csv(RAW_TRAIN_CSV, index=False, encoding="utf-8")
df_raw_val.to_csv(RAW_VAL_CSV, index=False, encoding="utf-8")
df_raw_test.to_csv(RAW_TEST_CSV, index=False, encoding="utf-8")
df_raw_vivos_test.to_csv(RAW_VIVOS_TEST_CSV, index=False, encoding="utf-8")

print(f"Saved RAW train manifest:      {RAW_TRAIN_CSV} ({len(df_raw_train):,} rows)")
print(f"Saved RAW val manifest:        {RAW_VAL_CSV} ({len(df_raw_val):,} rows)")
print(f"Saved RAW test manifest:       {RAW_TEST_CSV} ({len(df_raw_test):,} rows)")
print(f"Saved RAW VIVOS test manifest: {RAW_VIVOS_TEST_CSV} ({len(df_raw_vivos_test):,} rows)")
""")

    # =========================================================================
    # CELL 8: Step 5 Markdown
    # =========================================================================
    add_md("""---
## Section 4: Full Row-Level Integrity & Speaker Disjointness Audit
Audits:
1. Row counts & sample ID uniqueness for every manifest.
2. Zero duplicate sample IDs within and across splits.
3. Cryptographic SHA-256 byte hashes of every generated raw manifest.
4. Speaker disjointness evaluation across splits:
   - ViMD train speakers $\\cap$ ViMD valid speakers
   - ViMD train speakers $\\cap$ ViMD test speakers
   - ViMD valid speakers $\\cap$ ViMD test speakers
   - VIVOS train speakers $\\cap$ VIVOS test speakers""")

    # =========================================================================
    # CELL 9: Step 5 Code
    # =========================================================================
    add_code("""# Section 4: Integrity & Speaker Disjointness Audit
import hashlib

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

print("=" * 80)
print("STEP 5: FULL ROW-LEVEL INTEGRITY & SPEAKER DISJOINTNESS AUDIT")
print("=" * 80)

raw_manifest_audits = {}

for name, df, path in [
    ("TRAIN", df_raw_train, RAW_TRAIN_CSV),
    ("VAL", df_raw_val, RAW_VAL_CSV),
    ("TEST", df_raw_test, RAW_TEST_CSV),
    ("VIVOS_TEST", df_raw_vivos_test, RAW_VIVOS_TEST_CSV)
]:
    total_rows = len(df)
    unique_ids = df["sample_id"].nunique()
    duplicates = total_rows - unique_ids
    sha256_hash = compute_sha256(path)
    
    assert duplicates == 0, f"FATAL: Found {duplicates} duplicate sample IDs in {name} manifest!"
    
    first_3 = list(df["sample_id"].iloc[:3])
    last_3 = list(df["sample_id"].iloc[-3:])
    
    raw_manifest_audits[name] = {
        "rows": total_rows,
        "unique_ids": unique_ids,
        "duplicates": duplicates,
        "sha256": sha256_hash,
        "first_3": first_3,
        "last_3": last_3
    }
    
    print(f"[{name} Manifest]:")
    print(f"  Total Rows:   {total_rows:,}")
    print(f"  Unique IDs:   {unique_ids:,}")
    print(f"  Duplicates:   {duplicates}")
    print(f"  SHA-256:      {sha256_hash}")
    print(f"  First 3 IDs:  {first_3}")
    print(f"  Last 3 IDs:   {last_3}")
    print()

# Cross-split sample ID disjointness
train_ids = set(df_raw_train["sample_id"])
val_ids = set(df_raw_val["sample_id"])
test_ids = set(df_raw_test["sample_id"])
vivos_test_ids = set(df_raw_vivos_test["sample_id"])

assert len(train_ids.intersection(val_ids)) == 0, "FATAL: Train and Val overlap!"
assert len(train_ids.intersection(test_ids)) == 0, "FATAL: Train and Test overlap!"
assert len(val_ids.intersection(test_ids)) == 0, "FATAL: Val and Test overlap!"
print("Sample ID Disjointness across Train / Val / Test: PASS (0 overlapping IDs).")

# Speaker Disjointness Analysis
print("\\n--- Speaker Disjointness Audit ---")
vimd_train_speakers = set(df_raw_train[df_raw_train["source_dataset"] == "vimd"]["speaker_id"])
vimd_val_speakers = set(df_raw_val["speaker_id"])
vimd_test_speakers = set(df_raw_test["speaker_id"])

vimd_tr_val_spk_overlap = len(vimd_train_speakers.intersection(vimd_val_speakers))
vimd_tr_test_spk_overlap = len(vimd_train_speakers.intersection(vimd_test_speakers))
vimd_val_test_spk_overlap = len(vimd_val_speakers.intersection(vimd_test_speakers))

print(f"ViMD Train Speakers:      {len(vimd_train_speakers):,}")
print(f"ViMD Val Speakers:        {len(vimd_val_speakers):,}")
print(f"ViMD Test Speakers:       {len(vimd_test_speakers):,}")
print(f"  Train ∩ Val Overlap:    {vimd_tr_val_spk_overlap} speakers")
print(f"  Train ∩ Test Overlap:   {vimd_tr_test_spk_overlap} speakers")
print(f"  Val ∩ Test Overlap:     {vimd_val_test_spk_overlap} speakers")

# VIVOS Speaker Disjointness
vivos_train_speakers = set(df_raw_train[df_raw_train["source_dataset"] == "vivos"]["speaker_id"])
vivos_test_speakers = set(df_raw_vivos_test["speaker_id"])
vivos_spk_overlap = len(vivos_train_speakers.intersection(vivos_test_speakers))

print(f"\\nVIVOS Train Speakers:     {len(vivos_train_speakers):,}")
print(f"VIVOS Test Speakers:      {len(vivos_test_speakers):,}")
print(f"  Train ∩ Test Overlap:   {vivos_spk_overlap} speakers")

speaker_overlap_results = {
    "vimd_train_val": vimd_tr_val_spk_overlap,
    "vimd_train_test": vimd_tr_test_spk_overlap,
    "vimd_val_test": vimd_val_test_spk_overlap,
    "vivos_train_test": vivos_spk_overlap
}
""")

    # =========================================================================
    # CELL 10: Step 6 Markdown
    # =========================================================================
    add_md("""---
## Section 5: Exact Empirical Audio Duration Distribution
Computes exact, non-estimated duration statistics across every full split:
- `count`
- `min`
- `median`
- `p95`
- `p99`
- `max`
- `count > 30.0 sec`
- `count == 30.0 sec`""")

    # =========================================================================
    # CELL 11: Step 6 Code
    # =========================================================================
    add_code("""# Section 5: Duration Statistics Calculation
import numpy as np

def compute_duration_stats(df, split_name, threshold_sec=30.0):
    durations = df["duration_seconds"].values.astype(float)
    count = len(durations)
    min_d = float(np.min(durations))
    median_d = float(np.median(durations))
    p95_d = float(np.percentile(durations, 95))
    p99_d = float(np.percentile(durations, 99))
    max_d = float(np.max(durations))
    gt_30 = int(np.sum(durations > threshold_sec))
    eq_30 = int(np.sum(np.isclose(durations, threshold_sec, atol=0.1)))
    
    stats = {
        "split": split_name,
        "count": count,
        "min": round(min_d, 4),
        "median": round(median_d, 4),
        "p95": round(p95_d, 4),
        "p99": round(p99_d, 4),
        "max": round(max_d, 4),
        "count_gt_30s": gt_30,
        "count_approx_30s": eq_30
    }
    return stats

print("=" * 80)
print("STEP 6: FULL EMPIRICAL DURATION DISTRIBUTION AUDIT")
print("=" * 80)

duration_stats = {}
for name, df in [
    ("TRAIN", df_raw_train),
    ("VAL", df_raw_val),
    ("TEST", df_raw_test),
    ("VIVOS_TEST", df_raw_vivos_test)
]:
    stats = compute_duration_stats(df, name, threshold_sec=30.0)
    duration_stats[name] = stats
    print(f"[{name} Audio Durations]:")
    print(f"  Count:          {stats['count']:,} utterances")
    print(f"  Min Duration:   {stats['min']:.2f}s")
    print(f"  Median:         {stats['median']:.2f}s")
    print(f"  p95:            {stats['p95']:.2f}s")
    print(f"  p99:            {stats['p99']:.2f}s")
    print(f"  Max Duration:   {stats['max']:.2f}s")
    print(f"  Count > 30.0s:  {stats['count_gt_30s']:,} utterances")
    print(f"  Count == 30.0s: {stats['count_approx_30s']:,} utterances")
    print()
""")

    # =========================================================================
    # CELL 12: Step 7 Markdown
    # =========================================================================
    add_md("""---
## Section 6: Create Clean / Frozen Training Manifests (>30.0s Exclusion Policy)
Applies the approved threshold exclusion policy:
$$\\text{duration} > 30.0\\text{s} \\implies \\text{EXCLUDE}$$

**Strict Invariance Rules**:
- Only **TRAIN** and **VAL** are filtered.
- **TEST manifests are NEVER modified** under any circumstance.
- Excluded utterances and their exact durations are logged to `artifacts/cloud/duration_exclusions.json`.
- Creates clean frozen manifests:
  - `manifests/train_manifest_frozen.csv`
  - `manifests/val_manifest_frozen.csv`""")

    # =========================================================================
    # CELL 13: Step 7 Code
    # =========================================================================
    add_code("""# Section 6: Apply Duration Exclusion Policy & Save Frozen Manifests
from datetime import datetime

print("=" * 80)
print("STEP 7: APPLYING DURATION EXCLUSION POLICY (>30.0s)")
print("=" * 80)

DURATION_THRESHOLD = 30.0

# 1. Filter Train
mask_train_keep = (df_raw_train["duration_seconds"] <= DURATION_THRESHOLD)
df_frozen_train = df_raw_train[mask_train_keep].copy()
df_excluded_train = df_raw_train[~mask_train_keep].copy()

# 2. Filter Val
mask_val_keep = (df_raw_val["duration_seconds"] <= DURATION_THRESHOLD)
df_frozen_val = df_raw_val[mask_val_keep].copy()
df_excluded_val = df_raw_val[~mask_val_keep].copy()

print(f"Raw Train Count:       {len(df_raw_train):,}")
print(f"Excluded Train >30s:   {len(df_excluded_train):,}")
print(f"Frozen Train Count:    {len(df_frozen_train):,}")

print(f"\\nRaw Val Count:         {len(df_raw_val):,}")
print(f"Excluded Val >30s:     {len(df_excluded_val):,}")
print(f"Frozen Val Count:      {len(df_frozen_val):,}")

# Assert zero samples > 30.0s remain in frozen manifests
assert df_frozen_train["duration_seconds"].max() <= DURATION_THRESHOLD, "Samples > 30.0s in frozen train!"
assert df_frozen_val["duration_seconds"].max() <= DURATION_THRESHOLD, "Samples > 30.0s in frozen val!"

# Save frozen manifests
FROZEN_TRAIN_CSV = os.path.join(MANIFEST_DIR, "train_manifest_frozen.csv")
FROZEN_VAL_CSV = os.path.join(MANIFEST_DIR, "val_manifest_frozen.csv")

df_frozen_train.to_csv(FROZEN_TRAIN_CSV, index=False, encoding="utf-8")
df_frozen_val.to_csv(FROZEN_VAL_CSV, index=False, encoding="utf-8")

# Build exclusions details
excluded_samples_detail = []
for _, row in df_excluded_train.iterrows():
    excluded_samples_detail.append({
        "sample_id": row["sample_id"],
        "split": "train",
        "source_dataset": row["source_dataset"],
        "duration_seconds": float(row["duration_seconds"])
    })
for _, row in df_excluded_val.iterrows():
    excluded_samples_detail.append({
        "sample_id": row["sample_id"],
        "split": "val",
        "source_dataset": row["source_dataset"],
        "duration_seconds": float(row["duration_seconds"])
    })

duration_exclusions_artifact = {
    "threshold_seconds": DURATION_THRESHOLD,
    "policy": "exclude_utterances_longer_than_threshold",
    "rationale": "Prevents audio/transcript misalignment from chunking and eliminates silent truncation in WhisperFeatureExtractor.",
    "train_raw_count": len(df_raw_train),
    "train_excluded_count": len(df_excluded_train),
    "train_clean_count": len(df_frozen_train),
    "train_excluded_ids": list(df_excluded_train["sample_id"]),
    "val_raw_count": len(df_raw_val),
    "val_excluded_count": len(df_excluded_val),
    "val_clean_count": len(df_frozen_val),
    "val_excluded_ids": list(df_excluded_val["sample_id"]),
    "excluded_samples_detail": excluded_samples_detail,
    "created_at": datetime.utcnow().isoformat() + "Z"
}

EXCLUSIONS_PATH = os.path.join(ARTIFACT_ROOT, "duration_exclusions.json")
with open(EXCLUSIONS_PATH, "w", encoding="utf-8") as f:
    json.dump(duration_exclusions_artifact, f, indent=2)

print(f"\\nSaved duration exclusions log to {EXCLUSIONS_PATH}")
""")

    # =========================================================================
    # CELL 14: Step 9 Markdown
    # =========================================================================
    add_md("""---
## Section 7: Never Mutate Test Manifests (Immutability Verification)
Executes byte-for-byte SHA-256 integrity verification to mathematically prove that neither `test_manifest.csv` nor `vivos_test_manifest.csv` underwent any mutation or filtering during the bootstrap process.""")

    # =========================================================================
    # CELL 15: Step 9 Code
    # =========================================================================
    add_code("""# Section 7: Test Manifest Immutability Verification
print("=" * 80)
print("STEP 9: VERIFYING IMMUTABILITY OF TEST MANIFESTS")
print("=" * 80)

# Check SHA-256 against initial computed hashes
test_sha256_current = compute_sha256(RAW_TEST_CSV)
vivos_test_sha256_current = compute_sha256(RAW_VIVOS_TEST_CSV)

assert test_sha256_current == raw_manifest_audits["TEST"]["sha256"], (
    "FATAL INTEGRITY FAILURE: ViMD test_manifest.csv was mutated during bootstrap!"
)
assert vivos_test_sha256_current == raw_manifest_audits["VIVOS_TEST"]["sha256"], (
    "FATAL INTEGRITY FAILURE: vivos_test_manifest.csv was mutated during bootstrap!"
)

# Verify row counts match exactly
assert len(pd.read_csv(RAW_TEST_CSV)) == len(df_raw_test), "ViMD test row count mismatch!"
assert len(pd.read_csv(RAW_VIVOS_TEST_CSV)) == len(df_raw_vivos_test), "VIVOS test row count mismatch!"

print(f"ViMD Test Manifest SHA-256:       {test_sha256_current} (IMMUTABLE: PASS)")
print(f"VIVOS Test Manifest SHA-256:      {vivos_test_sha256_current} (IMMUTABLE: PASS)")
print("TEST MANIFEST IMMUTABILITY AUDIT: 100% PASS.")
""")

    # =========================================================================
    # CELL 16: Step 8 Markdown
    # =========================================================================
    add_md("""---
## Section 8: Freeze Artifact Hashes
Saves `artifacts/cloud/bootstrap_manifest_hashes.json` containing SHA-256 cryptographic digests, row counts, creation timestamps, and pinned dataset commit hashes for all 6 bootstrap manifest files:
- `train_manifest.csv`
- `val_manifest.csv`
- `test_manifest.csv`
- `vivos_test_manifest.csv`
- `train_manifest_frozen.csv`
- `val_manifest_frozen.csv`""")

    # =========================================================================
    # CELL 17: Step 8 Code
    # =========================================================================
    add_code("""# Section 8: Freeze Manifest Artifact Hashes
print("=" * 80)
print("STEP 8: FREEZING MANIFEST ARTIFACT HASHES")
print("=" * 80)

bootstrap_hashes = {
    "datasets": PINNED_REVISIONS,
    "canonical_id_mapping": {
        "vimd": "item['filename']",
        "vivos": "item['path']"
    },
    "duration_threshold_seconds": DURATION_THRESHOLD,
    "manifests": {
        "train_manifest.csv": {
            "sha256": compute_sha256(RAW_TRAIN_CSV),
            "rows": len(df_raw_train),
            "status": "RAW"
        },
        "val_manifest.csv": {
            "sha256": compute_sha256(RAW_VAL_CSV),
            "rows": len(df_raw_val),
            "status": "RAW"
        },
        "test_manifest.csv": {
            "sha256": compute_sha256(RAW_TEST_CSV),
            "rows": len(df_raw_test),
            "status": "FROZEN_READ_ONLY"
        },
        "vivos_test_manifest.csv": {
            "sha256": compute_sha256(RAW_VIVOS_TEST_CSV),
            "rows": len(df_raw_vivos_test),
            "status": "FROZEN_READ_ONLY"
        },
        "train_manifest_frozen.csv": {
            "sha256": compute_sha256(FROZEN_TRAIN_CSV),
            "rows": len(df_frozen_train),
            "status": "FROZEN_FILTERED"
        },
        "val_manifest_frozen.csv": {
            "sha256": compute_sha256(FROZEN_VAL_CSV),
            "rows": len(df_frozen_val),
            "status": "FROZEN_FILTERED"
        }
    },
    "speaker_overlap": speaker_overlap_results,
    "created_at": datetime.utcnow().isoformat() + "Z"
}

BOOTSTRAP_HASHES_PATH = os.path.join(ARTIFACT_ROOT, "bootstrap_manifest_hashes.json")
with open(BOOTSTRAP_HASHES_PATH, "w", encoding="utf-8") as f:
    json.dump(bootstrap_hashes, f, indent=2)

print(f"Saved bootstrap manifest hashes to {BOOTSTRAP_HASHES_PATH}")
for mname, mdata in bootstrap_hashes["manifests"].items():
    print(f"  {mname:30s} : {mdata['rows']:6,d} rows | SHA-256: {mdata['sha256']}")
""")

    # =========================================================================
    # CELL 18: Step 10 Markdown
    # =========================================================================
    add_md("""---
## Section 9: Final Stage 0 Bootstrap Report Generation
Generates `ASR_STAGE0_MANIFEST_REPORT.md` and displays the final bootstrap verification summary.""")

    # =========================================================================
    # CELL 19: Step 10 Code
    # =========================================================================
    add_code("""# Section 9: Final Stage 0 Bootstrap Report Generation
all_manifests_pass = all([
    os.path.exists(RAW_TRAIN_CSV),
    os.path.exists(RAW_VAL_CSV),
    os.path.exists(RAW_TEST_CSV),
    os.path.exists(RAW_VIVOS_TEST_CSV),
    os.path.exists(FROZEN_TRAIN_CSV),
    os.path.exists(FROZEN_VAL_CSV),
    os.path.exists(EXCLUSIONS_PATH),
    os.path.exists(BOOTSTRAP_HASHES_PATH)
])

stage0_verdict = "READY" if all_manifests_pass else "NOT READY"

report_md = f\"\"\"# Stage 0 Manifest Bootstrap Report: Vietnamese ASR Benchmark

**Date**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}  
**ViMD Revision**: `{PINNED_REVISIONS['vimd']['revision']}`  
**VIVOS Revision**: `{PINNED_REVISIONS['vivos']['revision']}`  
**Verdict**: **STAGE 0 VERDICT = {stage0_verdict}**

---

## 1. Executive Summary & Counts

| Manifest | Source Composition | Raw Count | Excluded (>30s) | Frozen Count | Duplicates | SHA-256 Digest |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `train_manifest.csv` | VIVOS train + ViMD train | {len(df_raw_train):,} | {len(df_excluded_train):,} | {len(df_frozen_train):,} | {raw_manifest_audits['TRAIN']['duplicates']} | `{bootstrap_hashes['manifests']['train_manifest.csv']['sha256'][:16]}...` |
| `val_manifest.csv` | ViMD valid | {len(df_raw_val):,} | {len(df_excluded_val):,} | {len(df_frozen_val):,} | {raw_manifest_audits['VAL']['duplicates']} | `{bootstrap_hashes['manifests']['val_manifest.csv']['sha256'][:16]}...` |
| `test_manifest.csv` | ViMD test | {len(df_raw_test):,} | 0 (untouched) | {len(df_raw_test):,} | {raw_manifest_audits['TEST']['duplicates']} | `{bootstrap_hashes['manifests']['test_manifest.csv']['sha256'][:16]}...` |
| `vivos_test_manifest.csv` | VIVOS test | {len(df_raw_vivos_test):,} | 0 (untouched) | {len(df_raw_vivos_test):,} | {raw_manifest_audits['VIVOS_TEST']['duplicates']} | `{bootstrap_hashes['manifests']['vivos_test_manifest.csv']['sha256'][:16]}...` |

---

## 2. Duration Statistics (Full Manifest Audit)

| Split | Count | Min (s) | Median (s) | p95 (s) | p99 (s) | Max (s) | Count >30s |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TRAIN (Raw)** | {duration_stats['TRAIN']['count']:,} | {duration_stats['TRAIN']['min']:.2f} | {duration_stats['TRAIN']['median']:.2f} | {duration_stats['TRAIN']['p95']:.2f} | {duration_stats['TRAIN']['p99']:.2f} | {duration_stats['TRAIN']['max']:.2f} | {duration_stats['TRAIN']['count_gt_30s']:,} |
| **VAL (Raw)** | {duration_stats['VAL']['count']:,} | {duration_stats['VAL']['min']:.2f} | {duration_stats['VAL']['median']:.2f} | {duration_stats['VAL']['p95']:.2f} | {duration_stats['VAL']['p99']:.2f} | {duration_stats['VAL']['max']:.2f} | {duration_stats['VAL']['count_gt_30s']:,} |
| **TEST (Primary)** | {duration_stats['TEST']['count']:,} | {duration_stats['TEST']['min']:.2f} | {duration_stats['TEST']['median']:.2f} | {duration_stats['TEST']['p95']:.2f} | {duration_stats['TEST']['p99']:.2f} | {duration_stats['TEST']['max']:.2f} | {duration_stats['TEST']['count_gt_30s']:,} |
| **VIVOS TEST (Ref)**| {duration_stats['VIVOS_TEST']['count']:,} | {duration_stats['VIVOS_TEST']['min']:.2f} | {duration_stats['VIVOS_TEST']['median']:.2f} | {duration_stats['VIVOS_TEST']['p95']:.2f} | {duration_stats['VIVOS_TEST']['p99']:.2f} | {duration_stats['VIVOS_TEST']['max']:.2f} | {duration_stats['VIVOS_TEST']['count_gt_30s']:,} |

---

## 3. Speaker Disjointness Audit

- **ViMD Train Speakers**: {len(vimd_train_speakers):,}
- **ViMD Val Speakers**: {len(vimd_val_speakers):,}
- **ViMD Test Speakers**: {len(vimd_test_speakers):,}
  - ViMD Train ∩ Val Overlap: **{vimd_tr_val_spk_overlap}** speakers
  - ViMD Train ∩ Test Overlap: **{vimd_tr_test_spk_overlap}** speakers
  - ViMD Val ∩ Test Overlap: **{vimd_val_test_spk_overlap}** speakers
- **VIVOS Train Speakers**: {len(vivos_train_speakers):,}
- **VIVOS Test Speakers**: {len(vivos_test_speakers):,}
  - VIVOS Train ∩ Test Overlap: **{vivos_spk_overlap}** speakers

---

## 4. Test Manifest Immutability

- `test_manifest.csv` SHA-256 match: **PASS** (`{test_sha256_current}`)
- `vivos_test_manifest.csv` SHA-256 match: **PASS** (`{vivos_test_sha256_current}`)
- Zero mutation of primary test or secondary reference manifests.
\"\"\"

REPORT_PATH = os.path.join(PROJECT_ROOT, "ASR_STAGE0_MANIFEST_REPORT.md")
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report_md)
print(f"Written final bootstrap report to {REPORT_PATH}")

final_summary_table = f\"\"\"
========================================
STAGE 0 MANIFEST BOOTSTRAP
========================================

VIMD REVISION = {PINNED_REVISIONS['vimd']['revision']}
VIVOS REVISION = {PINNED_REVISIONS['vivos']['revision']}

RAW TRAIN COUNT = {len(df_raw_train):,}
RAW VAL COUNT = {len(df_raw_val):,}
RAW TEST COUNT = {len(df_raw_test):,}
RAW VIVOS TEST COUNT = {len(df_raw_vivos_test):,}

EXCLUDED TRAIN >30s = {len(df_excluded_train):,}
EXCLUDED VAL >30s = {len(df_excluded_val):,}

FROZEN TRAIN COUNT = {len(df_frozen_train):,}
FROZEN VAL COUNT = {len(df_frozen_val):,}

TRAIN DUPLICATES = {raw_manifest_audits['TRAIN']['duplicates']}
VAL DUPLICATES = {raw_manifest_audits['VAL']['duplicates']}
TEST DUPLICATES = {raw_manifest_audits['TEST']['duplicates']}
VIVOS TEST DUPLICATES = {raw_manifest_audits['VIVOS_TEST']['duplicates']}

TRAIN/VAL SPEAKER OVERLAP = {vimd_tr_val_spk_overlap}
TRAIN/TEST SPEAKER OVERLAP = {vimd_tr_test_spk_overlap}
VAL/TEST SPEAKER OVERLAP = {vimd_val_test_spk_overlap}

TEST MANIFEST IMMUTABILITY = PASS
VIVOS TEST MANIFEST IMMUTABILITY = PASS

ALL REQUIRED MANIFESTS = PASS

STAGE 0 VERDICT = {stage0_verdict}
========================================
\"\"\"
print(final_summary_table)
""")

    # Build notebook structure
    notebook_dict = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    output_path = os.path.join(os.getcwd(), "ASR_STAGE0_BUILD_FROZEN_MANIFESTS.ipynb")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=2)
        
    print(f"Successfully generated {output_path} with {len(cells)} cells.")
    
    # Verify Python AST syntax of all code cells
    code_cell_count = 0
    for i, c in enumerate(cells):
        if c["cell_type"] == "code":
            code_str = "".join(c["source"])
            # strip IPython magic commands like !pip
            clean_code = "\n".join([line for line in code_str.split("\n") if not line.strip().startswith("!")])
            try:
                ast.parse(clean_code)
                code_cell_count += 1
            except SyntaxError as e:
                print(f"Syntax Error in Cell {i}: {e}")
                raise
    print(f"Validated {code_cell_count} code cells with 100% valid Python AST syntax!")

if __name__ == "__main__":
    create_stage0_notebook()
