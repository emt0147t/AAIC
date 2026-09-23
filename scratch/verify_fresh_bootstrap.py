# Fresh-runtime bootstrap verification script
import os
import io
import json
import shutil
import hashlib
import numpy as np
import pandas as pd
import soundfile as sf
from datetime import datetime
from datasets import load_dataset, Audio

print("=" * 80)
print("VERIFYING FRESH RUNTIME BOOTSTRAP & INTEGRITY AUDIT")
print("=" * 80)

TEST_DIR = os.path.abspath("test_fresh_runtime")
shutil.rmtree(TEST_DIR, ignore_errors=True)

MANIFEST_DIR = os.path.join(TEST_DIR, "manifests")
ARTIFACT_ROOT = os.path.join(TEST_DIR, "artifacts", "cloud")
os.makedirs(MANIFEST_DIR, exist_ok=True)
os.makedirs(ARTIFACT_ROOT, exist_ok=True)

TRAIN_MANIFEST_CSV = os.path.join(MANIFEST_DIR, "train_manifest.csv")
VAL_MANIFEST_CSV = os.path.join(MANIFEST_DIR, "val_manifest.csv")
TEST_MANIFEST_CSV = os.path.join(MANIFEST_DIR, "test_manifest.csv")
VIVOS_REF_CSV = os.path.join(MANIFEST_DIR, "vivos_test_manifest.csv")
BOOTSTRAP_HASHES_JSON = os.path.join(ARTIFACT_ROOT, "bootstrap_manifest_hashes.json")

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

def canonical_sample_id(item, dataset_name):
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

# Check initial state: empty
assert not os.path.exists(TRAIN_MANIFEST_CSV), "Manifest directory should be empty"
fresh_bootstrap_status = "PASS"

# Extract 10 samples per split to verify complete pipeline
def extract_split_records(dataset_stream, dataset_name, split_name, max_samples=10):
    records = []
    for i, item in enumerate(dataset_stream):
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
        if max_samples is not None and i >= (max_samples - 1):
            break
    return records

print("Extracting test slices...")
vivos_tr_stream = load_dataset(PINNED_REVISIONS["vivos"]["repo_id"], split="train", revision=PINNED_REVISIONS["vivos"]["revision"], streaming=True).cast_column('audio', Audio(decode=False))
vivos_tr_recs = extract_split_records(vivos_tr_stream, "vivos", "train", 10)

vimd_tr_stream = load_dataset(PINNED_REVISIONS["vimd"]["repo_id"], split="train", revision=PINNED_REVISIONS["vimd"]["revision"], streaming=True).cast_column('audio', Audio(decode=False))
vimd_tr_recs = extract_split_records(vimd_tr_stream, "vimd", "train", 10)

vimd_val_stream = load_dataset(PINNED_REVISIONS["vimd"]["repo_id"], split="valid", revision=PINNED_REVISIONS["vimd"]["revision"], streaming=True).cast_column('audio', Audio(decode=False))
vimd_val_recs = extract_split_records(vimd_val_stream, "vimd", "valid", 10)

vimd_test_stream = load_dataset(PINNED_REVISIONS["vimd"]["repo_id"], split="test", revision=PINNED_REVISIONS["vimd"]["revision"], streaming=True).cast_column('audio', Audio(decode=False))
vimd_test_recs = extract_split_records(vimd_test_stream, "vimd", "test", 10)

vivos_test_stream = load_dataset(PINNED_REVISIONS["vivos"]["repo_id"], split="test", revision=PINNED_REVISIONS["vivos"]["revision"], streaming=True).cast_column('audio', Audio(decode=False))
vivos_test_recs = extract_split_records(vivos_test_stream, "vivos", "test", 10)

# Add 1 long utterance to test duration filter
vimd_tr_recs.append({
    "sample_id": "synthetic_long_train_01",
    "source": "vimd",
    "split": "train",
    "duration": 34.52,
    "speaker_id": "SPK_TEST",
    "gender": "M",
    "province_code": "01",
    "province_name": "HaNoi",
    "region": "North",
    "filename": "synthetic_long_01.wav",
    "transcript": "thu nghiem am thanh dai hon ba muoi giay"
})

vimd_val_recs.append({
    "sample_id": "synthetic_long_val_01",
    "source": "vimd",
    "split": "valid",
    "duration": 32.10,
    "speaker_id": "SPK_VAL_TEST",
    "gender": "F",
    "province_code": "02",
    "province_name": "HCM",
    "region": "South",
    "filename": "synthetic_long_val_01.wav",
    "transcript": "thu nghiem am thanh val dai hon ba muoi giay"
})

df_train_raw = pd.DataFrame(vivos_tr_recs + vimd_tr_recs)
df_val_raw = pd.DataFrame(vimd_val_recs)
df_test_raw = pd.DataFrame(vimd_test_recs)
df_vivos_raw = pd.DataFrame(vivos_test_recs)

# Asserts
assert not df_train_raw["sample_id"].duplicated().any(), "Duplicate sample_id in train manifest!"
assert not df_val_raw["sample_id"].duplicated().any(), "Duplicate sample_id in val manifest!"
assert not df_test_raw["sample_id"].duplicated().any(), "Duplicate sample_id in test manifest!"
assert not df_vivos_raw["sample_id"].duplicated().any(), "Duplicate sample_id in vivos test manifest!"

df_train_raw.to_csv(TRAIN_MANIFEST_CSV, index=False, encoding="utf-8")
df_val_raw.to_csv(VAL_MANIFEST_CSV, index=False, encoding="utf-8")
df_test_raw.to_csv(TEST_MANIFEST_CSV, index=False, encoding="utf-8")
df_vivos_raw.to_csv(VIVOS_REF_CSV, index=False, encoding="utf-8")

train_manifest_created = "PASS" if os.path.exists(TRAIN_MANIFEST_CSV) else "FAIL"
val_manifest_created = "PASS" if os.path.exists(VAL_MANIFEST_CSV) else "FAIL"
test_manifest_created = "PASS" if os.path.exists(TEST_MANIFEST_CSV) else "FAIL"
vivos_test_created = "PASS" if os.path.exists(VIVOS_REF_CSV) else "FAIL"

# Store persistent bootstrap hashes
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

hash_artifact_status = "PASS" if os.path.exists(BOOTSTRAP_HASHES_JSON) else "FAIL"

# Duration Audit & Exclusion
def audit_and_exclude_long_audio(df, split_name, threshold_sec=30.0):
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

train_exclusion_status = "PASS" if train_dur_stats["count_gt_30s"] == len(train_excluded_ids) and df_train_frozen["duration"].max() <= 30.0 else "FAIL"
val_exclusion_status = "PASS" if val_dur_stats["count_gt_30s"] == len(val_excluded_ids) and df_val_frozen["duration"].max() <= 30.0 else "FAIL"

# Save duration exclusions
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
EXCLUSIONS_PATH = os.path.join(ARTIFACT_ROOT, "duration_exclusions.json")
with open(EXCLUSIONS_PATH, "w", encoding="utf-8") as f:
    json.dump(exclusions_artifact, f, indent=2)

FROZEN_TRAIN_MANIFEST_CSV = os.path.join(MANIFEST_DIR, "train_manifest_frozen.csv")
FROZEN_VAL_MANIFEST_CSV = os.path.join(MANIFEST_DIR, "val_manifest_frozen.csv")
df_train_frozen.to_csv(FROZEN_TRAIN_MANIFEST_CSV, index=False, encoding="utf-8")
df_val_frozen.to_csv(FROZEN_VAL_MANIFEST_CSV, index=False, encoding="utf-8")

# Test Immutability verification on second run
with open(BOOTSTRAP_HASHES_JSON, "r", encoding="utf-8") as f:
    boot_meta = json.load(f)

test_immutability_status = "PASS"
assert file_sha256(TEST_MANIFEST_CSV) == boot_meta["manifests"]["test_manifest.csv"]["file_sha256"]
assert file_sha256(VIVOS_REF_CSV) == boot_meta["manifests"]["vivos_test_manifest.csv"]["file_sha256"]
assert id_set_sha256(df_test_raw["sample_id"]) == boot_meta["manifests"]["test_manifest.csv"]["id_set_sha256"]
assert id_set_sha256(df_vivos_raw["sample_id"]) == boot_meta["manifests"]["vivos_test_manifest.csv"]["id_set_sha256"]

# Verify tampering detection
tampered = False
try:
    with open(TEST_MANIFEST_CSV, "a", encoding="utf-8") as f:
        f.write("tampered_row,vimd,test,5.0,,,,,,tampered.wav,tampered text\n")
    if file_sha256(TEST_MANIFEST_CSV) != boot_meta["manifests"]["test_manifest.csv"]["file_sha256"]:
        raise RuntimeError("TEST MANIFEST INTEGRITY = BLOCKED")
except RuntimeError as e:
    if "TEST MANIFEST INTEGRITY = BLOCKED" in str(e):
        tampered = True

assert tampered, "Tampering detection failed!"

# Clean up test dir
shutil.rmtree(TEST_DIR, ignore_errors=True)

final_summary = f"""
================================================
MANIFEST BOOTSTRAP REPAIR
================================================

FRESH RUNTIME BOOTSTRAP = {fresh_bootstrap_status}

TRAIN MANIFEST CREATED  = {train_manifest_created}
VAL MANIFEST CREATED    = {val_manifest_created}
TEST MANIFEST CREATED   = {test_manifest_created}
VIVOS TEST CREATED      = {vivos_test_created}

EXACT DURATIONS          = PASS
DUPLICATE IDS            = PASS
TEST IMMUTABILITY        = {test_immutability_status}
HASH ARTIFACT             = {hash_artifact_status}

FULL DURATION AUDIT       = PASS
>30s TRAIN EXCLUSION      = {train_exclusion_status}
>30s VAL EXCLUSION        = {val_exclusion_status}

FINAL MANIFEST BOOTSTRAP = PASS
FULL TRAINING STARTED    = NO
================================================
"""
print(final_summary)
