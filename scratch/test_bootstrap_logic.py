# Isolated test for manifest bootstrap logic
import os
import io
import json
import shutil
import hashlib
import numpy as np
import pandas as pd
import soundfile as sf
from datasets import load_dataset, Audio

TEST_DIR = os.path.abspath("test_fresh_runtime")
MANIFEST_DIR = os.path.join(TEST_DIR, "manifests")
ARTIFACT_ROOT = os.path.join(TEST_DIR, "artifacts", "cloud")

shutil.rmtree(TEST_DIR, ignore_errors=True)
os.makedirs(MANIFEST_DIR, exist_ok=True)
os.makedirs(ARTIFACT_ROOT, exist_ok=True)

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
            raise ValueError(f"Invalid duration value: {dur}")
        return round(dur, 4)
    except Exception as e:
        raise RuntimeError(f"EXACT DURATION = BLOCKED: Could not determine exact duration for sample {sid}: {e}")

print("Testing bootstrap helper with real streaming sample slices...")
# Test extraction of 5 samples per split to verify schema and duration
splits_to_test = [
    ("vivos", "train", PINNED_REVISIONS["vivos"]["repo_id"], PINNED_REVISIONS["vivos"]["revision"]),
    ("vimd", "train", PINNED_REVISIONS["vimd"]["repo_id"], PINNED_REVISIONS["vimd"]["revision"]),
    ("vimd", "valid", PINNED_REVISIONS["vimd"]["repo_id"], PINNED_REVISIONS["vimd"]["revision"]),
    ("vimd", "test", PINNED_REVISIONS["vimd"]["repo_id"], PINNED_REVISIONS["vimd"]["revision"]),
    ("vivos", "test", PINNED_REVISIONS["vivos"]["repo_id"], PINNED_REVISIONS["vivos"]["revision"])
]

records_by_key = {}
for dname, sname, repo, rev in splits_to_test:
    stream = load_dataset(repo, split=sname, revision=rev, streaming=True).cast_column('audio', Audio(decode=False))
    recs = []
    for i, item in enumerate(stream):
        sid = canonical_sample_id(item, dname)
        dur = get_exact_audio_duration(item["audio"], sid)
        if dname == "vimd":
            rec = {
                "sample_id": sid,
                "source": "vimd",
                "split": sname,
                "duration": dur,
                "speaker_id": str(item.get("speakerID", "")).strip(),
                "gender": str(item.get("gender", "")).strip(),
                "province_code": str(item.get("province_code", "")).strip(),
                "province_name": str(item.get("province_name", "")).strip(),
                "region": str(item.get("region", "")).strip(),
                "filename": str(item.get("filename", "")).strip(),
                "transcript": str(item.get("text", "")).strip()
            }
        else:
            rec = {
                "sample_id": sid,
                "source": "vivos",
                "split": sname,
                "duration": dur,
                "speaker_id": str(item.get("speaker_id", "")).strip(),
                "gender": "",
                "province_code": "",
                "province_name": "",
                "region": "",
                "filename": str(item.get("path", "")).strip(),
                "transcript": str(item.get("sentence", "")).strip()
            }
        recs.append(rec)
        if i >= 4:
            break
    records_by_key[f"{dname}_{sname}"] = recs
    print(f"Sample slice for {dname} {sname}: {len(recs)} records extracted with exact durations.")

print("All sample slices extracted successfully!")
