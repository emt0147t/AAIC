"""Gate 4 — Small sample inspection (metadata-only, UTF-8 safe)."""
import json, os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path
from datasets import load_dataset, Audio

OUT = Path(r"d:\Vietnamese_ASR_Week6\asr_experiment\sample_verification")
OUT.mkdir(parents=True, exist_ok=True)

def inspect_dataset(name, repo, split, n=5):
    print(f"\n{'='*60}")
    print(f"  {name}  (repo={repo}, split={split}, n={n})")
    print(f"{'='*60}")
    
    try:
        ds = load_dataset(repo, split=split, streaming=True)
        # Disable audio decoding
        if "audio" in ds.column_names:
            ds = ds.cast_column("audio", Audio(decode=False))
    except Exception as e:
        try:
            ds = load_dataset(repo, split=split, streaming=True)
            if "audio" in ds.column_names:
                ds = ds.remove_columns(["audio"])
        except Exception as e2:
            print(f"  ERROR: {e2}")
            return []
    
    rows = []
    for i, sample in enumerate(ds):
        if i >= n:
            break
        row = {"dataset": name, "idx": i}
        
        # Audio metadata
        if "audio" in sample:
            a = sample["audio"]
            if isinstance(a, dict):
                row["audio_path"] = str(a.get("path", ""))[:80]
                row["audio_bytes"] = len(a.get("bytes", b"")) if a.get("bytes") else 0
                row["sampling_rate"] = a.get("sampling_rate", "N/A")
        
        # Transcript
        for k in ["sentence", "text", "transcript", "transcription"]:
            if k in sample and sample[k]:
                row["transcript_field"] = k
                row["transcript"] = str(sample[k])[:150]
                row["transcript_chars"] = len(str(sample[k]))
                row["transcript_words"] = len(str(sample[k]).split())
                break
        
        # Speaker
        for k in ["speaker_id", "speakerID"]:
            if k in sample:
                row["speaker_id"] = str(sample[k])
                break
        
        # Metadata
        for k in ["province","region","gender","path","filename","set","length","duration","source","id","label"]:
            if k in sample:
                v = sample[k]
                row[k] = float(v) if isinstance(v, (int, float)) else str(v)[:80]
        
        row["columns"] = sorted(sample.keys())
        rows.append(row)
        
        spk = row.get("speaker_id", "?")
        dur = row.get("length", row.get("duration", "?"))
        sr = row.get("sampling_rate", "?")
        prov = row.get("province", "")
        region = row.get("region", "")
        txt_preview = row.get("transcript", "")[:60]
        print(f"  [{i}] dur={dur} sr={sr} spk={spk} prov={prov} region={region}")
        print(f"      text: {txt_preview}")
    
    if rows:
        print(f"  Columns: {rows[0]['columns']}")
    
    outpath = OUT / f"{name}_samples.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved: {outpath}")
    return rows

# === ViMD ===
print("DATASET 1: ViMD (canonical: nguyendv02/ViMD_Dataset)")
inspect_dataset("ViMD_train", "nguyendv02/ViMD_Dataset", "train", n=5)
inspect_dataset("ViMD_test", "nguyendv02/ViMD_Dataset", "test", n=5)
inspect_dataset("ViMD_valid", "nguyendv02/ViMD_Dataset", "valid", n=3)

# === VietMed labeled ===
print("\n\nDATASET 2: VietMed labeled (canonical: leduckhai/VietMed)")
inspect_dataset("VietMed_train", "leduckhai/VietMed", "train", n=5)

# === VIVOS (use Zenodo mirror approach - just check structure) ===
print("\n\nDATASET 3: VIVOS (from HF API metadata - no download needed)")
print("  Schema verified from HF API:")
print("  - Columns: speaker_id (string), path (string), audio (Audio 16kHz), sentence (string)")
print("  - Train: 11,660 utterances")
print("  - Test: 760 utterances")
print("  - Sampling rate: 16,000 Hz")
print("  - License: CC BY-NC-SA 4.0 (Zenodo)")
print("  - Audio format: WAV PCM 16-bit mono")
print("  NOTE: HF mirror returns 401 for file download; Zenodo is canonical")

# Save VIVOS metadata manually
vivos_meta = {
    "dataset": "VIVOS",
    "schema": {"columns": ["speaker_id", "path", "audio", "sentence"]},
    "splits": {"train": 11660, "test": 760},
    "audio": {"sampling_rate": 16000, "format": "WAV PCM 16-bit mono"},
    "license": "CC BY-NC-SA 4.0",
    "source": "Zenodo + HF (AILAB-VNUHCM/vivos)",
    "note": "HF file download requires auth; schema verified via API"
}
with open(OUT / "VIVOS_metadata.json", "w") as f:
    json.dump(vivos_meta, f, indent=2)
print(f"  Saved: {OUT / 'VIVOS_metadata.json'}")

# === FOSD (Mendeley — no HF streaming available) ===
print("\n\nDATASET 4: FOSD (Mendeley: 10.17632/k9sxg2twv4.4)")
fosd_meta = {
    "dataset": "FOSD",
    "description": "25,921 recorded Vietnamese speeches with transcripts",
    "audio_format": "MP3",
    "transcript_format": "TXT (UTF-8)",
    "license": "FPT Public License (permissive, similar to MIT)",
    "copyright": "Copyright 2018 FPT Corporation",
    "subsets": 3,
    "total_hours": "~30h",
    "speaker_metadata": "Limited/inconsistent",
    "official_splits": "None",
    "source": "Mendeley Data (k9sxg2twv4/4)",
    "note": "Audio is MP3 not WAV — needs conversion to 16kHz WAV for ASR"
}
with open(OUT / "FOSD_metadata.json", "w") as f:
    json.dump(fosd_meta, f, indent=2)
print("  FOSD verified from Mendeley page:")
print("  - Audio format: MP3 (NOT WAV — needs conversion)")
print("  - Transcript format: TXT (UTF-8)")
print("  - License: FPT Public License (permissive, includes commercial use)")
print("  - Copyright: 2018 FPT Corporation")
print("  - No official splits")
print(f"  Saved: {OUT / 'FOSD_metadata.json'}")

print("\n\n" + "="*60)
print("GATE 4 SAMPLE INSPECTION COMPLETE")
print(f"Output: {OUT}")
