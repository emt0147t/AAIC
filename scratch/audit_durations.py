# Script to inspect durations of VIVOS and ViMD
import time
import io
import numpy as np
from scipy.io import wavfile
from datasets import load_dataset, Audio

print("Inspecting VIVOS durations (first 100 samples)...")
ds_vivos = load_dataset("thanhduycao/vivos_ng_only", split="train", streaming=True).cast_column('audio', Audio(decode=False))
vivos_durs = []
for i, s in enumerate(ds_vivos):
    sr, data = wavfile.read(io.BytesIO(s["audio"]["bytes"]))
    vivos_durs.append(len(data) / sr)
    if i >= 99:
        break

print(f"VIVOS (100 samples): min={min(vivos_durs):.2f}s, max={max(vivos_durs):.2f}s, median={np.median(vivos_durs):.2f}s, count>30s={sum(1 for d in vivos_durs if d > 30)}")

print("\nInspecting ViMD durations (first 100 train samples)...")
ds_vimd = load_dataset("nguyendv02/ViMD_Dataset", split="train", streaming=True).cast_column('audio', Audio(decode=False))
vimd_durs = []
for i, s in enumerate(ds_vimd):
    sr, data = wavfile.read(io.BytesIO(s["audio"]["bytes"]))
    vimd_durs.append(len(data) / sr)
    if i >= 99:
        break

print(f"ViMD (100 train samples): min={min(vimd_durs):.2f}s, max={max(vimd_durs):.2f}s, median={np.median(vimd_durs):.2f}s, p95={np.percentile(vimd_durs, 95):.2f}s, count>30s={sum(1 for d in vimd_durs if d > 30)}")

print("\nInspecting ViMD valid durations (first 100 valid samples)...")
ds_vimd_val = load_dataset("nguyendv02/ViMD_Dataset", split="valid", streaming=True).cast_column('audio', Audio(decode=False))
vimd_val_durs = []
for i, s in enumerate(ds_vimd_val):
    sr, data = wavfile.read(io.BytesIO(s["audio"]["bytes"]))
    vimd_val_durs.append(len(data) / sr)
    if i >= 99:
        break

print(f"ViMD valid (100 samples): min={min(vimd_val_durs):.2f}s, max={max(vimd_val_durs):.2f}s, median={np.median(vimd_val_durs):.2f}s, p95={np.percentile(vimd_val_durs, 95):.2f}s, count>30s={sum(1 for d in vimd_val_durs if d > 30)}")
