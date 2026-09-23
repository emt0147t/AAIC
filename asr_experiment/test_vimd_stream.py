import time, io, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from scipy.io import wavfile
from datasets import load_dataset, Audio

t0 = time.time()
print("Starting ViMD train stream...")
ds = load_dataset('nguyendv02/ViMD_Dataset', split='train', streaming=True).cast_column('audio', Audio(decode=False))
count = 0
for s in ds:
    count += 1
    sr, data = wavfile.read(io.BytesIO(s['audio']['bytes']))
    print(f"[{count}] {s['filename']}: sr={sr}, dur={len(data)/sr:.2f}s, text={s['text'][:40]}")
    if count >= 5:
        break
dt = time.time() - t0
print(f"Successfully streamed {count} ViMD samples in {dt:.2f}s")
