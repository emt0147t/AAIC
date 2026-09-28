import hashlib
import os
import soundfile as sf
import numpy as np

from app import handle_iterative_manipulation

AUDIO = r"D:\Vietnamese_ASR_Week6\vimd_demo_9\audio\vimd_01.wav"

print("=" * 70)
print("ITERATIVE UI CALLBACK DEBUG")
print("=" * 70)

result = handle_iterative_manipulation(
    audio_file=AUDIO,
    k_val=3,
    strategy="random",
    cand_m=3,
    master_seed=42,
    ref_transcript="",
    enabled_ops=["additive_noise"],
)

print("\n[1] RETURN OBJECT")
print("type:", type(result))
print("length:", len(result))

print("\n[2] RETURN VALUES")
for i, value in enumerate(result):
    if hasattr(value, "shape"):
        print(i, "numpy:", value.shape)
    elif isinstance(value, str):
        print(i, "str:", value[:300])
    else:
        print(i, type(value), value)

print("\n[3] AUDIO PREVIEW PATHS")

paths = result[3:9]

for i, path in enumerate(paths):
    stage = f"x_{i}"

    print(f"\n{stage}:")
    print("  path:", path)

    if path is None:
        print("  STATUS: NONE")
        continue

    print("  exists:", os.path.exists(path))

    if not os.path.exists(path):
        continue

    data, sr = sf.read(path, dtype="float32")

    sha = hashlib.sha256(
        np.asarray(data, dtype=np.float32).tobytes()
    ).hexdigest()

    print("  sr:", sr)
    print("  shape:", data.shape)
    print("  duration:", len(data) / sr)
    print("  peak:", float(np.max(np.abs(data))))
    print("  rms:", float(np.sqrt(np.mean(data ** 2))))
    print("  sha256:", sha)

print("\n[4] DIRECT FILE COMPARISON")

existing = []

for i, path in enumerate(paths[:4]):
    if path and os.path.exists(path):
        data, sr = sf.read(path, dtype="float32")
        existing.append((i, data))

if existing:
    x0 = existing[0][1]

    for i, data in existing[1:]:
        n = min(len(x0), len(data))
        diff = np.abs(data[:n] - x0[:n])

        print(
            f"x0 -> x{i}: "
            f"max_delta={float(np.max(diff)):.8f}, "
            f"mean_delta={float(np.mean(diff)):.8f}"
        )