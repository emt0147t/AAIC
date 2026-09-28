import io

import numpy as np
import soundfile as sf

from asr.audio import load_audio


SR = 16000
N = SR

t = np.arange(N) / SR
signal = (
    0.2 * np.sin(2 * np.pi * 440 * t)
).astype(np.float32)

buffer = io.BytesIO()

sf.write(
    buffer,
    signal,
    SR,
    format="WAV",
    subtype="PCM_16",
)

buffer.seek(0)

audio, sr = load_audio(buffer)

print("SR:", sr)
print("Shape:", audio.shape)
print("Dtype:", audio.dtype)
print("Duration:", len(audio) / sr)

assert sr == 16000
assert audio.ndim == 1
assert audio.dtype == np.float32
assert np.all(np.isfinite(audio))

print("\nBYTESIO INPUT: PASS")