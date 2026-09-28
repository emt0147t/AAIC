import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from asr.audio import load_audio


SR = 44100
DURATION = 1.0
N = int(SR * DURATION)

t = np.arange(N) / SR

left = 0.2 * np.sin(2 * np.pi * 440 * t)
right = 0.1 * np.sin(2 * np.pi * 880 * t)

stereo = np.column_stack([left, right]).astype(np.float32)

path = Path(tempfile.gettempdir()) / "asr_test_stereo_44k.wav"

sf.write(path, stereo, SR)

audio, sr = load_audio(str(path))

print("Input:")
print("  SR:", SR)
print("  Shape:", stereo.shape)
print("  Dtype:", stereo.dtype)

print("\nOutput:")
print("  SR:", sr)
print("  Shape:", audio.shape)
print("  Dtype:", audio.dtype)
print("  Min:", float(audio.min()))
print("  Max:", float(audio.max()))
print("  Duration:", len(audio) / sr)

assert sr == 16000
assert audio.ndim == 1
assert audio.dtype == np.float32
assert np.all(np.isfinite(audio))

print("\nSTEREO -> MONO + RESAMPLE: PASS")