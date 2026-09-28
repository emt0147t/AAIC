from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf

from asr.audio import load_audio


SR = 16000
N = SR

t = np.arange(N) / SR
signal = 0.5 * np.sin(2 * np.pi * 440 * t)

pcm16 = np.int16(signal * 32767)

path = Path(tempfile.gettempdir()) / "asr_test_pcm16.wav"

sf.write(path, pcm16, SR, subtype="PCM_16")

audio, sr = load_audio(str(path))

print("SR:", sr)
print("Shape:", audio.shape)
print("Dtype:", audio.dtype)
print("Min:", float(audio.min()))
print("Max:", float(audio.max()))

assert sr == 16000
assert audio.ndim == 1
assert audio.dtype == np.float32
assert np.all(np.isfinite(audio))
assert float(audio.max()) <= 1.0 + 1e-5
assert float(audio.min()) >= -1.0 - 1e-5

print("\nINTEGER PCM -> FLOAT32: PASS")