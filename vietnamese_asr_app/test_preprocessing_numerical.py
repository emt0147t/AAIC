import numpy as np

from asr.audio import load_audio


PATH = r"D:\Vietnamese_ASR_Week6\vimd_demo_9\audio\vimd_01.wav"

audio, sr = load_audio(PATH)

finite = np.all(np.isfinite(audio))
nan_count = int(np.isnan(audio).sum())
inf_count = int(np.isinf(audio).sum())
peak = float(np.max(np.abs(audio)))
rms = float(np.sqrt(np.mean(audio ** 2)))

print("Sample rate:", sr)
print("Shape:", audio.shape)
print("Dtype:", audio.dtype)
print("Finite:", finite)
print("NaN count:", nan_count)
print("Inf count:", inf_count)
print("Peak:", peak)
print("RMS:", rms)

assert finite
assert nan_count == 0
assert inf_count == 0
assert audio.dtype == np.float32
assert audio.ndim == 1

print("\nNUMERICAL INTEGRITY: PASS")