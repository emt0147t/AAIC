import numpy as np

from asr.audio import load_audio


sr = 16000

audio = (
    0.2
    * np.sin(
        2
        * np.pi
        * 440
        * np.arange(sr)
        / sr
    )
).astype(np.float32)

output, out_sr = load_audio((sr, audio))

print("Input:")
print("  SR:", sr)
print("  Shape:", audio.shape)
print("  Dtype:", audio.dtype)

print("\nOutput:")
print("  SR:", out_sr)
print("  Shape:", output.shape)
print("  Dtype:", output.dtype)
print("  Min:", float(output.min()))
print("  Max:", float(output.max()))
print("  Duration:", len(output) / out_sr)

assert out_sr == 16000
assert output.ndim == 1
assert output.dtype == np.float32
assert np.all(np.isfinite(output))

print("\nGRADIO TUPLE INPUT: PASS")