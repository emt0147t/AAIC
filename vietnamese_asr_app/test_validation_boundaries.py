import numpy as np
from asr.audio import validate_audio


cases = {
    "exact_0_2s": np.full(3200, 0.05, dtype=np.float32),
    "exact_30s": np.full(480000, 0.05, dtype=np.float32),
    "over_30s": np.full(480001, 0.05, dtype=np.float32),
}


for name, audio in cases.items():
    result = validate_audio(audio, 16000)

    print(f"\n{name}")
    print("  is_valid :", result.is_valid)
    print("  duration :", result.duration_sec)
    print("  error    :", result.error_message)