from asr.audio import load_audio


PATH = r"D:\Vietnamese_ASR_Week6\vimd_demo_9\audio\vimd_01.wav"

audio, sr = load_audio(PATH)

print("Sample rate:", sr)
print("Shape:", audio.shape)
print("Dtype:", audio.dtype)
print("Min:", float(audio.min()))
print("Max:", float(audio.max()))
print("Samples:", len(audio))
print("Duration:", len(audio) / sr)

assert sr == 16000
assert audio.ndim == 1
assert str(audio.dtype) == "float32"

print("\nPREPROCESSING BASIC TEST: PASS")