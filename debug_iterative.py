import traceback
import numpy as np

from asr.audio import load_audio
from services.iterative_service import IterativeManipulationService
from services.asr_service import ASRService

AUDIO = r"D:\Vietnamese_ASR_Week6\vimd_demo_9\audio\vimd_01.wav"

print("=" * 70)
print("ITERATIVE BACKEND DEBUG")
print("=" * 70)

try:
    print("\n[1] Loading audio...")
    audio, sr = load_audio(AUDIO)

    print("sample_rate:", sr)
    print("shape:", audio.shape)
    print("dtype:", audio.dtype)
    print("duration:", len(audio) / sr)

    asr_service = ASRService(default_model_id="vinai/PhoWhisper-small")

    service = IterativeManipulationService(
        asr_service=asr_service,
        base_export_dir="exports"
    )

    print("\n[2] Executing chain...")

    records, audio_map, meta = service.execute_chain(
        audio_16k=audio,
        k=3,
        strategy="random",
        base_seed=42,
        reference_transcript=None,
        candidates_per_step=3,
        enabled_operators=["additive_noise"],
    )

    print("\n[3] SUCCESS")
    print("records:", len(records))
    print("audio_map keys:", list(audio_map.keys()))
    print("meta:", meta)

    print("\n[4] RECORDS")
    for r in records:
        print(
            r.iteration,
            r.manipulation_name,
            "qc=", r.qc_passed,
            "changed=", r.waveform_changed,
            "max_delta=", r.max_abs_delta,
            "mean_delta=", r.mean_abs_delta,
        )

except Exception as e:
    print("\n!!! BACKEND ERROR !!!")
    print(type(e).__name__)
    print(str(e))
    print("\nFULL TRACEBACK:")
    traceback.print_exc()