import sys, os
from pathlib import Path
sys.path.insert(0, os.path.abspath("vietnamese_asr_app"))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import soundfile as sf
import hashlib
from services.asr_service import ASRService
from services.iterative_service import IterativeManipulationService

def main():
    asr = ASRService(default_model_id="vinai/PhoWhisper-tiny")
    service = IterativeManipulationService(asr_service=asr)

    audio_path = "vimd_demo_9/audio/vimd_01.wav"
    y, sr = sf.read(audio_path)
    print(f"Original audio loaded: shape={y.shape}, sr={sr}, sha={hashlib.sha256(y.tobytes()).hexdigest()}")

    for strat in ["random", "best_preserving"]:
        print(f"\n==========================================")
        print(f"TESTING STRATEGY: {strat}")
        print(f"==========================================")
        records, audio_map, meta = service.execute_chain(
            audio_16k=y,
            k=4,
            strategy=strat,
            base_seed=42,
        )
        for rec in records:
            h = hashlib.sha256(audio_map[rec.iteration].tobytes()).hexdigest() if rec.iteration in audio_map else "N/A"
            print(f"Iter {rec.iteration}: op={rec.manipulation_name}, params={rec.parameters}, sha={h[:16]}, score={rec.score}")
            print(f"   ASR: \"{rec.asr_raw_transcript}\"")
        
        # Check if audio_map arrays are distinct
        for i in range(len(audio_map)):
            for j in range(i+1, len(audio_map)):
                same = np.array_equal(audio_map[i], audio_map[j])
                diff_len = len(audio_map[i]) != len(audio_map[j])
                if diff_len:
                    diff_str = f"Diff len ({len(audio_map[i])} vs {len(audio_map[j])})"
                else:
                    max_diff = np.max(np.abs(audio_map[i] - audio_map[j]))
                    diff_str = f"MaxDiff={max_diff:.6f}"
                print(f"audio_map[{i}] vs audio_map[{j}]: Identical={same}, {diff_str}")

if __name__ == "__main__":
    main()
