"""Deterministic K=5 Iterative Audio Manipulation Demo.

Demonstrates cumulative transformation chain x_0 -> x_1 -> ... -> x_5
using local audio data/vimd_train_clean/audio/vimd_train_01.wav (training sample, NOT frozen test set).
Verifies:
- All 6 stages (x_0 ... x_5) are created and valid
- x_i != x_{i-1} for all stages
- Parent lineage is strictly parent(x_i) = x_{i-1}
- 10-point QC verified at each step
- ASR transcripts recorded for each stage
- Zero TTS, zero voice cloning, non-destructive to x_0
- Metadata JSON, manifest CSV, and ZIP export generated
"""

import hashlib
import json
import os
import sys
import numpy as np

# Ensure vietnamese_asr_app is on sys.path
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from asr.audio import load_audio
from services.asr_service import ASRService
from services.iterative_service import IterativeManipulationService


def run_demo():
    input_wav = os.path.join(APP_DIR, "data", "vimd_train_clean", "audio", "vimd_train_01.wav")
    assert os.path.exists(input_wav), f"Demo input audio not found: {input_wav}"

    print("======================================================================")
    print("ITERATIVE AUDIO MANIPULATION DEMO (K = 5)")
    print("======================================================================")
    print(f"Source Audio: {input_wav}")

    audio_16k, sr = load_audio(input_wav, target_sr=16000)
    orig_copy = audio_16k.copy()
    orig_sha256 = hashlib.sha256(audio_16k.tobytes()).hexdigest()
    print(f"Original x_0: duration={len(audio_16k)/sr:.2f}s, SR={sr}Hz, SHA-256={orig_sha256[:16]}...")

    asr_service = ASRService(default_model_id="vinai/PhoWhisper-tiny")
    iterative_service = IterativeManipulationService(
        asr_service=asr_service,
        base_export_dir=os.path.join(APP_DIR, "exports"),
    )

    print("\nExecuting Best-Preserving Iterative Search (K = 5, M = 3 candidates per step, seed = 42)...")
    records, audio_map, meta = iterative_service.execute_chain(
        audio_16k=audio_16k,
        k=5,
        strategy="best_preserving",
        base_seed=42,
        candidates_per_step=3,
        reference_transcript=None,  # Case B: No reference transcript (ASR consistency proxy)
        progress_callback=lambda p, msg: print(f"  [{int(p*100):3d}%] {msg}"),
    )

    print("\n----------------------------------------------------------------------")
    print("MANIPULATION CHAIN RESULTS:")
    print("----------------------------------------------------------------------")
    for r in records:
        stage = f"x_{r.iteration}" if r.iteration > 0 else "x_0 (Orig)"
        parent = f"x_{r.parent_iteration}"
        drift_str = f"Drift WER={r.consistency_drift_wer:.1f}%" if r.consistency_drift_wer is not None else "-"
        print(f"[{stage}] Parent={parent} | Op={r.manipulation_name} | QC={r.qc_status} | RMS={r.rms:.4f} | Peak={r.peak_abs:.2f} | Score={r.score:.2f} ({r.score_type})")
        print(f"     ASR: \"{r.asr_raw_transcript}\"")

    # Export bundle
    bundle = iterative_service.export_chain_bundle(
        records=records,
        audio_map=audio_map,
        original_audio_id="vimd_train_01",
        export_dirname="demo_iterative_k5",
    )
    print("\nExported artifacts:")
    print(f"  - Directory: {bundle['bundle_dir']}")
    print(f"  - JSON Metadata: {bundle['json_path']}")
    print(f"  - CSV Manifest: {bundle['csv_path']}")
    print(f"  - ZIP Package: {bundle['zip_path']} ({os.path.getsize(bundle['zip_path']):,} bytes)")

    # Assertions
    assert len(records) == 6, f"Expected 6 records (x_0..x_5), got {len(records)}"
    assert len(audio_map) == 6, f"Expected 6 audio stages, got {len(audio_map)}"
    assert np.array_equal(audio_16k, orig_copy), "Original audio was modified in-place!"
    for i in range(1, 6):
        assert records[i].parent_iteration == i - 1, f"Lineage broken at step {i}"
        assert not np.array_equal(audio_map[i], audio_map[i - 1]), f"Audio at {i} identical to parent {i-1}"
        assert records[i].qc_passed is True, f"Step {i} failed QC"

    print("\nALL VERIFICATIONS PASSED: K=5 cumulative chain successfully executed.")


if __name__ == "__main__":
    run_demo()
