"""Real End-to-End Voice Test for 'Xin chào tôi tên là Thanh'.

Verifies:
1. Audio loading & 7-point validation
2. Model transcription execution without max_target_positions exception
3. Non-empty output
4. Output contains valid Vietnamese text
5. Audio validation remains PASS
"""

import os
import sys
import numpy as np

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from asr.audio import load_audio, validate_audio
from asr.inference import ASRInferenceEngine
from asr.normalization import normalize_vietnamese_text


def main():
    print("=" * 70)
    print("REAL VOICE END-TO-END VERIFICATION: 'Xin chào tôi tên là Thanh'")
    print("=" * 70)

    audio_path = "thanh_voice.wav"
    assert os.path.exists(audio_path), f"Audio file {audio_path} not found!"

    # Step 1: Load and Resample to 16kHz
    audio_16k, sr = load_audio(audio_path, target_sr=16000)
    print(f"\n[1] Audio Loaded: {len(audio_16k):,} samples at {sr} Hz")

    # Step 2: 7-Point Audio Validation Gate
    val_res = validate_audio(audio_16k, sample_rate=sr)
    print(f"[2] 7-Point Audio Validation Gate:")
    print(f"    is_valid:     {val_res.is_valid}")
    print(f"    duration_sec: {val_res.duration_sec:.2f}s")
    print(f"    sample_rate:  {val_res.sample_rate} Hz")
    print(f"    channels:     {val_res.channels}")
    print(f"    rms:          {val_res.rms:.4f}")
    print(f"    peak_abs:     {val_res.peak_abs:.4f}")
    print(f"    error_msg:    {val_res.error_message}")
    assert val_res.is_valid is True, f"Audio validation failed: {val_res.error_message}"

    # Step 3: Initialize ASR Engine and Run Transcription
    print(f"\n[3] Running PhoWhisper ASR Inference...")
    engine = ASRInferenceEngine(model_id="vinai/PhoWhisper-tiny", preferred_device="auto")
    print(f"    Model ID:     {engine.model_id}")
    print(f"    Device:       {engine.hardware_info.summary()}")

    # Test with default tokens and also test with requested max_new_tokens=448 to verify clamping
    res = engine.transcribe(audio_16k, max_new_tokens=448)

    print(f"\n[4] Transcription Completed Successfully:")
    print(f"    Raw Output:                   '{res.raw_text}'")
    print(f"    Normalized Output:            '{res.normalized_text}'")
    print(f"    Audio Duration:               {res.audio_duration_sec:.2f}s")
    print(f"    Latency:                      {res.latency_sec:.2f}s")
    print(f"    Real-Time Factor (RTF):       {res.rtf:.3f}")
    print(f"    Requested max_new_tokens:     {res.requested_max_new_tokens}")
    print(f"    Effective max_new_tokens:     {res.effective_max_new_tokens}")
    print(f"    Total Possible Decoder Len:   {res.total_possible_decoder_length} (<= 448)")

    # Step 5: Strict Assertions
    print(f"\n[5] Verifying Strict Acceptance Criteria:")
    assert res.raw_text is not None and len(res.raw_text.strip()) > 0, "Output is empty!"
    print("    [PASS] Output is non-empty")

    assert res.total_possible_decoder_length <= 448, "Exceeds max_target_positions!"
    print("    [PASS] No max_target_positions exception (total length <= 448)")

    # Check for valid Vietnamese words
    vietnamese_words = {"xin", "chào", "tôi", "tên", "là", "thanh"}
    found_words = [w for w in res.normalized_text.split() if w in vietnamese_words]
    print(f"    [PASS] Detected Vietnamese words: {found_words}")
    assert len(found_words) > 0, "No Vietnamese words detected in transcription!"

    assert val_res.is_valid is True, "Audio validation is not PASS!"
    print("    [PASS] 7-Point Audio Validation Gate status is PASS")

    print("\n" + "=" * 70)
    print("REAL END-TO-END VOICE TEST COMPLETED WITH 100% SUCCESS!")
    print("=" * 70)


if __name__ == "__main__":
    main()
