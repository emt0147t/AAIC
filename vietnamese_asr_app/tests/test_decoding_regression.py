"""Regression Tests for Whisper Decoding Bounds and Max Target Positions.

Tests:
a. prompt_len=4, requested=448 -> safely clamped to 443 (total 447 <= 448)
b. prompt_len=4, requested=128 -> remains 128 (total 132 <= 448)
c. a longer decoder prompt -> dynamically accounts for prompt length
d. model max_target_positions different from 448 (e.g. 256 or 100)
e. short Vietnamese microphone input end-to-end
"""

import numpy as np
import pytest
import torch

from asr.audio import load_audio, validate_audio
from asr.inference import ASRInferenceEngine, DecodingConfigurationError


def test_regression_case_a_prompt4_requested448():
    """Case A: prompt_len=4, requested=448. Must be clamped to 443 with safety margin=1."""
    prompt_len = 4
    requested = 448
    max_target_positions = 448
    safety_margin = 1

    effective = ASRInferenceEngine.compute_safe_max_new_tokens(
        requested_max_new_tokens=requested,
        prompt_len=prompt_len,
        max_target_positions=max_target_positions,
        safety_margin=safety_margin,
    )

    assert effective == 443
    total_len = prompt_len + effective
    assert total_len == 447
    assert total_len <= max_target_positions


def test_regression_case_b_prompt4_requested128():
    """Case B: prompt_len=4, requested=128. Safe value remains 128."""
    prompt_len = 4
    requested = 128
    max_target_positions = 448

    effective = ASRInferenceEngine.compute_safe_max_new_tokens(
        requested_max_new_tokens=requested,
        prompt_len=prompt_len,
        max_target_positions=max_target_positions,
        safety_margin=1,
    )

    assert effective == 128
    total_len = prompt_len + effective
    assert total_len == 132
    assert total_len <= max_target_positions


def test_regression_case_c_longer_decoder_prompt():
    """Case C: A longer decoder prompt (e.g. prompt_len=32, requested=448)."""
    prompt_len = 32
    requested = 448
    max_target_positions = 448

    effective = ASRInferenceEngine.compute_safe_max_new_tokens(
        requested_max_new_tokens=requested,
        prompt_len=prompt_len,
        max_target_positions=max_target_positions,
        safety_margin=1,
    )

    # 448 - 32 - 1 = 415
    assert effective == 415
    total_len = prompt_len + effective
    assert total_len == 447
    assert total_len <= max_target_positions


def test_regression_case_d_different_max_target_positions():
    """Case D: Model architecture with max_target_positions different from 448 (e.g. 256, 100)."""
    # Test with 256
    prompt_len = 4
    requested = 300
    effective_256 = ASRInferenceEngine.compute_safe_max_new_tokens(
        requested_max_new_tokens=requested,
        prompt_len=prompt_len,
        max_target_positions=256,
        safety_margin=1,
    )
    # 256 - 4 - 1 = 251
    assert effective_256 == 251
    assert prompt_len + effective_256 <= 256

    # Test with 100
    effective_100 = ASRInferenceEngine.compute_safe_max_new_tokens(
        requested_max_new_tokens=150,
        prompt_len=6,
        max_target_positions=100,
        safety_margin=1,
    )
    # 100 - 6 - 1 = 93
    assert effective_100 == 93
    assert 6 + effective_100 <= 100


def test_regression_case_e_short_vietnamese_microphone_input():
    """Case E: Short Vietnamese microphone input tuple (sr, data) loaded and validated."""
    sr = 16000
    duration_sec = 1.5
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False, dtype=np.float32)
    # Synthetic speech harmonic tone
    mic_audio = (
        0.3 * np.sin(2 * np.pi * 220 * t) +
        0.15 * np.sin(2 * np.pi * 440 * t)
    )

    # Gradio microphone format tuple: (sample_rate, numpy_array)
    mic_tuple = (sr, mic_audio)
    loaded_audio, loaded_sr = load_audio(mic_tuple, target_sr=16000)

    val_res = validate_audio(loaded_audio, sample_rate=loaded_sr)
    assert val_res.is_valid is True
    assert val_res.rms > 0.01
    assert abs(val_res.duration_sec - 1.5) < 1e-2


def test_engine_inference_with_clamped_tokens():
    """Verify that transcribe() runs successfully even when max_new_tokens=448 is requested."""
    engine = ASRInferenceEngine(model_id="vinai/PhoWhisper-tiny", preferred_device="cpu")

    # 1.0s clean test signal
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    audio = 0.3 * np.sin(2 * np.pi * 300 * t)

    # Request max_new_tokens=448 (which previously crashed with ValueError)
    result = engine.transcribe(audio, max_new_tokens=448)

    assert result.requested_max_new_tokens == 448
    assert result.effective_max_new_tokens == 443
    assert result.total_possible_decoder_length == 447
    assert result.total_possible_decoder_length <= 448
    assert result.latency_sec > 0.0
