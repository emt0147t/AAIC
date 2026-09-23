"""Unit Tests for Audio Validation, Normalization, and Inference Pipeline."""

import numpy as np
import pytest
from asr.audio import load_audio, validate_audio
from asr.normalization import normalize_vietnamese_text


def test_normalization_vietnamese():
    # Combining accents, uppercase, punctuation, extra spaces
    raw = "  Hà   Nội, Thủ Đô   CỦA   Việt Nam!!!   "
    norm = normalize_vietnamese_text(raw)
    assert norm == "hà nội thủ đô của việt nam"


def test_normalization_punctuation_stripping():
    raw = "Alo? 1, 2, 3... Alo! [tiếng động]"
    norm = normalize_vietnamese_text(raw)
    assert norm == "alo 1 2 3 alo tiếng động"


def test_audio_validation_clean():
    sr = 16000
    t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False, dtype=np.float32)
    sig = 0.4 * np.sin(2 * np.pi * 440 * t)

    res = validate_audio(sig, sample_rate=sr)
    assert res.is_valid is True
    assert abs(res.duration_sec - 1.5) < 1e-3
    assert res.channels == 1
    assert res.rms > 0.01


def test_audio_validation_silent():
    sr = 16000
    sig = np.zeros(sr * 2, dtype=np.float32)
    res = validate_audio(sig, sample_rate=sr)
    assert res.is_valid is False
    assert "silent" in res.error_message.lower()


def test_audio_validation_too_short():
    sr = 16000
    sig = 0.5 * np.ones(int(0.1 * sr), dtype=np.float32)
    res = validate_audio(sig, sample_rate=sr, min_duration_sec=0.2)
    assert res.is_valid is False
    assert "below minimum threshold" in res.error_message


def test_audio_validation_too_long():
    sr = 16000
    sig = 0.5 * np.ones(int(31.0 * sr), dtype=np.float32)
    # When allow_chunking=False, must fail upper-bound
    res = validate_audio(sig, sample_rate=sr, max_duration_sec=30.0, allow_chunking=False)
    assert res.is_valid is False
    assert "exceeds maximum threshold" in res.error_message

    # When allow_chunking=True, passes 7-point gate
    res_chunk = validate_audio(sig, sample_rate=sr, max_duration_sec=30.0, allow_chunking=True)
    assert res_chunk.is_valid is True


def test_audio_validation_nan():
    sr = 16000
    sig = np.ones(sr, dtype=np.float32)
    sig[50] = np.nan
    res = validate_audio(sig, sample_rate=sr)
    assert res.is_valid is False
    assert "nan or infinite" in res.error_message.lower()
