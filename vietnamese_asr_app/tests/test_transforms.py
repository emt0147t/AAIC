"""Unit Tests for Waveform Audio Transforms."""

import numpy as np
import pytest
from augmentation.transforms import (
    apply_additive_noise,
    apply_frequency_filter,
    apply_gain,
    apply_pitch_shift,
    apply_synthetic_reverb,
    apply_time_shift,
    apply_time_stretch,
)


@pytest.fixture
def clean_signal():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * 300 * t), sr


def test_apply_gain(clean_signal):
    sig, _ = clean_signal
    boosted = apply_gain(sig, gain_db=3.0)
    attenuated = apply_gain(sig, gain_db=-3.0)

    assert isinstance(boosted, np.ndarray)
    assert np.max(np.abs(boosted)) > np.max(np.abs(sig))
    assert np.max(np.abs(attenuated)) < np.max(np.abs(sig))
    assert np.all(np.isfinite(boosted))
    assert np.max(np.abs(boosted)) <= 1.0


def test_apply_additive_noise(clean_signal):
    sig, _ = clean_signal
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)

    noisy1 = apply_additive_noise(sig, snr_db=20.0, rng=rng1)
    noisy2 = apply_additive_noise(sig, snr_db=20.0, rng=rng2)

    # Determinism with same seed
    assert np.allclose(noisy1, noisy2)
    assert not np.allclose(noisy1, sig)
    assert np.all(np.isfinite(noisy1))


def test_apply_time_shift(clean_signal):
    sig, sr = clean_signal
    shifted = apply_time_shift(sig, shift_sec=0.1, sample_rate=sr, mode="zero")
    assert len(shifted) == len(sig)
    # The first 0.1s should be zeros in zero mode
    n_zeros = int(round(0.1 * sr))
    assert np.allclose(shifted[:n_zeros], 0.0)


def test_apply_frequency_filter(clean_signal):
    sig, sr = clean_signal
    filtered = apply_frequency_filter(sig, sample_rate=sr, filter_type="bandpass", low_cut=200, high_cut=3000)
    assert len(filtered) == len(sig)
    assert np.all(np.isfinite(filtered))


def test_apply_synthetic_reverb(clean_signal):
    sig, sr = clean_signal
    reverbed = apply_synthetic_reverb(sig, sample_rate=sr, decay_time_sec=0.2, wet_mix=0.3)
    assert len(reverbed) == len(sig)
    assert np.all(np.isfinite(reverbed))
