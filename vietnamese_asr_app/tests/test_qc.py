"""Unit Tests for 10-Check Audio Quality Control (QC) Pipeline."""

import numpy as np
import pytest
from augmentation.qc import run_10_point_qc


def test_qc_passes_on_clean_audio():
    # 2 seconds of 440 Hz sine wave at 16kHz
    sr = 16000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False, dtype=np.float32)
    clean_audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    res = run_10_point_qc(clean_audio, sample_rate=sr)
    assert res.passed is True
    assert res.status == "PASSED"
    assert len(res.failed_checks) == 0
    assert len(res.sha256_hash) == 64
    assert res.details["nan_count"] == 0
    assert res.details["inf_count"] == 0
    assert res.details["clipping_ratio"] == 0.0


def test_qc_fails_on_silent_audio():
    sr = 16000
    silent_audio = np.zeros(sr * 2, dtype=np.float32)

    res = run_10_point_qc(silent_audio, sample_rate=sr)
    assert res.passed is False
    assert res.status == "FAILED"
    assert "7_silent_signal_rms" in res.failed_checks


def test_qc_fails_on_nan_and_inf():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    bad_audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    bad_audio[100] = np.nan
    bad_audio[200] = np.inf

    res = run_10_point_qc(bad_audio, sample_rate=sr)
    assert res.passed is False
    assert "2_finite_values" in res.failed_checks
    assert "3_no_nans" in res.failed_checks
    assert "4_no_infs" in res.failed_checks


def test_qc_fails_on_excessive_clipping():
    sr = 16000
    # Create square wave pinned at +/- 1.0
    clipped = np.sign(np.sin(np.linspace(0, 100, sr * 2, dtype=np.float32)))

    res = run_10_point_qc(clipped, sample_rate=sr, max_clipping_ratio=0.001)
    assert res.passed is False
    assert "6_excessive_clipping" in res.failed_checks


def test_qc_fails_on_duration_out_of_bounds():
    sr = 16000
    # Too short (< 0.2s)
    short_audio = np.ones(int(0.1 * sr), dtype=np.float32) * 0.1
    res_short = run_10_point_qc(short_audio, sample_rate=sr, min_duration_sec=0.2)
    assert res_short.passed is False
    assert "8_duration_bounds" in res_short.failed_checks

    # Too long (> 30.0s)
    long_audio = np.ones(int(31.0 * sr), dtype=np.float32) * 0.1
    res_long = run_10_point_qc(long_audio, sample_rate=sr, max_duration_sec=30.0)
    assert res_long.passed is False
    assert "8_duration_bounds" in res_long.failed_checks
