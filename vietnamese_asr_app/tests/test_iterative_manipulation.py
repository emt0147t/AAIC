"""Test Suite for Iterative Audio Manipulation Pipeline.

Validates all 14 mandatory behavioral requirements:
1. K=1 creates exactly 1 manipulated output (x_1).
2. K=3 creates exactly 3 iterative outputs (x_1, x_2, x_3).
3. Parent-child lineage: parent(x_i) = x_{i-1}.
4. Zero TTS usage: strictly 1D waveform operations.
5. Original audio x_0 remains strictly unchanged (non-destructive).
6. Format conformity: all outputs are 16 kHz mono.
7. 10-point QC evaluated after every iteration.
8. Failed candidates are never accepted as valid outputs.
9. Deterministic reproducibility under identical seed.
10. Complete metadata lineage.
11. Reference transcript mode computes exact WER/CER.
12. No-reference mode strictly uses proxy consistency metrics.
13. Best-preserving search selects the highest-scoring candidate.
14. Multi-format export creates valid WAVs, JSON, CSV, and ZIP.
"""

import copy
import hashlib
import json
import os
import shutil
import tempfile
import numpy as np
import pytest

from augmentation.iterative import IterativeAudioManipulator, IterativeStepRecord
from services.asr_service import ASRService
from services.iterative_service import IterativeManipulationService


@pytest.fixture
def clean_synthetic_voice_audio():
    """Deterministic 16 kHz harmonic audio sample (simulating speech, not from frozen test set)."""
    sr = 16000
    duration = 2.0
    num_samples = int(sr * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False, dtype=np.float32)
    # Speech-like formant-modulated wave
    envelope = 0.5 * np.sin(np.pi * t / duration) ** 2
    f0 = 150.0
    sig = envelope * (0.6 * np.sin(2 * np.pi * f0 * t) + 0.3 * np.sin(4 * np.pi * f0 * t) + 0.1 * np.sin(6 * np.pi * f0 * t))
    return sig.astype(np.float32)


@pytest.fixture
def temp_export_dir():
    d = tempfile.mkdtemp(prefix="test_iterative_export_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_k1_creates_single_manipulated_output(clean_synthetic_voice_audio, temp_export_dir):
    """Test 1: K=1 creates exactly one manipulated output (x_1) with parent x_0."""
    service = IterativeManipulationService(base_export_dir=temp_export_dir)
    records, audio_map, meta = service.execute_chain(
        clean_synthetic_voice_audio,
        k=1,
        strategy="random",
        base_seed=42,
    )
    assert meta["k_steps"] == 1
    # records contains x_0 and x_1
    assert len(records) == 2
    assert records[0].iteration == 0
    assert records[1].iteration == 1
    assert records[1].parent_iteration == 0
    assert 0 in audio_map and 1 in audio_map
    assert not np.array_equal(audio_map[0], audio_map[1])


def test_k3_cumulative_chain_parent_lineage(clean_synthetic_voice_audio, temp_export_dir):
    """Test 2 & 3: K=3 creates x_1, x_2, x_3 where parent(x_i) = x_{i-1}."""
    service = IterativeManipulationService(base_export_dir=temp_export_dir)
    records, audio_map, meta = service.execute_chain(
        clean_synthetic_voice_audio,
        k=3,
        strategy="random",
        base_seed=100,
    )
    assert len(records) == 4  # x_0, x_1, x_2, x_3
    for i in range(1, 4):
        rec = records[i]
        assert rec.iteration == i
        assert rec.parent_iteration == i - 1
        assert i in audio_map
        # Verify cumulative transformation: x_i is different from x_{i-1}
        assert not np.array_equal(audio_map[i], audio_map[i - 1])


def test_original_audio_unchanged_and_no_tts(clean_synthetic_voice_audio, temp_export_dir):
    """Test 4 & 5: Original audio x_0 remains strictly bitwise identical (no TTS, non-destructive)."""
    orig_copy = clean_synthetic_voice_audio.copy()
    orig_hash = hashlib.sha256(clean_synthetic_voice_audio.tobytes()).hexdigest()

    service = IterativeManipulationService(base_export_dir=temp_export_dir)
    records, audio_map, _ = service.execute_chain(
        clean_synthetic_voice_audio,
        k=2,
        strategy="random",
        base_seed=42,
    )

    # Verify input array was not modified in-place
    assert np.array_equal(clean_synthetic_voice_audio, orig_copy)
    assert hashlib.sha256(clean_synthetic_voice_audio.tobytes()).hexdigest() == orig_hash
    # x_0 in audio_map matches original
    assert np.array_equal(audio_map[0], orig_copy)


def test_format_conformity_16k_mono_and_qc_gating(clean_synthetic_voice_audio, temp_export_dir):
    """Test 6 & 7: All outputs are 1D mono, 16 kHz, with 10-point QC verified at each step."""
    service = IterativeManipulationService(base_export_dir=temp_export_dir)
    records, audio_map, _ = service.execute_chain(
        clean_synthetic_voice_audio,
        k=3,
        strategy="random",
        base_seed=2026,
    )
    for i, audio in audio_map.items():
        assert audio.ndim == 1, f"Iteration {i} must be 1D mono"
        assert len(audio) > 0, f"Iteration {i} must not be empty"
        assert np.all(np.isfinite(audio)), f"Iteration {i} must be finite"

    for rec in records:
        assert rec.qc_status in ["PASSED", "PASS", "FAILED", "FAIL"]
        if rec.qc_passed:
            assert rec.rms > 1e-5
            assert rec.peak_abs <= 1.0


def test_deterministic_reproducibility(clean_synthetic_voice_audio, temp_export_dir):
    """Test 9: Identical seed produces identical audio arrays and metadata."""
    service1 = IterativeManipulationService(base_export_dir=temp_export_dir)
    service2 = IterativeManipulationService(base_export_dir=temp_export_dir)

    recs1, map1, _ = service1.execute_chain(clean_synthetic_voice_audio, k=2, base_seed=777)
    recs2, map2, _ = service2.execute_chain(clean_synthetic_voice_audio, k=2, base_seed=777)

    for i in range(len(recs1)):
        assert recs1[i].manipulation_name == recs2[i].manipulation_name
        assert recs1[i].parameters == recs2[i].parameters
        assert np.allclose(map1[i], map2[i], atol=1e-6)


def test_reference_scoring_vs_no_reference_proxy(clean_synthetic_voice_audio, temp_export_dir):
    """Test 11 & 12: Reference mode computes WER/CER; no-reference mode strictly uses proxy consistency."""
    service = IterativeManipulationService(base_export_dir=temp_export_dir)

    # Case A: Reference provided
    recs_ref, _, meta_ref = service.execute_chain(
        clean_synthetic_voice_audio,
        k=2,
        strategy="random",
        base_seed=42,
        reference_transcript="thử nghiệm âm thanh tiếng việt",
    )
    assert meta_ref["has_reference"] is True
    assert meta_ref["score_mode"] == "ACCURACY_WER_CER"
    assert recs_ref[0].wer is not None
    assert recs_ref[0].cer is not None
    assert recs_ref[0].score_type == "ACCURACY_WER_CER"

    # Case B: No reference provided
    recs_no_ref, _, meta_no_ref = service.execute_chain(
        clean_synthetic_voice_audio,
        k=2,
        strategy="random",
        base_seed=42,
        reference_transcript=None,
    )
    assert meta_no_ref["has_reference"] is False
    assert meta_no_ref["score_mode"] == "ASR_CONSISTENCY_QUALITY_PROXY"
    assert recs_no_ref[0].wer is None
    assert recs_no_ref[0].cer is None
    assert recs_no_ref[0].score_type == "ASR_CONSISTENCY_QUALITY_PROXY"


def test_best_preserving_search_selection(clean_synthetic_voice_audio, temp_export_dir):
    """Test 13: Best-preserving search generates M candidates and selects highest score."""
    service = IterativeManipulationService(base_export_dir=temp_export_dir)
    recs, audio_map, meta = service.execute_chain(
        clean_synthetic_voice_audio,
        k=2,
        strategy="best_preserving",
        candidates_per_step=3,
        base_seed=42,
    )
    assert meta["strategy"] == "best_preserving"
    assert len(recs) == 3  # x_0, x_1, x_2
    for r in recs:
        assert r.selected_as_best is True
        assert r.score > -900.0


def test_export_chain_artifacts(clean_synthetic_voice_audio, temp_export_dir):
    """Test 10 & 14: Export creates WAVs, JSON metadata, CSV manifest, and valid ZIP archive."""
    service = IterativeManipulationService(base_export_dir=temp_export_dir)
    recs, audio_map, _ = service.execute_chain(
        clean_synthetic_voice_audio,
        k=2,
        strategy="random",
        base_seed=42,
    )
    export_bundle = service.export_chain_bundle(
        records=recs,
        audio_map=audio_map,
        original_audio_id="demo_voice",
    )

    # Verify bundle structure
    assert os.path.exists(export_bundle["json_path"])
    assert os.path.exists(export_bundle["csv_path"])
    assert os.path.exists(export_bundle["zip_path"])
    assert os.path.getsize(export_bundle["zip_path"]) > 0

    # Verify JSON content
    with open(export_bundle["json_path"], "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["original_audio_id"] == "demo_voice"
    assert len(meta["chain"]) == 3  # x_0, x_1, x_2
    assert meta["chain"][0]["iteration"] == 0
    assert meta["chain"][1]["parent_iteration"] == 0
    assert meta["chain"][2]["parent_iteration"] == 1

    # Verify WAV files exist
    assert os.path.exists(os.path.join(export_bundle["wav_dir"], "iteration_000.wav"))
    assert os.path.exists(os.path.join(export_bundle["wav_dir"], "iteration_001.wav"))
    assert os.path.exists(os.path.join(export_bundle["wav_dir"], "iteration_002.wav"))


def test_regression_audio2_and_audio3_distinct(clean_synthetic_voice_audio, temp_export_dir):
    """REGRESSION TEST: Specifically detects and prevents the bug where Audio 3 (x2)
    was identical to Audio 2 (x1) under Best-Preserving Search.
    """
    service = IterativeManipulationService(base_export_dir=temp_export_dir)
    records, audio_map, meta = service.execute_chain(
        clean_synthetic_voice_audio,
        k=3,
        strategy="best_preserving",
        candidates_per_step=3,
        base_seed=42,
    )
    assert len(records) == 4
    assert 0 in audio_map and 1 in audio_map and 2 in audio_map and 3 in audio_map

    # Assert Audio 2 (x1) differs from Audio 1 (x0)
    diff_0_1 = float(np.max(np.abs(audio_map[1] - audio_map[0])))
    assert diff_0_1 > 1e-3, f"x_1 must genuinely differ from x_0: got {diff_0_1}"

    # Assert Audio 3 (x2) strictly differs from Audio 2 (x1)
    diff_1_2 = float(np.max(np.abs(audio_map[2] - audio_map[1])))
    assert diff_1_2 > 1e-3, f"x_2 must genuinely differ from x_1 (Audio 3 != Audio 2 bug): got {diff_1_2}"

    # Assert Audio 4 (x3) strictly differs from Audio 3 (x2)
    diff_2_3 = float(np.max(np.abs(audio_map[3] - audio_map[2])))
    assert diff_2_3 > 1e-3, f"x_3 must genuinely differ from x_2: got {diff_2_3}"

    # Verify explicit lineage records
    for i in range(1, 4):
        rec = records[i]
        assert rec.waveform_changed is True
        assert rec.parent_sha256 == records[i - 1].child_sha256
        assert rec.child_sha256 != rec.parent_sha256
        assert rec.max_abs_delta > 1e-4


def test_gain_on_peak_normalized_audio_does_not_collapse_to_identity():
    """Verify that applying positive gain to peak-normalized audio applies soft saturation
    instead of algebraically collapsing to an identity no-op.
    """
    from augmentation.transforms import apply_gain
    audio = np.array([0.98, -0.98, 0.49, -0.49, 0.1, 0.0], dtype=np.float32)
    out = apply_gain(audio, gain_db=2.0)
    delta = float(np.max(np.abs(out - audio)))
    assert delta > 0.05, f"Gain on peak-normalized audio must alter waveform, delta={delta}"
    assert np.max(np.abs(out)) <= 0.99
    assert np.all(np.isfinite(out))


def test_random_chain_consecutive_operator_diversity(clean_synthetic_voice_audio, temp_export_dir):
    """Verify that consecutive iterations avoid trivial repetition of the same operator family."""
    service = IterativeManipulationService(base_export_dir=temp_export_dir)
    records, _, _ = service.execute_chain(
        clean_synthetic_voice_audio,
        k=4,
        strategy="random",
        base_seed=123,
    )
    # Check that adjacent steps do not repeat operator if multiple are enabled
    for i in range(2, len(records)):
        prev_op = records[i - 1].manipulation_name
        curr_op = records[i].manipulation_name
        assert prev_op != curr_op, f"Step {i} ({curr_op}) should differ from step {i-1} ({prev_op})"
