"""Regression tests for Full E1 streaming data pipeline and disk safety.

Verifies:
1. Disk Safety Gate aborts when free disk is below threshold.
2. Datasets are configured as IterableDataset (streaming=True).
3. Audio decoding happens on-the-fly without parquet file materialization.
4. Deterministic exposure policy matches the audited 17 repeated indices per epoch.
"""

import io
import shutil
import sys
import pytest
import numpy as np
import soundfile as sf

# Ensure Hugging Face datasets is imported from site-packages, bypassing local package name collision
if "datasets" in sys.modules and "site-packages" not in getattr(sys.modules["datasets"], "__file__", ""):
    sys.modules.pop("datasets", None)
orig_sys_path = list(sys.path)
sys.path = [p for p in sys.path if not (p == "" or p == "." or "vietnamese_asr_app" in p.lower())]
import datasets as hf_datasets
from datasets import Audio, IterableDataset, load_dataset
sys.path = orig_sys_path

from scripts.train_full_e1 import (
    check_disk_safety,
    decode_audio_record,
    verify_exposure_policy,
    resolve_frozen_test_samples,
    compute_canonical_crlf_sha256,
    find_test_manifest_path,
    FROZEN_TEST_SHA256,
    AUDITED_REPEATED_INDICES,
    PINNED_VIVOS_REVISION,
    PINNED_VIMD_REVISION,
    TOTAL_TRAINING_READY,
)


def test_disk_safety_gate_abort():
    """Verify that disk safety gate raises RuntimeError when free space is below threshold."""
    with pytest.raises(RuntimeError, match="CRITICAL DISK SAFETY ABORT"):
        check_disk_safety(min_free_gb=999999.0)


def test_disk_safety_gate_pass():
    """Verify that disk safety gate passes when free space is sufficient."""
    free_gb = check_disk_safety(min_free_gb=0.01)
    assert free_gb > 0.01


def test_decode_audio_record_from_bytes():
    """Verify on-demand audio decoding from in-memory WAV bytes without disk writes."""
    samplerate = 16000
    duration_sec = 0.5
    t = np.linspace(0, duration_sec, int(samplerate * duration_sec), endpoint=False)
    sig = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, sig, samplerate, format="WAV")
    raw_bytes = buf.getvalue()

    audio_dict = {"bytes": raw_bytes, "path": None}
    decoded = decode_audio_record(audio_dict, target_sr=16000)

    assert isinstance(decoded, np.ndarray)
    assert decoded.dtype == np.float32
    assert len(decoded) == len(sig)
    assert np.allclose(decoded, sig, atol=1e-3)


def test_deterministic_exposure_indices_preserved():
    """Verify that the audited 17 repeated indices are strictly preserved for all 3 epochs."""
    for ep in [1, 2, 3]:
        padding_indices = verify_exposure_policy(ep, n_samples=TOTAL_TRAINING_READY, base_seed=42)
        assert len(padding_indices) == 17
        assert padding_indices == AUDITED_REPEATED_INDICES[ep]


def test_streaming_loader_returns_iterable_dataset():
    """Verify that the cloud dataset loader returns IterableDataset without downloading parquet shards."""
    total, used, free_before = shutil.disk_usage(".")

    ds_vivos = load_dataset(
        "thanhduycao/vivos_ng_only",
        split="train",
        streaming=True,
        revision=PINNED_VIVOS_REVISION,
    ).cast_column("audio", Audio(decode=False))

    ds_vimd = load_dataset(
        "nguyendv02/ViMD_Dataset",
        split="train",
        streaming=True,
        revision=PINNED_VIMD_REVISION,
    ).cast_column("audio", Audio(decode=False))

    assert isinstance(ds_vivos, IterableDataset)
    assert isinstance(ds_vimd, IterableDataset)

    total, used, free_after = shutil.disk_usage(".")
    delta_mb = (free_before - free_after) / (1024 ** 2)
    # Assert zero/negligible disk consumption (no 500 MB parquet shards downloaded)
    assert delta_mb < 20.0


def test_resolve_frozen_test_samples_portability_windows_and_colab():
    """Verify that resolve_frozen_test_samples resolves all 9 samples in streaming mode,

    ensuring complete cross-platform portability on Linux/Colab without Windows path dependence.
    """
    manifest_path = find_test_manifest_path("manifests/test_manifest.csv")
    raw_bytes = open(manifest_path, "rb").read()
    assert compute_canonical_crlf_sha256(raw_bytes) == FROZEN_TEST_SHA256

    samples = resolve_frozen_test_samples(
        test_manifest_path=manifest_path,
        vimd_revision=PINNED_VIMD_REVISION,
        force_streaming=True,
    )

    # 1. Exactly 9 matched samples
    assert len(samples) == 9

    # 2. No missing or duplicate sample IDs
    sample_ids = [s["sample_id"] for s in samples]
    assert len(set(sample_ids)) == 9
    expected_ids = [f"vimd_test_vimd_{i:02d}" for i in range(1, 10)]
    assert sample_ids == expected_ids

    # 3. Audio waveform checks
    for s in samples:
        assert "audio" in s, f"Sample {s['sample_id']} missing 'audio' key!"
        wf = s["audio"]
        assert isinstance(wf, np.ndarray), f"Audio for {s['sample_id']} must be np.ndarray"
        assert wf.ndim == 1, f"Audio for {s['sample_id']} must be mono 1D array"
        assert wf.dtype == np.float32, f"Audio for {s['sample_id']} must be float32"
        assert len(wf) > 0, f"Audio for {s['sample_id']} is empty"
        assert "transcript" in s and len(s["transcript"]) > 0
        assert "speaker_id" in s and s["speaker_id"].startswith("spk_")
        assert 0.5 <= s["duration_sec"] <= 30.0


def test_evaluate_model_on_manifest_receives_audio_waveform():
    """Verify that evaluate_model_on_manifest receives samples with 'audio' key without skipping."""
    from research.evaluator import evaluate_model_on_manifest
    from unittest.mock import MagicMock
    import torch

    # Create mock model and processor
    mock_model = MagicMock()
    mock_model.name_or_path = "mock-whisper"
    mock_model.eval.return_value = None
    mock_model.to.return_value = mock_model
    mock_model.generate.return_value = torch.tensor([[50258, 50259, 50257]])

    mock_processor = MagicMock()
    feat_mock = MagicMock()
    feat_mock.input_features = torch.zeros(1, 80, 3000)
    mock_processor.return_value = feat_mock
    mock_processor.batch_decode.return_value = ["đầu năm cũng đã cùng"]

    waveform = np.zeros(16000, dtype=np.float32)
    test_samples = [
        {
            "sample_id": "vimd_test_vimd_01",
            "audio": waveform,
            "transcript": "đầu năm cũng đã cùng",
        },
        {
            "sample_id": "vimd_test_vimd_02",
            "audio": waveform,
            "transcript": "hỗ trợ thứ nhất",
        },
    ]

    report = evaluate_model_on_manifest(
        model=mock_model,
        processor=mock_processor,
        samples=test_samples,
        device="cpu",
    )

    # Proves exactly 2 samples were processed and NOT skipped
    assert report.total_samples == 2
    assert len(report.per_utterance_results) == 2
    assert report.per_utterance_results[0]["sample_id"] == "vimd_test_vimd_01"
    assert report.per_utterance_results[1]["sample_id"] == "vimd_test_vimd_02"


def test_frozen_test_resolution_fails_on_tampered_manifest(tmp_path):
    """Verify that resolve_frozen_test_samples fails loudly if the manifest content is altered."""
    tampered_manifest = tmp_path / "test_manifest.csv"
    tampered_manifest.write_text("sample_id,audio_path,transcript\nfake_01,path/fake.wav,fake\n", encoding="utf-8")

    with pytest.raises(ValueError, match="CRITICAL: Test manifest content altered"):
        resolve_frozen_test_samples(test_manifest_path=str(tampered_manifest))

