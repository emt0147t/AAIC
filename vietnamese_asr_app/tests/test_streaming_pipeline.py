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
