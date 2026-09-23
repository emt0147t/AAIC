"""Unit Tests for Session Exporter."""

import os
import shutil
import numpy as np
import pytest
from augmentation.pipeline import AugmentationStudioEngine
from export.exporter import AugmentationSessionExporter


def test_export_session_creates_structure_and_zip(tmp_path):
    exporter = AugmentationSessionExporter(base_export_dir=str(tmp_path))
    engine = AugmentationStudioEngine(sample_rate=16000)

    # 1 second test audio
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 400 * t)

    # Generate 2 augmentations
    augs = engine.generate_random_k(audio, k=2, base_seed=42)

    export_res = exporter.export_session(
        session_id="test_session_01",
        original_audio_16k=audio,
        original_transcript="thử nghiệm xuất dữ liệu",
        augmented_samples=augs,
        augmented_transcripts=["thử nghiệm một", "thử nghiệm hai"],
        reference_transcript="thử nghiệm xuất dữ liệu chuẩn",
    )

    session_dir = export_res["session_dir"]
    zip_path = export_res["zip_path"]

    assert os.path.exists(session_dir)
    assert os.path.exists(zip_path)
    assert os.path.exists(os.path.join(session_dir, "original", "original.wav"))
    assert os.path.exists(os.path.join(session_dir, "augmented", "aug_01.wav"))
    assert os.path.exists(os.path.join(session_dir, "manifest.csv"))
    assert os.path.exists(os.path.join(session_dir, "results.json"))
