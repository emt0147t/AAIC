"""Unit Tests for Dataset Mixture & Manifest Builder."""

import pytest
from datasets.mixture import DatasetMixtureBuilder


def test_speaker_disjoint_partition():
    builder = DatasetMixtureBuilder(val_ratio=0.25, seed=42, enforce_speaker_disjoint=True)

    # 4 distinct speakers, 3 utterances each
    raw_samples = []
    for spk_idx in range(4):
        spk_id = f"spk_{spk_idx:02d}"
        for u_idx in range(3):
            raw_samples.append({
                "id": f"{spk_id}_{u_idx}",
                "dataset": "vivos",
                "speaker_id": spk_id,
                "duration_sec": 3.5,
                "transcript": "xin chào việt nam",
            })

    train_rows, val_rows, audit = builder.partition_samples(raw_samples)

    train_spks = {r.speaker_id for r in train_rows}
    val_spks = {r.speaker_id for r in val_rows}

    # Strict assertion: NO OVERLAP between train speakers and val speakers
    assert len(train_spks.intersection(val_spks)) == 0
    assert len(train_rows) + len(val_rows) == 12


def test_zero_test_contamination_enforcement():
    builder = DatasetMixtureBuilder(val_ratio=0.2, seed=42)

    quarantined = {"test_sample_forbidden_01"}
    pool = [
        {"id": "train_sample_ok", "dataset": "vivos", "speaker_id": "spk_1", "duration_sec": 2.0, "transcript": "ok"},
        {"id": "test_sample_forbidden_01", "dataset": "vivos", "speaker_id": "spk_2", "duration_sec": 2.0, "transcript": "bad"},
    ]

    with pytest.raises(ValueError, match="CRITICAL DATA CONTAMINATION DETECTED"):
        builder.partition_samples(pool, quarantined_test_ids=quarantined)


def test_duration_boundary_filter():
    builder = DatasetMixtureBuilder(max_duration_sec=30.0)

    pool = [
        {"id": "s1", "dataset": "vimd", "speaker_id": "spk_1", "duration_sec": 5.0, "transcript": "a"},
        {"id": "s2_toolong", "dataset": "vimd", "speaker_id": "spk_2", "duration_sec": 32.5, "transcript": "b"},
        {"id": "s3_tooshort", "dataset": "vimd", "speaker_id": "spk_3", "duration_sec": 0.1, "transcript": "c"},
    ]

    train_rows, val_rows, audit = builder.partition_samples(pool)
    retained_ids = {r.sample_id for r in train_rows + val_rows}

    assert "s1" in retained_ids
    assert "s2_toolong" not in retained_ids
    assert "s3_tooshort" not in retained_ids
    assert audit["excluded_duration_count"] == 2


def test_manifest_cross_platform_line_ending_canonicalization():
    """Verify that LF, CRLF, and CR text content yield identical canonical hashes,

    while actual content modifications yield distinct hashes.
    """
    from scripts.train_full_e1 import compute_canonical_crlf_sha256

    lf_content = b"sample_id,audio_path,transcript\ns1,audio/1.wav,xin chao\ns2,audio/2.wav,viet nam\n"
    crlf_content = b"sample_id,audio_path,transcript\r\ns1,audio/1.wav,xin chao\r\ns2,audio/2.wav,viet nam\r\n"
    cr_content = b"sample_id,audio_path,transcript\rs1,audio/1.wav,xin chao\rs2,audio/2.wav,viet nam\r"

    hash_lf = compute_canonical_crlf_sha256(lf_content)
    hash_crlf = compute_canonical_crlf_sha256(crlf_content)
    hash_cr = compute_canonical_crlf_sha256(cr_content)

    assert hash_lf == hash_crlf
    assert hash_lf == hash_cr

    # Actual field/character change must alter hash
    altered_content = b"sample_id,audio_path,transcript\r\ns1,audio/1.wav,xin chao\r\ns2,audio/2.wav,viet nam 2\r\n"
    hash_altered = compute_canonical_crlf_sha256(altered_content)
    assert hash_altered != hash_crlf

