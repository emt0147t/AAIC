"""End-to-End System Verification Script.

Simulates complete user flow:
1. Audio loading & 7-point validation
2. Waveform K-augmentation (K=3)
3. 10-check QC verification on all generated waveforms
4. Levenshtein WER/CER and consistency metrics calculation
5. Full session export to directory and ZIP bundle
6. Speaker-disjoint multi-dataset partition simulation
"""

import os
import sys
import numpy as np

# Ensure app root in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from asr.audio import save_wav_pcm16, validate_audio
from asr.normalization import normalize_vietnamese_text
from augmentation.pipeline import AugmentationStudioEngine
from augmentation.qc import run_10_point_qc
from datasets.mixture import DatasetMixtureBuilder
from evaluation.metrics import compute_levenshtein_cer, compute_levenshtein_wer, evaluate_augmentation_consistency
from export.exporter import AugmentationSessionExporter


def main():
    print("=" * 70)
    print("VIETNAMESE ASR APPLICATION — END-TO-END VERIFICATION")
    print("=" * 70)

    # 1. Generate clean 2.0s sample audio (simulating user voice input)
    sr = 16000
    duration_s = 2.0
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
    # Fundamental frequency ~200Hz with harmonics
    sample_audio = (
        0.4 * np.sin(2 * np.pi * 200 * t) +
        0.2 * np.sin(2 * np.pi * 400 * t) +
        0.1 * np.sin(2 * np.pi * 600 * t)
    )

    print("\n[Step 1] 7-Point Audio Validation Gate:")
    val_res = validate_audio(sample_audio, sample_rate=sr)
    print(f"  Passed: {val_res.is_valid}")
    print(f"  Duration: {val_res.duration_sec:.2f}s | RMS: {val_res.rms:.4f} | Peak: {val_res.peak_abs:.2f}")
    assert val_res.is_valid, "Sample audio failed validation!"

    # 2. Augmentation Studio: K=3 Waveform Augmentations
    print("\n[Step 2] Waveform K-Augmentation (K=3):")
    engine = AugmentationStudioEngine(sample_rate=sr)
    augs = engine.generate_random_k(
        sample_audio,
        k=3,
        base_seed=42,
        enabled_transforms=["gain", "additive_noise", "time_shift", "synthetic_reverb"],
    )
    print(f"  Generated {len(augs)} authentic waveform variants.")

    # 3. 10-Check QC Pipeline on All Variants
    print("\n[Step 3] 10-Check Quality Control (QC) Pipeline:")
    for idx, aug in enumerate(augs):
        qc = aug.qc_result
        print(f"  Variant {aug.aug_id} (Seed: {aug.seed}):")
        print(f"    QC Status: {qc.status}")
        print(f"    SHA-256: {qc.sha256_hash[:16]}...")
        print(f"    Duration: {aug.duration_sec:.2f}s | RMS: {qc.details['rms']:.4f} | Clipping: {qc.details['clipping_ratio']*100:.3f}%")
        print(f"    Transforms: {aug.transform_chain}")
        assert qc.passed, f"Augmentation {aug.aug_id} failed QC check!"

    # 4. Evaluation & Consistency Metrics
    print("\n[Step 4] Levenshtein WER, CER & Consistency Evaluation:")
    orig_tx = "cộng hòa xã hội chủ nghĩa việt nam độc lập tự do hạnh phúc"
    ref_tx = "cộng hòa xã hội chủ nghĩa việt nam độc lập tự do hạnh phúc"
    aug_txs = [
        "cộng hòa xã hội chủ nghĩa việt nam độc lập tự do hạnh phúc",
        "cộng hòa xã hội việt nam độc lập tự do hạnh phúc",  # 1 deletion
        "cộng hòa xã hội chủ nghĩa việt nam độc lập tự do hạnh phúc tươi đẹp",  # 2 insertions
    ]

    for i, hyp in enumerate(aug_txs):
        eval_res = evaluate_augmentation_consistency(original_hyp=orig_tx, augmented_hyp=hyp, reference_text=ref_tx)
        print(f"  Variant aug_{i+1:02d}:")
        print(f"    WER vs Reference: {(eval_res.wer_vs_ref or 0.0) * 100:.1f}%")
        print(f"    CER vs Reference: {(eval_res.cer_vs_ref or 0.0) * 100:.1f}%")
        print(f"    Drift WER vs Orig: {eval_res.consistency_drift_wer * 100:.1f}%")

    # 5. Session Export & Packaging
    print("\n[Step 5] Session Artifact Export & Standalone ZIP Creation:")
    exporter = AugmentationSessionExporter(base_export_dir="exports")
    session_id = "e2e_verification_session"
    exp_res = exporter.export_session(
        session_id=session_id,
        original_audio_16k=sample_audio,
        original_transcript=orig_tx,
        augmented_samples=augs,
        augmented_transcripts=aug_txs,
        reference_transcript=ref_tx,
        model_id="vinai/PhoWhisper-small",
    )

    print(f"  Session Directory: {exp_res['session_dir']}")
    print(f"  ZIP Archive:       {exp_res['zip_path']}")
    print(f"  Manifest CSV:      {exp_res['manifest_csv']}")
    print(f"  Results JSON:      {exp_res['results_json']}")

    assert os.path.exists(exp_res["zip_path"]), "Exported ZIP file does not exist!"
    print(f"  ZIP File Size:     {os.path.getsize(exp_res['zip_path']):,} bytes")

    # 6. Speaker-Disjoint Multi-Corpus Split Simulation
    print("\n[Step 6] Multi-Dataset Mixture Speaker-Disjoint Verification:")
    builder = DatasetMixtureBuilder(val_ratio=0.2, seed=42, enforce_speaker_disjoint=True)
    raw_pool = []
    for spk_i in range(10):
        for u in range(3):
            raw_pool.append({
                "id": f"spk_{spk_i}_u{u}",
                "dataset": "vivos",
                "speaker_id": f"spk_{spk_i:02d}",
                "duration_sec": 3.0,
                "transcript": "chào buổi sáng",
            })

    train_rows, val_rows, audit = builder.partition_samples(raw_pool)
    train_spks = {r.speaker_id for r in train_rows}
    val_spks = {r.speaker_id for r in val_rows}
    assert len(train_spks.intersection(val_spks)) == 0, "Speaker overlap detected!"
    print(f"  Total: {audit['total_input_samples']} | Train: {len(train_rows)} | Val: {len(val_rows)}")
    print(f"  Train Speakers: {len(train_spks)} | Val Speakers: {len(val_spks)} | Overlap: 0 (Strictly Disjoint)")

    print("\n" + "=" * 70)
    print("ALL END-TO-END VERIFICATION CHECKS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
