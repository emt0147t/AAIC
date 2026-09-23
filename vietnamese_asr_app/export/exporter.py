"""Artifact Exporter and Packaging Engine.

Saves original and augmented audio in standard 16-bit PCM WAV,
generates individual metadata JSONs, manifest CSVs, and packages
everything into a single downloadable ZIP bundle.
"""

from dataclasses import asdict
import json
import os
import shutil
import zipfile
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

try:
    from asr.audio import save_wav_pcm16
    from augmentation.pipeline import AugmentedAudioSample
except (ImportError, ValueError):
    from ..asr.audio import save_wav_pcm16
    from ..augmentation.pipeline import AugmentedAudioSample


class AugmentationSessionExporter:
    """Exports session artifacts into structured folders and creates ZIP bundles."""

    def __init__(self, base_export_dir: str = "exports"):
        self.base_export_dir = os.path.abspath(base_export_dir)
        os.makedirs(self.base_export_dir, exist_ok=True)

    def export_session(
        self,
        session_id: str,
        original_audio_16k: np.ndarray,
        original_transcript: str,
        augmented_samples: List[AugmentedAudioSample],
        augmented_transcripts: Optional[List[str]] = None,
        reference_transcript: Optional[str] = None,
        model_id: str = "vinai/PhoWhisper-small",
    ) -> Dict[str, str]:
        """Export session to directory and create ZIP archive.

        Directory layout:
          exports/{session_id}/
            ├── original/
            │   └── original.wav
            ├── augmented/
            │   ├── aug_01.wav
            │   ├── ...
            ├── transcripts/
            │   ├── original.txt
            │   ├── reference.txt (if provided)
            │   ├── aug_01.txt
            │   └── ...
            ├── metadata/
            │   ├── original.json
            │   ├── aug_01.json
            │   └── ...
            ├── manifest.csv
            ├── results.csv
            └── results.json
        """
        session_dir = os.path.join(self.base_export_dir, session_id)
        orig_dir = os.path.join(session_dir, "original")
        aug_dir = os.path.join(session_dir, "augmented")
        tx_dir = os.path.join(session_dir, "transcripts")
        meta_dir = os.path.join(session_dir, "metadata")

        for d in [orig_dir, aug_dir, tx_dir, meta_dir]:
            os.makedirs(d, exist_ok=True)

        # 1. Save original audio & transcript
        orig_wav_path = os.path.join(orig_dir, "original.wav")
        save_wav_pcm16(original_audio_16k, orig_wav_path, sample_rate=16000)

        with open(os.path.join(tx_dir, "original.txt"), "w", encoding="utf-8") as f:
            f.write(original_transcript.strip() + "\n")

        if reference_transcript:
            with open(os.path.join(tx_dir, "reference.txt"), "w", encoding="utf-8") as f:
                f.write(reference_transcript.strip() + "\n")

        # 2. Save augmented samples
        manifest_rows = []
        results_list = []

        # Add original row to manifest
        orig_meta = {
            "sample_id": "original",
            "type": "original",
            "duration_sec": round(float(len(original_audio_16k)) / 16000.0, 3),
            "transcript": original_transcript,
            "reference": reference_transcript or "",
            "qc_status": "PASSED",
            "model_id": model_id,
        }
        manifest_rows.append(orig_meta)

        with open(os.path.join(meta_dir, "original.json"), "w", encoding="utf-8") as f:
            json.dump(orig_meta, f, indent=2, ensure_ascii=False)

        for i, aug in enumerate(augmented_samples):
            aug_fname = f"{aug.aug_id}.wav"
            aug_wav_path = os.path.join(aug_dir, aug_fname)
            save_wav_pcm16(aug.audio_16k, aug_wav_path, sample_rate=aug.sample_rate)

            aug_hyp = augmented_transcripts[i] if (augmented_transcripts and i < len(augmented_transcripts)) else ""
            with open(os.path.join(tx_dir, f"{aug.aug_id}.txt"), "w", encoding="utf-8") as f:
                f.write(aug_hyp.strip() + "\n")

            aug_meta = aug.to_summary_dict()
            aug_meta["transcript"] = aug_hyp
            aug_meta["reference"] = reference_transcript or ""
            aug_meta["wav_rel_path"] = f"augmented/{aug_fname}"

            with open(os.path.join(meta_dir, f"{aug.aug_id}.json"), "w", encoding="utf-8") as f:
                json.dump(aug_meta, f, indent=2, ensure_ascii=False)

            manifest_rows.append({
                "sample_id": aug.aug_id,
                "type": "augmented",
                "duration_sec": round(aug.duration_sec, 3),
                "transcript": aug_hyp,
                "reference": reference_transcript or "",
                "qc_status": aug.qc_result.status,
                "transforms": ";".join([op["op"] for op in aug.transform_chain]),
                "sha256": aug.qc_result.sha256_hash,
                "model_id": model_id,
            })
            results_list.append(aug_meta)

        # 3. Write manifest.csv and results.csv
        df_manifest = pd.DataFrame(manifest_rows)
        csv_manifest_path = os.path.join(session_dir, "manifest.csv")
        df_manifest.to_csv(csv_manifest_path, index=False, encoding="utf-8")

        csv_results_path = os.path.join(session_dir, "results.csv")
        df_manifest.to_csv(csv_results_path, index=False, encoding="utf-8")

        # 4. Write results.json
        json_results_path = os.path.join(session_dir, "results.json")
        full_json = {
            "session_id": session_id,
            "model_id": model_id,
            "original": orig_meta,
            "augmented_samples": results_list,
        }
        with open(json_results_path, "w", encoding="utf-8") as f:
            json.dump(full_json, f, indent=2, ensure_ascii=False)

        # 5. Build ZIP archive
        zip_path = os.path.join(self.base_export_dir, f"{session_id}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(session_dir):
                for file in files:
                    full_file = os.path.join(root, file)
                    rel_file = os.path.relpath(full_file, session_dir)
                    zipf.write(full_file, arcname=rel_file)

        return {
            "session_dir": session_dir,
            "zip_path": zip_path,
            "manifest_csv": csv_manifest_path,
            "results_json": json_results_path,
        }
