"""Export Service for Vietnamese Speech AI Studio.

Exports experimental results, model predictions, QC audits, and session audio
to CSV, JSON (Section 10), Markdown (Section 12), and downloadable ZIP packages.
"""

import json
import os
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional
import pandas as pd

from export.exporter import AugmentationSessionExporter


class ExportService:
    """Service handling multi-format export of experiments, sessions, and audits."""

    def __init__(self, base_export_dir: str = "exports"):
        self.base_export_dir = base_export_dir
        os.makedirs(self.base_export_dir, exist_ok=True)
        self.session_exporter = AugmentationSessionExporter(base_export_dir=base_export_dir)

    def export_experiment_bundle(
        self,
        experiment_id: str,
        results_df: pd.DataFrame,
        json_obj: Dict[str, Any],
        markdown_report: str,
    ) -> Dict[str, str]:
        """Export complete experiment package including CSV, JSON, Markdown, and ZIP."""
        timestamp = int(time.time())
        bundle_name = f"experiment_{experiment_id}_{timestamp}"
        bundle_dir = os.path.join(self.base_export_dir, bundle_name)
        os.makedirs(bundle_dir, exist_ok=True)

        # 1. Write CSV
        csv_path = os.path.join(bundle_dir, "results_summary.csv")
        results_df.to_csv(csv_path, index=False, encoding="utf-8")

        # 2. Write JSON
        json_path = os.path.join(bundle_dir, "experiment_result.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_obj, f, indent=2, ensure_ascii=False)

        # 3. Write Markdown
        md_path = os.path.join(bundle_dir, "experiment_report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_report)

        # 4. Package ZIP
        zip_base = os.path.join(self.base_export_dir, bundle_name)
        zip_path = shutil.make_archive(zip_base, "zip", bundle_dir)

        return {
            "bundle_dir": bundle_dir,
            "csv_path": csv_path,
            "json_path": json_path,
            "markdown_path": md_path,
            "zip_path": zip_path,
        }

    def export_session_bundle(
        self,
        session_id: str,
        original_audio_16k: Any,
        original_transcript: str,
        augmented_samples: List[Any],
        augmented_transcripts: List[str],
        reference_transcript: Optional[str] = None,
        model_id: str = "vinai/PhoWhisper-small",
    ) -> Dict[str, Any]:
        """Wrap existing AugmentationSessionExporter."""
        return self.session_exporter.export_session(
            session_id=session_id,
            original_audio_16k=original_audio_16k,
            original_transcript=original_transcript,
            augmented_samples=augmented_samples,
            augmented_transcripts=augmented_transcripts,
            reference_transcript=reference_transcript,
            model_id=model_id,
        )
