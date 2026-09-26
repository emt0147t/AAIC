"""Provenance and Lineage Tracking for Synthetic Speech Utterances.

Maintains comprehensive metadata linking every synthetic sample strictly to its
originating training transcript, ensuring zero contamination and full auditability.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import peft
import torch
import transformers


def get_git_commit_hash() -> str:
    """Retrieve current git commit hash."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out
    except Exception:
        return "unknown"


def get_software_versions() -> Dict[str, str]:
    """Capture exact software dependency versions."""
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "qwen_tts": "0.1.1",
    }


def compute_file_sha256(filepath: str) -> str:
    """Compute cryptographic SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass
class SyntheticSampleProvenance:
    synthetic_id: str
    source_text_id: str
    source_dataset: str
    source_split: str
    transcript: str
    tts_model: str
    tts_model_revision: str
    voice_id: str
    built_in_voice: str
    generation_parameters: str  # JSON-encoded string
    original_sample_rate: int
    final_sample_rate: int
    duration_sec: float
    qc_status: str  # "PASSED" or "FAILED"
    audio_sha256: str
    audio_rel_path: str
    git_commit: str
    generation_timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
