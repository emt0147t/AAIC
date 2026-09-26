"""Data Contamination and Leakage Quarantine Enforcer.

Enforces zero-leakage constraints:
- Tracks quarantined sample IDs (e.g. VIVOS test, ViMD test)
- Tracks cryptographic SHA-256 digests of normalized evaluation transcripts
- Prevents training or synthetic speech generation on evaluation data
- Raises DataContaminationError immediately upon any detected overlap
"""

import hashlib
import os
from typing import Dict, Iterable, List, Optional, Set, Tuple
import pandas as pd

try:
    from asr.normalization import normalize_vietnamese_text
except (ImportError, ValueError):
    from ..asr.normalization import normalize_vietnamese_text


class DataContaminationError(ValueError):
    """Raised when training data or synthetic speech attempts to use quarantined evaluation samples."""
    pass


def hash_normalized_text(text: str) -> str:
    """Compute deterministic SHA-256 hex digest of NFC-normalized, lowercased, punctuation-stripped text."""
    norm = normalize_vietnamese_text(text)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


class QuarantineEnforcer:
    """Enforces strict isolation between evaluation data and training / synthetic datasets."""

    def __init__(self):
        self.quarantined_sample_ids: Set[str] = set()
        self.quarantined_text_hashes: Set[str] = set()
        self.hash_to_reference_text: Dict[str, str] = {}

    def register_quarantined_sample(self, sample_id: str, transcript: Optional[str] = None) -> None:
        """Register a single evaluation sample into the quarantine registry."""
        sid_clean = str(sample_id).strip()
        if sid_clean:
            self.quarantined_sample_ids.add(sid_clean)

        if transcript:
            h = hash_normalized_text(transcript)
            self.quarantined_text_hashes.add(h)
            self.hash_to_reference_text[h] = str(transcript).strip()

    def register_quarantined_manifest(
        self,
        manifest_csv_path: str,
        id_col: str = "sample_id",
        transcript_col: str = "transcript",
    ) -> int:
        """Register all samples from an evaluation manifest CSV.

        Returns:
            Number of samples registered.
        """
        if not os.path.exists(manifest_csv_path):
            raise FileNotFoundError(f"Quarantine manifest file not found: {manifest_csv_path}")

        df = pd.read_csv(manifest_csv_path)

        # Fallback column detection if standard names differ
        id_candidates = [id_col, "id", "sample_id", "filename", "path"]
        found_id_col = next((c for c in id_candidates if c in df.columns), None)

        text_candidates = [transcript_col, "text", "transcript", "sentence", "normalized_transcript"]
        found_text_col = next((c for c in text_candidates if c in df.columns), None)

        count = 0
        for _, row in df.iterrows():
            sid = str(row[found_id_col]).strip() if found_id_col else ""
            txt = str(row[found_text_col]).strip() if found_text_col else ""
            self.register_quarantined_sample(sample_id=sid, transcript=txt)
            count += 1

        return count

    def check_sample(self, sample_id: str, transcript: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """Check if a sample overlaps with quarantined data.

        Returns:
            (is_safe, error_reason)
        """
        sid_clean = str(sample_id).strip()
        if sid_clean and sid_clean in self.quarantined_sample_ids:
            return False, f"Sample ID '{sid_clean}' is present in the quarantined evaluation registry"

        if transcript:
            h = hash_normalized_text(transcript)
            if h in self.quarantined_text_hashes:
                ref_txt = self.hash_to_reference_text.get(h, "")
                return False, f"Normalized transcript matches quarantined evaluation text: '{ref_txt}' (hash: {h[:12]}...)"

        return True, None

    def assert_safe(self, sample_id: str, transcript: Optional[str] = None, context: str = "") -> None:
        """Assert that a sample is safe for training / synthetic generation.

        Raises:
            DataContaminationError if any overlap is detected.
        """
        is_safe, reason = self.check_sample(sample_id, transcript)
        if not is_safe:
            ctx_str = f" in context [{context}]" if context else ""
            raise DataContaminationError(f"CRITICAL DATA CONTAMINATION DETECTED{ctx_str}: {reason}")
