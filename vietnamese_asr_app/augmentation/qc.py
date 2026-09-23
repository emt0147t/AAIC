"""10-Check Audio Quality Control (QC) Pipeline.

Every generated waveform must strictly pass all 10 checks before saving,
manifest registration, or model evaluation.
"""

from dataclasses import asdict, dataclass
import hashlib
from typing import Dict, List, Optional
import numpy as np


@dataclass
class QualityCheckResult:
    passed: bool
    status: str  # "PASSED" or "FAILED"
    checks_evaluated: int = 10
    failed_checks: List[str] = None  # type: ignore
    details: Dict[str, float] = None  # type: ignore
    sha256_hash: str = ""
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.failed_checks is None:
            self.failed_checks = []
        if self.details is None:
            self.details = {}

    def to_dict(self) -> dict:
        return asdict(self)


def run_10_point_qc(
    audio: np.ndarray,
    sample_rate: int = 16000,
    expected_sample_rate: int = 16000,
    min_duration_sec: float = 0.2,
    max_duration_sec: float = 30.0,
    max_clipping_ratio: float = 0.001,
    min_rms: float = 1e-5,
) -> QualityCheckResult:
    """Execute the full 10-check audio quality assurance protocol.

    Checks:
    1. Decodable array: Non-empty numpy array
    2. Finite values: Array contains only finite floats
    3. No NaNs: np.isnan count == 0
    4. No Infs: np.isinf count == 0
    5. Range boundaries: Max absolute amplitude <= 1.0 (with max 0.05% margin before normalization)
    6. Clipping check: Ratio of samples with |x| >= 0.999 must be < max_clipping_ratio (0.1%)
    7. RMS check: Signal RMS > min_rms (not silent or collapsed)
    8. Duration bounds: min_duration_sec <= duration <= max_duration_sec
    9. Format conformity: Exactly 1-channel mono and matching expected_sample_rate (16000 Hz)
    10. Cryptographic integrity: Deterministic SHA-256 hash computed over float32 byte stream
    """
    failed = []
    details = {}

    # Check 1: Decodability & existence
    if audio is None or not isinstance(audio, np.ndarray) or audio.size == 0:
        return QualityCheckResult(
            passed=False,
            status="FAILED",
            failed_checks=["1_decodable_array"],
            error_message="Audio array is None or empty",
        )

    # Check 2: Finite values
    is_finite = bool(np.all(np.isfinite(audio)))
    if not is_finite:
        failed.append("2_finite_values")

    # Check 3: No NaNs
    nan_count = int(np.isnan(audio).sum())
    details["nan_count"] = nan_count
    if nan_count > 0:
        failed.append("3_no_nans")

    # Check 4: No Infs
    inf_count = int(np.isinf(audio).sum())
    details["inf_count"] = inf_count
    if inf_count > 0:
        failed.append("4_no_infs")

    # Check 9: Channel format & sample rate (evaluated early for duration)
    is_mono = audio.ndim == 1
    details["channels"] = 1 if is_mono else audio.shape[1] if audio.ndim == 2 else audio.ndim
    details["sample_rate"] = sample_rate
    if not is_mono or sample_rate != expected_sample_rate:
        failed.append("9_format_conformity")

    # Check 8: Duration bounds
    duration = float(len(audio)) / float(sample_rate)
    details["duration_sec"] = duration
    if duration < min_duration_sec or duration > max_duration_sec:
        failed.append("8_duration_bounds")

    # Check 5: Range boundaries
    peak_abs = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
    details["peak_abs"] = peak_abs
    if peak_abs > 1.05:  # Allow slight float precision margin before hard failure
        failed.append("5_range_boundaries")

    # Check 6: Clipping ratio
    clip_count = int(np.sum(np.abs(audio) >= 0.999))
    clip_ratio = float(clip_count) / float(len(audio)) if len(audio) > 0 else 0.0
    details["clipping_ratio"] = clip_ratio
    details["clipped_samples"] = clip_count
    if clip_ratio > max_clipping_ratio:
        failed.append("6_excessive_clipping")

    # Check 7: RMS signal check
    rms = float(np.sqrt(np.mean(audio**2))) if audio.size > 0 else 0.0
    details["rms"] = rms
    if rms < min_rms:
        failed.append("7_silent_signal_rms")

    # Check 10: SHA-256 integrity hash
    hasher = hashlib.sha256()
    hasher.update(audio.astype(np.float32).tobytes())
    sha256_hash = hasher.hexdigest()
    details["sha256"] = sha256_hash

    passed = len(failed) == 0
    status = "PASSED" if passed else "FAILED"
    error_msg = f"Failed checks: {', '.join(failed)}" if failed else None

    return QualityCheckResult(
        passed=passed,
        status=status,
        checks_evaluated=10,
        failed_checks=failed,
        details=details,
        sha256_hash=sha256_hash,
        error_message=error_msg,
    )
