"""ASR Evaluation Metrics and Robustness Diagnostics.

Computes exact Levenshtein Word Error Rate (WER), Character Error Rate (CER),
alignment breakdowns (S, D, I, N), and augmentation consistency scores.
"""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional
import jiwer
try:
    from asr.normalization import normalize_vietnamese_text
except (ImportError, ValueError):
    from ..asr.normalization import normalize_vietnamese_text


@dataclass
class MetricBreakdown:
    error_rate: float
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    total_reference_tokens: int

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ASRComparisonEvaluation:
    reference_text: Optional[str]
    original_hypothesis: str
    augmented_hypothesis: str
    wer_vs_ref: Optional[float]
    cer_vs_ref: Optional[float]
    consistency_drift_cer: float  # Levenshtein distance between original and augmented hyp
    consistency_drift_wer: float


def compute_levenshtein_wer(
    reference: str,
    hypothesis: str,
    normalize: bool = True,
) -> MetricBreakdown:
    """Compute exact Word Error Rate (WER) with alignment details."""
    ref_str = normalize_vietnamese_text(reference) if normalize else reference
    hyp_str = normalize_vietnamese_text(hypothesis) if normalize else hypothesis

    if not ref_str:
        hyp_words = hyp_str.split()
        return MetricBreakdown(
            error_rate=1.0 if hyp_words else 0.0,
            substitutions=0,
            deletions=0,
            insertions=len(hyp_words),
            hits=0,
            total_reference_tokens=0,
        )

    out = jiwer.process_words(ref_str, hyp_str)
    return MetricBreakdown(
        error_rate=float(out.wer),
        substitutions=int(out.substitutions),
        deletions=int(out.deletions),
        insertions=int(out.insertions),
        hits=int(out.hits),
        total_reference_tokens=int(len(ref_str.split())),
    )


def compute_levenshtein_cer(
    reference: str,
    hypothesis: str,
    normalize: bool = True,
) -> MetricBreakdown:
    """Compute exact Character Error Rate (CER) with alignment details."""
    ref_str = normalize_vietnamese_text(reference) if normalize else reference
    hyp_str = normalize_vietnamese_text(hypothesis) if normalize else hypothesis

    if not ref_str:
        return MetricBreakdown(
            error_rate=1.0 if hyp_str else 0.0,
            substitutions=0,
            deletions=0,
            insertions=len(hyp_str),
            hits=0,
            total_reference_tokens=0,
        )

    out = jiwer.process_characters(ref_str, hyp_str)
    return MetricBreakdown(
        error_rate=float(out.cer),
        substitutions=int(out.substitutions),
        deletions=int(out.deletions),
        insertions=int(out.insertions),
        hits=int(out.hits),
        total_reference_tokens=int(len(ref_str)),
    )


def evaluate_augmentation_consistency(
    original_hyp: str,
    augmented_hyp: str,
    reference_text: Optional[str] = None,
) -> ASRComparisonEvaluation:
    """Evaluate stability of transcription before and after augmentation."""
    norm_orig = normalize_vietnamese_text(original_hyp)
    norm_aug = normalize_vietnamese_text(augmented_hyp)

    drift_wer = float(jiwer.wer(norm_orig, norm_aug)) if norm_orig else (0.0 if not norm_aug else 1.0)
    drift_cer = float(jiwer.cer(norm_orig, norm_aug)) if norm_orig else (0.0 if not norm_aug else 1.0)

    wer_ref = None
    cer_ref = None
    if reference_text:
        norm_ref = normalize_vietnamese_text(reference_text)
        wer_ref = float(jiwer.wer(norm_ref, norm_aug))
        cer_ref = float(jiwer.cer(norm_ref, norm_aug))

    return ASRComparisonEvaluation(
        reference_text=reference_text,
        original_hypothesis=norm_orig,
        augmented_hypothesis=norm_aug,
        wer_vs_ref=wer_ref,
        cer_vs_ref=cer_ref,
        consistency_drift_cer=drift_cer,
        consistency_drift_wer=drift_wer,
    )
