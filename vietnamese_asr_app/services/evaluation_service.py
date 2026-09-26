"""Evaluation Service for Vietnamese Speech AI Studio.

Enforces ground-truth reference requirement for accuracy metrics (WER/CER).
Labels missing-reference evaluations explicitly as PREDICTION DRIFT / CONSISTENCY.
Calculates neutral descriptive deltas (absolute and relative) without biased winner labels.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from evaluation.metrics import (
    MetricBreakdown,
    compute_levenshtein_cer,
    compute_levenshtein_wer,
    evaluate_augmentation_consistency,
)


@dataclass
class EvaluationMetricBundle:
    mode: str  # "ACCURACY" or "PREDICTION_DRIFT_CONSISTENCY"
    wer: Optional[float]
    cer: Optional[float]
    exact_match: Optional[bool]
    substitutions: Optional[int]
    deletions: Optional[int]
    insertions: Optional[int]
    summary_text: str


class EvaluationService:
    """Service handling ASR evaluation, reference transcript gating, and neutral comparison."""

    @staticmethod
    def evaluate_prediction(
        hypothesis: str,
        reference: Optional[str] = None,
        baseline_hypothesis: Optional[str] = None,
    ) -> EvaluationMetricBundle:
        """Evaluate hypothesis against reference (Accuracy) or baseline hypothesis (Drift).
        
        Strictly requires reference before reporting WER/CER as accuracy.
        """
        if reference and reference.strip():
            # Mode A: Ground-truth reference available -> ACCURACY
            wer_res = compute_levenshtein_wer(reference.strip(), hypothesis.strip())
            cer_res = compute_levenshtein_cer(reference.strip(), hypothesis.strip())
            em = hypothesis.strip().lower() == reference.strip().lower()

            summary = (
                f"Accuracy vs Reference: WER = {wer_res.error_rate * 100:.2f}% "
                f"(S={wer_res.substitutions}, D={wer_res.deletions}, I={wer_res.insertions}) | "
                f"CER = {cer_res.error_rate * 100:.2f}% | Exact Match: {'YES' if em else 'NO'}"
            )
            return EvaluationMetricBundle(
                mode="ACCURACY",
                wer=round(wer_res.error_rate * 100.0, 2),
                cer=round(cer_res.error_rate * 100.0, 2),
                exact_match=em,
                substitutions=wer_res.substitutions,
                deletions=wer_res.deletions,
                insertions=wer_res.insertions,
                summary_text=summary,
            )

        elif baseline_hypothesis and baseline_hypothesis.strip():
            # Mode B: No reference supplied, comparing against baseline hypothesis -> PREDICTION DRIFT
            drift_res = evaluate_augmentation_consistency(
                original_hyp=baseline_hypothesis.strip(),
                augmented_hyp=hypothesis.strip(),
                reference_text=None,
            )
            summary = (
                f"PREDICTION DRIFT / CONSISTENCY (Reference missing - Not accuracy): "
                f"Drift WER = {drift_res.consistency_drift_wer * 100:.2f}% | "
                f"Drift CER = {drift_res.consistency_drift_cer * 100:.2f}%"
            )
            return EvaluationMetricBundle(
                mode="PREDICTION_DRIFT_CONSISTENCY",
                wer=round(drift_res.consistency_drift_wer * 100.0, 2),
                cer=round(drift_res.consistency_drift_cer * 100.0, 2),
                exact_match=(baseline_hypothesis.strip().lower() == hypothesis.strip().lower()),
                substitutions=None,
                deletions=None,
                insertions=None,
                summary_text=summary,
            )

        else:
            # Mode C: No reference and no baseline provided
            return EvaluationMetricBundle(
                mode="UNAVAILABLE",
                wer=None,
                cer=None,
                exact_match=None,
                substitutions=None,
                deletions=None,
                insertions=None,
                summary_text="Reference transcript required to compute WER/CER accuracy.",
            )

    @staticmethod
    def compute_neutral_comparison(
        base_wer: float,
        adapted_wer: float,
        base_cer: float,
        adapted_cer: float,
        label_a: str = "Base Model",
        label_b: str = "LoRA-Adapted Model",
    ) -> Dict[str, Any]:
        """Compute neutral descriptive deltas between two model runs without promotional bias."""
        delta_wer_abs = adapted_wer - base_wer
        delta_cer_abs = adapted_cer - base_cer

        rel_wer = (delta_wer_abs / base_wer * 100.0) if base_wer > 0 else 0.0
        rel_cer = (delta_cer_abs / base_cer * 100.0) if base_cer > 0 else 0.0

        wer_sign = "+" if delta_wer_abs > 0 else ""
        cer_sign = "+" if delta_cer_abs > 0 else ""

        narrative = (
            f"Comparison ({label_a} vs {label_b}):\n"
            f"- WER: {base_wer:.2f}% → {adapted_wer:.2f}% (Delta: {wer_sign}{delta_wer_abs:.2f} percentage points, relative {wer_sign}{rel_wer:.2f}%)\n"
            f"- CER: {base_cer:.2f}% → {adapted_cer:.2f}% (Delta: {cer_sign}{delta_cer_abs:.2f} percentage points, relative {cer_sign}{rel_cer:.2f}%)"
        )

        return {
            "label_a": label_a,
            "label_b": label_b,
            "base_wer": base_wer,
            "adapted_wer": adapted_wer,
            "delta_wer_absolute": round(delta_wer_abs, 2),
            "delta_wer_relative_pct": round(rel_wer, 2),
            "base_cer": base_cer,
            "adapted_cer": adapted_cer,
            "delta_cer_absolute": round(delta_cer_abs, 2),
            "delta_cer_relative_pct": round(rel_cer, 2),
            "narrative": narrative,
        }

    @staticmethod
    def aggregate_batch_statistics(batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate corpus-level metrics across a batch run."""
        total = len(batch_results)
        if total == 0:
            return {
                "total_samples": 0,
                "successful_samples": 0,
                "failed_samples": 0,
                "mean_latency_sec": 0.0,
                "median_latency_sec": 0.0,
                "mean_rtf": 0.0,
                "median_rtf": 0.0,
                "micro_wer": None,
                "micro_cer": None,
            }

        successful = [r for r in batch_results if r.get("status") == "PASS"]
        failed_count = total - len(successful)

        latencies = [r["latency_sec"] for r in successful if "latency_sec" in r]
        rtfs = [r["rtf"] for r in successful if "rtf" in r]

        total_words = 0
        total_word_errors = 0
        total_chars = 0
        total_char_errors = 0

        for r in successful:
            if "reference" in r and r["reference"]:
                ref = r["reference"]
                hyp = r.get("hypothesis", "")
                w_res = compute_levenshtein_wer(ref, hyp)
                c_res = compute_levenshtein_cer(ref, hyp)
                total_words += w_res.total_reference_tokens
                total_word_errors += (w_res.substitutions + w_res.deletions + w_res.insertions)
                total_chars += c_res.total_reference_tokens
                total_char_errors += (c_res.substitutions + c_res.deletions + c_res.insertions)

        micro_wer = round(100.0 * total_word_errors / max(total_words, 1), 2) if total_words > 0 else None
        micro_cer = round(100.0 * total_char_errors / max(total_chars, 1), 2) if total_chars > 0 else None

        return {
            "total_samples": total,
            "successful_samples": len(successful),
            "failed_samples": failed_count,
            "mean_latency_sec": round(float(np.mean(latencies)), 3) if latencies else 0.0,
            "median_latency_sec": round(float(np.median(latencies)), 3) if latencies else 0.0,
            "mean_rtf": round(float(np.mean(rtfs)), 3) if rtfs else 0.0,
            "median_rtf": round(float(np.median(rtfs)), 3) if rtfs else 0.0,
            "micro_wer": micro_wer,
            "micro_cer": micro_cer,
        }
