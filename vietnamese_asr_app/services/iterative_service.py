"""Iterative Audio Manipulation Service.

Coordinates cumulative waveform manipulations x_0 -> x_1 -> ... -> x_K.
Supports:
- Mode 1: Random Iterative Manipulation
- Mode 2: Best-Preserving Iterative Search (gated on reference accuracy or proxy consistency)
- 10-point audio QC gating after every iteration
- Comprehensive parent-child lineage tracking
- Multi-artifact export (WAVs, JSON metadata, CSV manifest, ZIP bundle)
"""

import hashlib
import json
import os
import shutil
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import soundfile as sf

from augmentation.iterative import IterativeAudioManipulator, IterativeStepRecord
from augmentation.qc import QualityCheckResult
from evaluation.metrics import compute_levenshtein_cer, compute_levenshtein_wer, evaluate_augmentation_consistency
from services.asr_service import ASRService


class IterativeManipulationService:
    """High-level service managing iterative cumulative audio manipulation pipelines."""

    def __init__(self, asr_service: Optional[ASRService] = None, base_export_dir: str = "exports"):
        self.asr_service = asr_service or ASRService(default_model_id="vinai/PhoWhisper-small")
        self.manipulator = IterativeAudioManipulator(sample_rate=16000)
        self.base_export_dir = base_export_dir
        os.makedirs(self.base_export_dir, exist_ok=True)

    def execute_chain(
        self,
        audio_16k: np.ndarray,
        k: int = 3,
        strategy: str = "random",  # "random" or "best_preserving"
        base_seed: int = 42,
        reference_transcript: Optional[str] = None,
        candidates_per_step: int = 3,
        enabled_operators: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Tuple[List[IterativeStepRecord], Dict[int, np.ndarray], Dict[str, Any]]:
        """Execute iterative manipulation chain producing x_1 ... x_K from x_0.
        
        Args:
            audio_16k: Input original waveform x_0 at 16kHz mono.
            k: Number of cumulative manipulation steps (1 <= K <= 20).
            strategy: 'random' or 'best_preserving'.
            base_seed: Deterministic RNG seed.
            reference_transcript: Optional ground-truth reference text.
            candidates_per_step: Number of candidate transforms M per step (for best_preserving).
            enabled_operators: Optional subset of allowed operators.
            progress_callback: Progress hook (progress_val, status_msg).
        """
        k = max(1, min(20, int(k)))
        candidates_per_step = max(2, min(8, int(candidates_per_step)))
        has_reference = bool(reference_transcript and reference_transcript.strip())
        ref_text = reference_transcript.strip() if has_reference else None

        records: List[IterativeStepRecord] = []
        audio_map: Dict[int, np.ndarray] = {}

        # ----------------------------------------------------------------------
        # STAGE 0: Evaluate Original Audio x_0 (Parent of x_1)
        # ----------------------------------------------------------------------
        if progress_callback:
            progress_callback(0.05, "Evaluating original audio x_0 through ASR and 10-point QC...")

        x_0 = audio_16k.copy().astype(np.float32)
        audio_map[0] = x_0

        qc_0, rms_0, peak_0 = self.manipulator.evaluate_qc(x_0)
        asr_0 = self.asr_service.transcribe(x_0)
        sha_0 = hashlib.sha256(x_0.tobytes()).hexdigest()

        wer_0: Optional[float] = None
        cer_0: Optional[float] = None
        if has_reference:
            w_res = compute_levenshtein_wer(ref_text, asr_0.raw_text)
            c_res = compute_levenshtein_cer(ref_text, asr_0.raw_text)
            wer_0 = round(w_res.error_rate * 100.0, 2)
            cer_0 = round(c_res.error_rate * 100.0, 2)
            score_0 = round(-(wer_0 * 10.0 + cer_0), 4)
            score_type_0 = "ACCURACY_WER_CER"
        else:
            score_0 = 100.0
            score_type_0 = "ASR_CONSISTENCY_QUALITY_PROXY"

        record_0 = IterativeStepRecord(
            iteration=0,
            parent_iteration=0,
            manipulation_name="None (Original Source)",
            parameters={},
            random_seed=base_seed,
            duration_sec=float(len(x_0)) / 16000.0,
            rms=rms_0,
            peak_abs=peak_0,
            qc_passed=qc_0.passed,
            qc_status=qc_0.status,
            failed_checks=qc_0.failed_checks,
            asr_raw_transcript=asr_0.raw_text,
            asr_normalized_transcript=asr_0.normalized_text,
            wer=wer_0,
            cer=cer_0,
            consistency_drift_wer=0.0,
            consistency_drift_cer=0.0,
            score=score_0,
            score_type=score_type_0,
            selected_as_best=True,
            audio_sha256=sha_0,
            filename="iteration_000.wav",
            parent_sha256=sha_0,
            child_sha256=sha_0,
            waveform_changed=False,
            max_abs_delta=0.0,
            mean_abs_delta=0.0,
            correlation_with_parent=1.0,
        )
        records.append(record_0)

        # ----------------------------------------------------------------------
        # CUMULATIVE CHAIN: x_i = T_i(x_{i-1}) for i = 1 ... K
        # ----------------------------------------------------------------------
        current_audio = x_0
        current_parent_idx = 0
        current_parent_tx = asr_0.raw_text
        last_operator: Optional[str] = None

        for i in range(1, k + 1):
            parent_sha = records[current_parent_idx].audio_sha256
            step_seed = base_seed + (i * 1000)
            if progress_callback:
                progress_callback(
                    0.05 + 0.85 * (i / k),
                    f"Iteration {i} of {k}: applying cumulative manipulation T_{i}(x_{{{i-1}}})...",
                )

            if strategy == "random":
                # Mode 1: Random iterative manipulation from x_{i-1}
                rng = np.random.default_rng(step_seed)
                # Anti-repetition: avoid consecutively sampling the same operator if alternatives exist
                op_name, params = self.manipulator.sample_random_operator(
                    rng=rng,
                    allowed_operators=enabled_operators,
                    exclude_operator=last_operator,
                )
                candidate_audio = self.manipulator.apply_operator(
                    current_audio, op_name, params, rng=rng
                )
                qc_res, rms_val, peak_val = self.manipulator.evaluate_qc(candidate_audio)

                # Verify non-identity
                min_len = min(len(candidate_audio), len(current_audio))
                if min_len > 0:
                    cand_max_diff = float(np.max(np.abs(candidate_audio[:min_len] - current_audio[:min_len])))
                    cand_mean_diff = float(np.mean(np.abs(candidate_audio[:min_len] - current_audio[:min_len])))
                    if len(candidate_audio) != len(current_audio):
                        cand_max_diff = max(cand_max_diff, 0.05)
                else:
                    cand_max_diff = 0.0
                    cand_mean_diff = 0.0

                if qc_res.passed and cand_max_diff >= 1e-4:
                    t_res = self.asr_service.transcribe(candidate_audio)
                    drift = evaluate_augmentation_consistency(
                        original_hyp=current_parent_tx,
                        augmented_hyp=t_res.raw_text,
                        reference_text=ref_text,
                    )
                    c_wer = round(drift.wer_vs_ref * 100.0, 2) if (has_reference and drift.wer_vs_ref is not None) else None
                    c_cer = round(drift.cer_vs_ref * 100.0, 2) if (has_reference and drift.cer_vs_ref is not None) else None
                    d_wer = round(drift.consistency_drift_wer * 100.0, 2)
                    d_cer = round(drift.consistency_drift_cer * 100.0, 2)

                    if has_reference:
                        step_score = round(-(c_wer * 10.0 + c_cer), 4)
                        score_type = "ACCURACY_WER_CER"
                    else:
                        step_score = round((100.0 - d_wer) - (d_cer * 0.1), 4)
                        score_type = "ASR_CONSISTENCY_QUALITY_PROXY"

                    sha_val = hashlib.sha256(candidate_audio.tobytes()).hexdigest()

                    # Compute correlation
                    a1 = current_audio[:min_len]
                    a2 = candidate_audio[:min_len]
                    std1, std2 = float(np.std(a1)), float(np.std(a2))
                    if std1 > 1e-6 and std2 > 1e-6:
                        corr_val = float(np.corrcoef(a1, a2)[0, 1])
                    else:
                        corr_val = 1.0

                    rec = IterativeStepRecord(
                        iteration=i,
                        parent_iteration=current_parent_idx,
                        manipulation_name=op_name,
                        parameters=params,
                        random_seed=step_seed,
                        duration_sec=float(len(candidate_audio)) / 16000.0,
                        rms=rms_val,
                        peak_abs=peak_val,
                        qc_passed=True,
                        qc_status=qc_res.status,
                        failed_checks=[],
                        asr_raw_transcript=t_res.raw_text,
                        asr_normalized_transcript=t_res.normalized_text,
                        wer=c_wer,
                        cer=c_cer,
                        consistency_drift_wer=d_wer,
                        consistency_drift_cer=d_cer,
                        score=step_score,
                        score_type=score_type,
                        selected_as_best=True,
                        audio_sha256=sha_val,
                        filename=f"iteration_{i:03d}.wav",
                        parent_sha256=parent_sha,
                        child_sha256=sha_val,
                        waveform_changed=bool(sha_val != parent_sha and cand_max_diff >= 1e-4),
                        max_abs_delta=cand_max_diff,
                        mean_abs_delta=cand_mean_diff,
                        correlation_with_parent=corr_val,
                    )
                    records.append(rec)
                    audio_map[i] = candidate_audio.copy()
                    current_audio = candidate_audio
                    current_parent_idx = i
                    current_parent_tx = t_res.raw_text
                    last_operator = op_name

                else:
                    fail_reasons = list(qc_res.failed_checks)
                    if cand_max_diff < 1e-4:
                        fail_reasons.append("degenerate_identity_transform")
                    rec = IterativeStepRecord(
                        iteration=i,
                        parent_iteration=current_parent_idx,
                        manipulation_name=op_name,
                        parameters=params,
                        random_seed=step_seed,
                        duration_sec=float(len(candidate_audio)) / 16000.0,
                        rms=rms_val,
                        peak_abs=peak_val,
                        qc_passed=False,
                        qc_status="FAILED",
                        failed_checks=fail_reasons,
                        asr_raw_transcript="",
                        asr_normalized_transcript="",
                        wer=None,
                        cer=None,
                        consistency_drift_wer=None,
                        consistency_drift_cer=None,
                        score=-999.0,
                        score_type="QC_FAILURE",
                        selected_as_best=False,
                        audio_sha256="",
                        filename=f"iteration_{i:03d}_failed.wav",
                        parent_sha256=parent_sha,
                        child_sha256="",
                        waveform_changed=False,
                        max_abs_delta=cand_max_diff,
                        mean_abs_delta=cand_mean_diff,
                        correlation_with_parent=1.0,
                    )
                    records.append(rec)

            else:
                # Mode 2: Best-Preserving Iterative Search from x_{i-1}
                candidates_evaluated = []
                pool_ops = enabled_operators or self.manipulator.AVAILABLE_OPERATORS
                # Filter out last operator if multiple operators available to ensure family diversity
                candidate_pool = [op for op in pool_ops if op != last_operator] if (last_operator and len(pool_ops) > 1 and last_operator in pool_ops) else pool_ops

                for m_idx in range(candidates_per_step):
                    cand_seed = step_seed + (m_idx + 1) * 37
                    c_rng = np.random.default_rng(cand_seed)
                    # Round-robin selection across operators in pool for maximal candidate diversity
                    cand_op = candidate_pool[m_idx % len(candidate_pool)]
                    _, c_params = self.manipulator.sample_random_operator(
                        rng=c_rng, allowed_operators=[cand_op]
                    )
                    c_audio = self.manipulator.apply_operator(
                        current_audio, cand_op, c_params, rng=c_rng
                    )
                    c_qc, c_rms, c_peak = self.manipulator.evaluate_qc(c_audio)

                    min_len = min(len(c_audio), len(current_audio))
                    if min_len > 0:
                        cand_max_diff = float(np.max(np.abs(c_audio[:min_len] - current_audio[:min_len])))
                        cand_mean_diff = float(np.mean(np.abs(c_audio[:min_len] - current_audio[:min_len])))
                        if len(c_audio) != len(current_audio):
                            cand_max_diff = max(cand_max_diff, 0.05)
                    else:
                        cand_max_diff = 0.0
                        cand_mean_diff = 0.0

                    # CRITICAL: Reject degenerate identity candidate
                    if (not c_qc.passed) or (cand_max_diff < 1e-3 and len(c_audio) == len(current_audio)):
                        fail_checks = list(c_qc.failed_checks)
                        if cand_max_diff < 1e-3:
                            fail_checks.append("degenerate_identity_transform")
                        candidates_evaluated.append({
                            "op": cand_op,
                            "params": c_params,
                            "seed": cand_seed,
                            "audio": c_audio,
                            "qc": c_qc,
                            "rms": c_rms,
                            "peak": c_peak,
                            "passed": False,
                            "max_diff": cand_max_diff,
                            "mean_diff": cand_mean_diff,
                            "score": -999.0,
                            "t_res": None,
                            "wer": None,
                            "cer": None,
                            "drift_wer": None,
                            "drift_cer": None,
                        })
                        continue

                    # Transcribe valid non-identity candidate
                    c_tres = self.asr_service.transcribe(c_audio)
                    c_drift = evaluate_augmentation_consistency(
                        original_hyp=current_parent_tx,
                        augmented_hyp=c_tres.raw_text,
                        reference_text=ref_text,
                    )
                    c_wer = round(c_drift.wer_vs_ref * 100.0, 2) if (has_reference and c_drift.wer_vs_ref is not None) else None
                    c_cer = round(c_drift.cer_vs_ref * 100.0, 2) if (has_reference and c_drift.cer_vs_ref is not None) else None
                    d_wer = round(c_drift.consistency_drift_wer * 100.0, 2)
                    d_cer = round(c_drift.consistency_drift_cer * 100.0, 2)

                    # Scoring objective: reward high intelligibility + slight bonus for non-trivial acoustic transformation
                    if has_reference:
                        score_val = -(c_wer * 10.0 + c_cer)
                    else:
                        score_val = (100.0 - d_wer) - (d_cer * 0.1)

                    candidates_evaluated.append({
                        "op": cand_op,
                        "params": c_params,
                        "seed": cand_seed,
                        "audio": c_audio,
                        "qc": c_qc,
                        "rms": c_rms,
                        "peak": c_peak,
                        "passed": True,
                        "max_diff": cand_max_diff,
                        "mean_diff": cand_mean_diff,
                        "score": round(score_val, 4),
                        "t_res": c_tres,
                        "wer": c_wer,
                        "cer": c_cer,
                        "drift_wer": d_wer,
                        "drift_cer": d_cer,
                    })

                valid_cands = [c for c in candidates_evaluated if c["passed"]]
                if valid_cands:
                    best_cand = max(valid_cands, key=lambda c: c["score"])
                    sha_val = hashlib.sha256(best_cand["audio"].tobytes()).hexdigest()

                    min_len = min(len(best_cand["audio"]), len(current_audio))
                    a1 = current_audio[:min_len]
                    a2 = best_cand["audio"][:min_len]
                    std1, std2 = float(np.std(a1)), float(np.std(a2))
                    if std1 > 1e-6 and std2 > 1e-6:
                        corr_val = float(np.corrcoef(a1, a2)[0, 1])
                    else:
                        corr_val = 1.0

                    rec = IterativeStepRecord(
                        iteration=i,
                        parent_iteration=current_parent_idx,
                        manipulation_name=best_cand["op"],
                        parameters=best_cand["params"],
                        random_seed=best_cand["seed"],
                        duration_sec=float(len(best_cand["audio"])) / 16000.0,
                        rms=best_cand["rms"],
                        peak_abs=best_cand["peak"],
                        qc_passed=True,
                        qc_status=best_cand["qc"].status,
                        failed_checks=[],
                        asr_raw_transcript=best_cand["t_res"].raw_text,
                        asr_normalized_transcript=best_cand["t_res"].normalized_text,
                        wer=best_cand["wer"],
                        cer=best_cand["cer"],
                        consistency_drift_wer=best_cand["drift_wer"],
                        consistency_drift_cer=best_cand["drift_cer"],
                        score=best_cand["score"],
                        score_type="ACCURACY_WER_CER" if has_reference else "ASR_CONSISTENCY_QUALITY_PROXY",
                        selected_as_best=True,
                        audio_sha256=sha_val,
                        filename=f"iteration_{i:03d}.wav",
                        parent_sha256=parent_sha,
                        child_sha256=sha_val,
                        waveform_changed=bool(sha_val != parent_sha and best_cand["max_diff"] >= 1e-4),
                        max_abs_delta=best_cand["max_diff"],
                        mean_abs_delta=best_cand["mean_diff"],
                        correlation_with_parent=corr_val,
                    )
                    records.append(rec)
                    audio_map[i] = best_cand["audio"].copy()
                    current_audio = best_cand["audio"]
                    current_parent_idx = i
                    current_parent_tx = best_cand["t_res"].raw_text
                    last_operator = best_cand["op"]
                else:
                    # All candidates failed QC
                    rec = IterativeStepRecord(
                        iteration=i,
                        parent_iteration=current_parent_idx,
                        manipulation_name="all_candidates_failed_qc",
                        parameters={},
                        random_seed=step_seed,
                        duration_sec=float(len(current_audio)) / 16000.0,
                        rms=rms_0,
                        peak_abs=peak_0,
                        qc_passed=False,
                        qc_status="FAILED",
                        failed_checks=["all_candidates_rejected_by_qc"],
                        asr_raw_transcript="",
                        asr_normalized_transcript="",
                        wer=None,
                        cer=None,
                        consistency_drift_wer=None,
                        consistency_drift_cer=None,
                        score=-999.0,
                        score_type="QC_FAILURE",
                        selected_as_best=False,
                        audio_sha256="",
                        filename=f"iteration_{i:03d}_failed.wav",
                        parent_sha256=parent_sha,
                        child_sha256="",
                        waveform_changed=False,
                        max_abs_delta=0.0,
                        mean_abs_delta=0.0,
                        correlation_with_parent=1.0,
                    )
                    records.append(rec)

        summary_meta = {
            "strategy": strategy,
            "k_steps": k,
            "base_seed": base_seed,
            "has_reference": has_reference,
            "score_mode": "ACCURACY_WER_CER" if has_reference else "ASR_CONSISTENCY_QUALITY_PROXY",
            "successful_iterations": sum(1 for r in records if r.qc_passed and r.iteration > 0),
            "failed_iterations": sum(1 for r in records if not r.qc_passed and r.iteration > 0),
        }

        if progress_callback:
            progress_callback(1.0, f"Completed iterative manipulation chain: {k} steps processed.")

        return records, audio_map, summary_meta

    def export_chain_bundle(
        self,
        records: List[IterativeStepRecord],
        audio_map: Dict[int, np.ndarray],
        original_audio_id: str = "sample_000",
        export_dirname: Optional[str] = None,
    ) -> Dict[str, str]:
        """Export all WAVs, metadata JSON, manifest CSV, and ZIP bundle."""
        ts = int(time.time())
        dirname = export_dirname or f"iterative_chain_{original_audio_id}_{ts}"
        bundle_dir = os.path.join(self.base_export_dir, dirname)
        wav_dir = os.path.join(bundle_dir, "wavs")
        os.makedirs(wav_dir, exist_ok=True)

        # 1. Save WAV files
        for rec in records:
            if rec.iteration in audio_map:
                w_path = os.path.join(wav_dir, rec.filename)
                sf.write(w_path, audio_map[rec.iteration], 16000, subtype="PCM_16")

        # 2. Save JSON metadata
        meta_items = [r.to_dict() for r in records]
        json_path = os.path.join(bundle_dir, "iterative_manipulation_metadata.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "original_audio_id": original_audio_id,
                "timestamp_utc": time.asctime(),
                "total_steps": len(records) - 1,
                "chain": meta_items,
            }, f, indent=2, ensure_ascii=False)

        # 3. Save CSV manifest
        manifest_rows = []
        for r in records:
            row = r.to_dict()
            row["parameters"] = json.dumps(row["parameters"])
            row["failed_checks"] = "; ".join(row["failed_checks"])
            manifest_rows.append(row)

        df = pd.DataFrame(manifest_rows)
        csv_path = os.path.join(bundle_dir, "iterative_manipulation_manifest.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8")

        # 4. Create ZIP package
        zip_base = os.path.join(self.base_export_dir, dirname)
        zip_path = shutil.make_archive(zip_base, "zip", bundle_dir)

        return {
            "bundle_dir": bundle_dir,
            "wav_dir": wav_dir,
            "json_path": json_path,
            "csv_path": csv_path,
            "zip_path": zip_path,
        }

    @staticmethod
    def format_records_dataframe(records: List[IterativeStepRecord]) -> pd.DataFrame:
        """Format step records into a user-friendly DataFrame for UI visualization."""
        display_rows = []
        for r in records:
            params_str = ", ".join(f"{k}={v}" for k, v in r.parameters.items()) if r.parameters else "-"
            wer_display = f"{r.wer:.2f}%" if r.wer is not None else "-"
            cer_display = f"{r.cer:.2f}%" if r.cer is not None else "-"
            drift_display = f"WER: {r.consistency_drift_wer:.1f}% | CER: {r.consistency_drift_cer:.1f}%" if r.consistency_drift_wer is not None else "-"
            changed_str = "YES" if r.waveform_changed else ("NO (Orig)" if r.iteration == 0 else "NO")

            display_rows.append({
                "Stage": f"x_{r.iteration}" if r.iteration > 0 else "x_0 (Orig)",
                "Iteration": r.iteration,
                "Parent": f"x_{r.parent_iteration}",
                "Operator": r.manipulation_name,
                "Parameters": params_str,
                "Changed": changed_str,
                "Max |Δ|": f"{r.max_abs_delta:.4f}" if r.iteration > 0 else "-",
                "Corr": f"{r.correlation_with_parent:.3f}" if r.iteration > 0 else "-",
                "Duration (s)": round(r.duration_sec, 2),
                "RMS": round(r.rms, 4),
                "Peak": round(r.peak_abs, 2),
                "QC Status": r.qc_status,
                "Transcript": r.asr_raw_transcript or "(QC Failed / None)",
                "WER vs Ref": wer_display,
                "CER vs Ref": cer_display,
                "Drift vs Parent": drift_display,
                "Score": f"{r.score:.2f}",
                "Score Mode": r.score_type,
            })
        return pd.DataFrame(display_rows)
