"""Experiment and Results Tracking Service for Vietnamese Speech AI Studio.

Maintains registry across all experiments (E0, E1_PILOT, E2_REAL_SYNTH_PILOT, FULL_E1).
Differentiates between PIPELINE PILOT and FULL BENCHMARK.
Generates Section 10 machine-readable JSON result object and Section 12 Markdown report.
"""

import json
import os
from typing import Any, Dict, List, Optional
import pandas as pd


class ExperimentService:
    """Service tracking historical experiments, training metrics, and report formatting."""

    def __init__(self, reports_dir: str = "reports", checkpoints_dir: str = "checkpoints"):
        self.reports_dir = reports_dir
        self.checkpoints_dir = checkpoints_dir

    def get_all_experiment_records(self) -> List[Dict[str, Any]]:
        """Return unified table of all registered experiments."""
        records = [
            {
                "experiment_id": "E0",
                "label": "E0_baseline",
                "category": "PIPELINE PILOT — NOT FINAL BENCHMARK",
                "model": "vinai/PhoWhisper-tiny",
                "adapter": "None (Zero-shot Base)",
                "data": "None (Pretrained)",
                "samples": 9,
                "wer": 18.93,
                "cer": 11.07,
                "exact_match": "N/A",
                "mean_latency_s": 1.412,
                "median_latency_s": 1.390,
                "mean_rtf": 0.140,
                "trainable_parameters": 0,
                "synthetic_ratio": 0.0,
                "status": "COMPLETED (PILOT)",
            },
            {
                "experiment_id": "E1_PILOT",
                "label": "E1_lora_real",
                "category": "PIPELINE PILOT — NOT FINAL BENCHMARK",
                "model": "vinai/PhoWhisper-tiny",
                "adapter": "LoRA (r=8, alpha=16, q/v_proj)",
                "data": "22 Real Speech (VIVOS + ViMD)",
                "samples": 9,
                "wer": 19.21,
                "cer": 12.16,
                "exact_match": "N/A",
                "mean_latency_s": 1.264,
                "median_latency_s": 1.240,
                "mean_rtf": 0.124,
                "trainable_parameters": 147456,
                "synthetic_ratio": 0.0,
                "status": "COMPLETED (PILOT)",
            },
            {
                "experiment_id": "E2_REAL_SYNTH_PILOT",
                "label": "E2_lora_real_synth",
                "category": "PIPELINE PILOT — NOT FINAL BENCHMARK",
                "model": "vinai/PhoWhisper-tiny",
                "adapter": "LoRA (r=8, alpha=16, q/v_proj)",
                "data": "15 Real + 7 Synth (Gwen-TTS)",
                "samples": 9,
                "wer": 19.21,
                "cer": 12.16,
                "exact_match": "N/A",
                "mean_latency_s": 1.389,
                "median_latency_s": 1.360,
                "mean_rtf": 0.137,
                "trainable_parameters": 147456,
                "synthetic_ratio": 0.318,
                "status": "COMPLETED (PILOT)",
            },
            {
                "experiment_id": "FULL_E1",
                "label": "Full_E1_26k",
                "category": "FULL BENCHMARK (FROZEN PROTOCOL)",
                "model": "vinai/PhoWhisper-tiny (cc51d32)",
                "adapter": "LoRA (r=8, alpha=16, q/v_proj)",
                "data": "26,671 Training-Ready (VIVOS + ViMD)",
                "samples": 9,
                "wer": "FULL E1 RESULT: NOT YET AVAILABLE",
                "cer": "FULL E1 RESULT: NOT YET AVAILABLE",
                "exact_match": "N/A",
                "mean_latency_s": "N/A",
                "median_latency_s": "N/A",
                "mean_rtf": "N/A",
                "trainable_parameters": 147456,
                "synthetic_ratio": 0.0,
                "status": "NOT YET AVAILABLE (Awaiting Cloud GPU Execution)",
            },
        ]
        return records

    def get_results_dataframe(self) -> pd.DataFrame:
        """Format experiments into user-facing pandas DataFrame."""
        records = self.get_all_experiment_records()
        display_rows = []
        for r in records:
            display_rows.append({
                "Experiment ID": r["experiment_id"],
                "Category / Protocol": r["category"],
                "Model Architecture": r["model"],
                "Adapter": r["adapter"],
                "Training Corpus": r["data"],
                "WER (%)": f"{r['wer']:.2f}%" if isinstance(r["wer"], (int, float)) else str(r["wer"]),
                "CER (%)": f"{r['cer']:.2f}%" if isinstance(r["cer"], (int, float)) else str(r["cer"]),
                "Mean Latency (s)": f"{r['mean_latency_s']:.3f}" if isinstance(r["mean_latency_s"], (int, float)) else str(r["mean_latency_s"]),
                "Mean RTF": f"{r['mean_rtf']:.3f}" if isinstance(r["mean_rtf"], (int, float)) else str(r["mean_rtf"]),
                "Trainable Params": f"{r['trainable_parameters']:,}" if isinstance(r["trainable_parameters"], int) else str(r["trainable_parameters"]),
                "Status": r["status"],
            })
        return pd.DataFrame(display_rows)

    def load_training_curves(self, experiment_id: str) -> Dict[str, Any]:
        """Load epoch-wise training loss and validation loss from actual checkpoint metadata."""
        ckpt_sub = "E1_lora_real" if "E1" in experiment_id else "E2_lora_real_synth" if "E2" in experiment_id else None
        if not ckpt_sub:
            return {"error": f"No training checkpoints associated with {experiment_id}"}

        exp_dir = os.path.join(self.checkpoints_dir, ckpt_sub)
        epochs = [1, 2, 3]
        train_loss = []
        val_loss = []

        for ep in epochs:
            meta_file = os.path.join(exp_dir, f"checkpoint_epoch_{ep}", "experiment_metadata.json")
            if os.path.exists(meta_file):
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        m = json.load(f)
                    metrics = m.get("metrics", {})
                    train_loss.append(metrics.get("train_loss"))
                    val_loss.append(metrics.get("val_loss"))
                except Exception:
                    train_loss.append(None)
                    val_loss.append(None)
            else:
                train_loss.append(None)
                val_loss.append(None)

        df = pd.DataFrame({
            "Epoch": epochs,
            "Train Loss": [f"{v:.4f}" if v is not None else "N/A — NOT RECORDED" for v in train_loss],
            "Val Loss": [f"{v:.4f}" if v is not None else "N/A — NOT RECORDED" for v in val_loss],
            "Val WER": ["N/A — NOT RECORDED"] * len(epochs),
            "Val CER": ["N/A — NOT RECORDED"] * len(epochs),
        })

        return {
            "experiment_id": experiment_id,
            "table": df,
            "has_recorded_curves": any(v is not None for v in train_loss),
        }

    def generate_section10_json(self, experiment_id: str = "FULL_E1") -> Dict[str, Any]:
        """Generate machine-readable JSON result object following Section 10 specification."""
        if experiment_id == "E0":
            return {
                "experiment_id": "E0",
                "model": {
                    "name": "vinai/PhoWhisper-tiny",
                    "revision": "cc51d32be916efebde04ff549854fa1741cb5c02",
                },
                "adapter": {
                    "enabled": False,
                    "path": None,
                    "peft_version": None,
                },
                "data": {
                    "dataset": "zero_shot",
                    "split": "test",
                    "sample_count": 9,
                },
                "metrics": {
                    "wer": 18.93,
                    "cer": 11.07,
                    "exact_match": None,
                    "mean_latency_s": 1.412,
                    "median_latency_s": 1.390,
                    "mean_rtf": 0.140,
                    "median_rtf": 0.138,
                },
                "training": {
                    "epochs": None,
                    "optimizer_steps": None,
                    "trainable_parameters": 0,
                    "training_duration_s": None,
                    "peak_vram_mb": None,
                },
                "synthetic": {
                    "enabled": False,
                    "generated_count": 0,
                    "qc_pass_count": 0,
                    "qc_fail_count": 0,
                },
            }

        elif experiment_id == "E1_PILOT":
            return {
                "experiment_id": "E1_PILOT",
                "model": {
                    "name": "vinai/PhoWhisper-tiny",
                    "revision": "cc51d32be916efebde04ff549854fa1741cb5c02",
                },
                "adapter": {
                    "enabled": True,
                    "path": "checkpoints/E1_lora_real/checkpoint_epoch_3",
                    "peft_version": "0.21.0",
                },
                "data": {
                    "dataset": "VIVOS_ViMD_pilot",
                    "split": "train",
                    "sample_count": 22,
                },
                "metrics": {
                    "wer": 19.21,
                    "cer": 12.16,
                    "exact_match": None,
                    "mean_latency_s": 1.264,
                    "median_latency_s": 1.240,
                    "mean_rtf": 0.124,
                    "median_rtf": 0.122,
                },
                "training": {
                    "epochs": 3,
                    "optimizer_steps": 33,
                    "trainable_parameters": 147456,
                    "training_duration_s": 18.4,
                    "peak_vram_mb": None,
                },
                "synthetic": {
                    "enabled": False,
                    "generated_count": 0,
                    "qc_pass_count": 0,
                    "qc_fail_count": 0,
                },
            }

        elif experiment_id == "E2_REAL_SYNTH_PILOT":
            return {
                "experiment_id": "E2_REAL_SYNTH_PILOT",
                "model": {
                    "name": "vinai/PhoWhisper-tiny",
                    "revision": "cc51d32be916efebde04ff549854fa1741cb5c02",
                },
                "adapter": {
                    "enabled": True,
                    "path": "checkpoints/E2_lora_real_synth/checkpoint_epoch_3",
                    "peft_version": "0.21.0",
                },
                "data": {
                    "dataset": "VIVOS_ViMD_Real_GwenTTS_Synth",
                    "split": "train",
                    "sample_count": 22,
                },
                "metrics": {
                    "wer": 19.21,
                    "cer": 12.16,
                    "exact_match": None,
                    "mean_latency_s": 1.389,
                    "median_latency_s": 1.360,
                    "mean_rtf": 0.137,
                    "median_rtf": 0.135,
                },
                "training": {
                    "epochs": 3,
                    "optimizer_steps": 33,
                    "trainable_parameters": 147456,
                    "training_duration_s": 19.1,
                    "peak_vram_mb": None,
                },
                "synthetic": {
                    "enabled": True,
                    "generated_count": 7,
                    "qc_pass_count": 7,
                    "qc_fail_count": 0,
                },
            }

        else:
            # FULL_E1 (Protocol audited, preflight measured, execution pending cloud run)
            return {
                "experiment_id": "FULL_E1",
                "model": {
                    "name": "vinai/PhoWhisper-tiny",
                    "revision": "cc51d32be916efebde04ff549854fa1741cb5c02",
                },
                "adapter": {
                    "enabled": True,
                    "path": "checkpoints/FULL_E1",
                    "peft_version": "0.21.0",
                },
                "data": {
                    "dataset": "VIVOS_train_plus_ViMD_train",
                    "split": "train",
                    "sample_count": 26671,
                },
                "metrics": {
                    "wer": None,
                    "cer": None,
                    "exact_match": None,
                    "mean_latency_s": None,
                    "median_latency_s": None,
                    "mean_rtf": None,
                    "median_rtf": None,
                },
                "training": {
                    "epochs": 3,
                    "optimizer_steps": 2502,
                    "trainable_parameters": 147456,
                    "training_duration_s": 2358.084,  # ESTIMATED based on measured T4 preflight (2502 * 0.942s)
                    "peak_vram_mb": 726.16,          # MEASURED during T4 CUDA preflight
                },
                "synthetic": {
                    "enabled": False,
                    "generated_count": 0,
                    "qc_pass_count": 0,
                    "qc_fail_count": 0,
                },
            }

    def generate_section12_markdown_report(self, experiment_id: str = "FULL_E1") -> str:
        """Generate human-readable Markdown summary following Section 12 specification."""
        if experiment_id == "FULL_E1":
            return """# EXPERIMENT SUMMARY

**Experiment ID:** FULL_E1  
**Protocol Status:** AUDITED & FROZEN (Pending Cloud Execution)  
**Model:** PhoWhisper-tiny  
**Revision:** `cc51d32be916efebde04ff549854fa1741cb5c02`  
**Dataset:** VIVOS train (11,660) + ViMD train clean (15,011)  
**Training samples:** 26,671 training-ready utterances  
**Validation samples:** 1,900 utterances  
**LoRA Configuration:** r=8, alpha=16, dropout=0.05, target_modules=["q_proj", "v_proj"], bias="none"  
**Training Parameters:** micro_batch=4, grad_accum=8, effective_batch=32, lr=1e-4, AdamW, cosine scheduler, epochs=3, total_steps=2,502, seed=42  

---

## RESULTS
- **WER:** FULL E1 RESULT: NOT YET AVAILABLE
- **CER:** FULL E1 RESULT: NOT YET AVAILABLE
- **Exact Match:** N/A
- **Latency:** N/A
- **RTF:** N/A

---

## TRAINING BENCHMARKS & PREFLIGHT
- **Preflight Measurement (MEASURED):** NVIDIA Tesla T4 GPU (0.942 s / optimizer step)
- **Peak VRAM Allocated (MEASURED):** 726.16 MB
- **Peak VRAM Reserved (MEASURED):** 812.00 MB
- **Full E1 Estimated Duration (ESTIMATED):** 2,358 seconds (~39.3 minutes / 0.655 hours)
- **Local Host Execution Status (MEASURED):** ABORTED (PyTorch CPU runtime detected; training blocked by execution safety gate)

---

## CHECKPOINT
- **Target Path:** `checkpoints/FULL_E1` (Awaiting Cloud GPU run via `ASR_FULL_E1_TRAINING.ipynb`)

---

## PROVENANCE & GOVERNANCE
- **VIVOS License:** CC BY-NC-SA 4.0
- **ViMD License:** CC BY-NC-ND 4.0 (Non-commercial, strictly audited, non-redistributable)
- **Test Set Contamination:** VERIFIED ZERO (Cryptographic SHA-256 CRLF hash: `efe54856d12613f109cebc9482a6f224ca4e2370d0ff3f8a2fe4ea0a013b60ca`)

---

## LIMITATIONS & VALUE DISTINCTIONS
- **MEASURED:** CUDA preflight step latency (0.942s) and VRAM allocation (726 MB) on Tesla T4.
- **ESTIMATED:** Total training duration (2,358s) derived from measured step latency.
- **N/A:** Final WER/CER metrics are pending actual cloud execution and will NOT be fabricated.
"""
        else:
            return f"""# EXPERIMENT SUMMARY

**Experiment ID:** {experiment_id}  
**Protocol Status:** PIPELINE PILOT — NOT FINAL BENCHMARK  
**Model:** PhoWhisper-tiny (`cc51d32`)  
**Evaluation Set:** 9-sample frozen ViMD test set  

---

## RESULTS (MEASURED PILOT METRICS)
- **WER:** 18.93% (E0 baseline) → 19.21% ({experiment_id})
- **CER:** 11.07% (E0 baseline) → 12.16% ({experiment_id})
- **Mean Latency:** ~1.26 – 1.41 s
- **Mean RTF:** ~0.124 – 0.140

---

## LIMITATIONS
- Evaluated on a 9-sample pilot test set designed for pipeline verification, not statistical significance.
"""
