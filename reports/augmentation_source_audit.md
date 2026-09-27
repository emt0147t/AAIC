# Training Augmentation Source Verification & Isolation Audit Report

> **D STATUS: PASS**  
> **Investigation Target:** Verification that training datasets (E1, E2, Full E1) contain zero compounding iterative audio from Tab 3, confirming strict quarantine of Tab 3 for post-training robustness probing only, and establishing hard guardrails for any future usage.

---

## 1. Executive Summary

An independent scientific audit questioned whether the training data for LoRA fine-tuning (E1, E2, or Full E1) ingested compounding iterative audio from Tab 3 ($x_i = T_i(x_{i-1})$), which could introduce severe acoustic degradation, feedback loops, or drift into training.

An exhaustive forensic line-by-line lineage audit was conducted across all training manifests, scripts, and dataloaders:
- **Full E1 Training:** **Zero** Tab 3 iterative audio. Sourced exclusively from 26,671 clean real utterances (11,660 VIVOS train + 15,011 ViMD train).
- **Phase 2 E1 Pilot:** **Zero** Tab 3 iterative audio. Sourced from 22 clean real utterances.
- **Phase 2 E2 Pilot:** **Zero** Tab 3 iterative audio. Sourced from 15 real utterances + 7 single-step Gwen-TTS synthesized samples ($K=1$, non-compounded, verified by 10-point QC).
- **Tab 3 Engine Status:** Confirmed **strictly quarantined** as an inference-time acoustic stress-testing tool.

---

## 2. Lineage Audit of Training Partitions

| Pipeline Stage | Manifest / Source | Iterative (Tab 3) Count | Real Speech Count | Synthetic (TTS) Count | Total Audio Samples | Audit Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Phase 2 E1** | `manifests/train_manifest.csv` | **0 (0.0%)** | 22 (100.0%) | 0 (0.0%) | 22 | **PASS: Pure Real** |
| **Phase 2 E2** | `manifests/train_manifest.csv` + `manifests/synthetic_qc_passed.csv` | **0 (0.0%)** | 15 (68.2%) | 7 (31.8% Gwen-TTS) | 22 | **PASS: Isolated TTS** |
| **Full E1** | `manifests/full_train_manifest.csv` / HF Streaming | **0 (0.0%)** | 26,671 (100.0%) | 0 (0.0%) | 26,671 | **PASS: Pure Real** |

---

## 3. Tab 3 Functional Isolation & Purpose

The iterative manipulation engine ([`vietnamese_asr_app/augmentation/iterative.py`](file:///d:/Vietnamese_ASR_Week6/vietnamese_asr_app/augmentation/iterative.py)) generates compounding transformations:

$$x_i = T_i(x_{i-1}), \quad i \in [1, K]$$

This module is integrated exclusively into:
1. **Application UI Tab 3 (`Compounding Iterative Manipulation`):** For interactive user exploration of degradation effects.
2. **Robustness Diagnostic Service (`services/iterative_service.py`):** For evaluating ASR transcription stability and CER/WER degradation curves under cumulative acoustic stress.

Tab 3 outputs are **never written** to `manifests/train_manifest.csv`, `manifests/full_train_manifest.csv`, or any training input stream.

---

## 4. Hard Guardrails for Future Usage

If compounding iterative audio from Tab 3 is ever proposed as training data in future iterations, the following 3 hard gates are enforced:

1. **Maximum Iteration Ceiling:** $K_{\max} \le 3$ (to prevent signal-to-noise ratio collapse and unintelligible harmonic distortion).
2. **Transcription Sanity Gate:** ASR WER Gate $\le 15\%$ against the ground truth reference to guarantee phonetic intelligibility before admitting any sample into training.
3. **Partition Isolation Invariant:** Compounded samples must strictly originate from the designated training split and remain disjoint from validation and test partitions.
