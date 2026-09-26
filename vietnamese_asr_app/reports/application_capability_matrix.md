# Vietnamese Speech AI Studio — Application Capability Matrix

## 1. Executive Summary
The Vietnamese Speech AI Studio (`vietnamese_asr_app/app.py`) is an interactive, text-first engineering platform unifying inference, waveform augmentation, parameter-efficient fine-tuning (PEFT LoRA), synthetic speech generation, and reproducible research benchmarking.

---

## 2. Comprehensive Capability Matrix

| Feature Domain | Functional Capability | Status | Implementation Module / Service | Primary Outputs & Evidence |
| :--- | :--- | :--- | :--- | :--- |
| **ASR Inference** | Audio file ingestion & 16 kHz mono resampling | **VERIFIED** | `asr/audio.py`, `services/asr_service.py` | 1D float32 normalized waveform array |
| **Audio Quality** | 7-point pre-inference validation gate | **VERIFIED** | `asr/audio.py::validate_audio` | Pass/Fail status, RMS, peak, duration breakdown |
| **Decoding Safety** | Token clamping against Whisper architectural limit | **VERIFIED** | `asr/inference.py::compute_safe_max_new_tokens` | Clamped `effective_max_new_tokens` ($\le 448$) |
| **Speech Recognition** | PhoWhisper greedy & beam search inference | **VERIFIED** | `asr/inference.py::ASRInferenceEngine` | Raw transcript, normalized transcript, latency, RTF |
| **Model Selection** | PhoWhisper variant switching (tiny, base, small, med) | **VERIFIED** | `asr/model.py`, `services/asr_service.py` | Dynamic model swapping with pinned revisions |
| **LoRA Adaptation** | PEFT LoRA adapter dynamic attaching & detaching | **VERIFIED** | `services/asr_service.py`, `research/checkpoint.py` | `PeftModel` inference without base weight merging |
| **Parameter Audit** | Dynamic calculation of trainable vs frozen parameters | **VERIFIED** | `research/lora.py::count_parameters`, `services/lora_service.py` | Total: 37,908,096 \| Trainable: 147,456 (0.3890%) |
| **LoRA Protocol** | Read-only display of immutable Full E1 protocol | **VERIFIED** | `services/lora_service.py::FROZEN_FULL_E1_PROTOCOL` | Read-only hyperparameter and budget dashboard |
| **Execution Safety** | Safety preflight gate preventing unauthorized CPU runs | **VERIFIED** | `services/lora_service.py::validate_full_e1_execution_gate` | Strict execution block with clear CUDA requirement |
| **Augmentation** | 7 physical waveform transformations (K=1–20) | **VERIFIED** | `augmentation/transforms.py`, `augmentation/pipeline.py` | Gain, noise, shift, stretch, pitch, reverb, bandpass |
| **Augmentation QC** | 10-point audio quality control verification | **VERIFIED** | `augmentation/qc.py::run_10_point_qc` | Finite, non-clipping, duration, RMS, SHA-256 |
| **Synthetic Audio** | Gwen-TTS 0.6B lazy-loaded voice generation | **VERIFIED** | `synthetic/generator.py`, `services/synthetic_service.py` | 16 kHz mono PCM WAV utterances |
| **Speaker Policy** | Voice selection with zero real speaker cloning | **VERIFIED** | `synthetic/generator.py::SYNTHETIC_VOICE_MAP` | 9 built-in reference voices (`spk_synth_01`..`09`) |
| **Provenance Tracking** | Cryptographic sample lineage & audit logging | **VERIFIED** | `synthetic/provenance.py`, `services/synthetic_service.py` | Source text ID, dataset, hash, voice, timestamp |
| **Evaluation Metrics** | Ground-truth reference gated accuracy (WER/CER) | **VERIFIED** | `evaluation/metrics.py`, `services/evaluation_service.py` | Levenshtein WER, CER, substitutions, deletions, insertions |
| **Prediction Drift** | Non-reference consistency & augmentation drift | **VERIFIED** | `services/evaluation_service.py` | Explicit `PREDICTION DRIFT / CONSISTENCY` labeling |
| **Neutral Delta** | Bias-free comparative metrics (absolute & relative) | **VERIFIED** | `services/evaluation_service.py::compute_neutral_comparison` | `+0.28 pp` absolute, `+1.48%` relative (no winner tags) |
| **Experiment Registry** | Multi-experiment history (E0, E1, E2, Full E1) | **VERIFIED** | `services/experiment_service.py` | Pilot vs Benchmark labeling, neutral tables |
| **Training Curves** | Epoch loss curves from checkpoint metadata | **VERIFIED** | `services/experiment_service.py::load_training_curves` | Epoch-wise train/val loss, `N/A — NOT RECORDED` |
| **Numerical Report** | Section 10 machine-readable JSON object | **VERIFIED** | `services/experiment_service.py::generate_section10_json` | Formal JSON schema with explicit nulls |
| **Human Report** | Section 12 human-readable Markdown summary | **VERIFIED** | `services/experiment_service.py::generate_section12_markdown_report` | MEASURED vs ESTIMATED vs PROJECTED vs N/A |
| **Multi-Format Export**| Multi-artifact packaging to CSV, JSON, MD, and ZIP | **VERIFIED** | `services/export_service.py`, `export/exporter.py` | Downloadable ZIP bundles with manifests |
| **Batch Processing** | Sequential streaming audio transcription | **VERIFIED** | `app.py::handle_batch_transcription` | Low-memory sequential processing with CSV export |
| **Legal Governance** | Provenance registry & CC BY-NC-ND 4.0 enforcement | **VERIFIED** | `datasets/registry.py`, `app.py` | ViMD non-commercial policy, zero exposed tokens |

---

## 3. Measured Hardware & Benchmark Baseline
- **Hardware Device:** NVIDIA Tesla T4 (Measured Google Colab Preflight)
- **Step Latency:** 0.942 seconds / optimizer step (50 measured steps, warmup excluded)
- **Measured Throughput:** 33.95 samples/second
- **Peak VRAM Allocated:** 726.16 MB
- **Estimated Full E1 Training Duration:** 2,358 seconds (~39.3 minutes / 0.655 hours)
- **Local Host Execution Gate:** Blocked (PyTorch CPU runtime detected under Python 3.14).
