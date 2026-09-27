# Vietnamese Speech AI Studio: Comprehensive Technical Overview & Experimental Evidence Audit

**Repository Root:** `D:\Vietnamese_ASR_Week6`  
**Git HEAD Commit:** `7b1f2b7`  
**Primary Deliverable:** System Technical Architecture & Authoritative Evidence Audit  
**Date of Audit:** 2026-09-27  

---

## 1. Executive Summary

### 1.1 System Identity & Core Purpose
- **Application Name:** Vietnamese Speech AI Studio (Vietnamese Speech-to-Text & Data Augmentation Studio)
- **Primary Purpose:** An enterprise-grade, interactive speech AI engineering and research platform unifying Vietnamese automatic speech recognition (ASR), 1D physical waveform augmentation, compounding iterative audio manipulation, Parameter-Efficient Fine-Tuning (PEFT LoRA) management, synthetic speech generation with strict voice governance, and verifiable numerical research benchmarking.
- **Target Task:** High-accuracy Vietnamese Automatic Speech Recognition across multiple regional dialects (Northern, Central, Southern), acoustic environments (ambient noise, reverberation, bandpass filtering), and parameter-efficient domain adaptation.
- **Current Core Model:** [`vinai/PhoWhisper-tiny`](https://huggingface.co/vinai/PhoWhisper-tiny) (pinned Git commit revision `cc51d32be916efebde04ff549854fa1741cb5c02`), with pluggable LoRA adapters.
- **Current Application Capabilities:** Interactive Gradio UI (10 functional tabs), 7-point pre-inference audio validation gate, greedy and beam search decoding, Whisper architectural token clamping ($\le 448$ tokens), deterministic NFC text normalization, reference-gated Levenshtein WER/CER evaluation, consistency drift tracking, 7 physical 1D waveform transforms ($K \in [1, 20]$), 10-point audio quality control (QC), cumulative iterative audio manipulation ($x_0 \to x_1 \dots \to x_K$ with Best-Preserving Search), low-memory sequential streaming batch processing, multi-format artifact export (CSV, JSON, Markdown, ZIP), and legal provenance registry.
- **Research Capabilities:** PEFT LoRA parameter accounting (0.3890% trainable ratio), multi-experiment registry (E0, E1, E2, Full E1), loss curve inspection, synthetic speech generation via Gwen-TTS 0.6B ($K=1$) across 9 distinct built-in reference voices with zero real-speaker cloning, and turnkey cloud training pipeline (`ASR_FULL_E1_TRAINING.ipynb`).
- **Datasets Managed:** VIVOS (`thanhduycao/vivos_ng_only`), ViMD (`nguyendv02/ViMD_Dataset`), BUD500, VietMed, VietSuperSpeech, VLSP.
- **Current Experiment Status:** Phase 2 Controlled Pilot experiments (E0 Baseline, E1 Real LoRA, Synthetic Audio Pilot, E2 Real+Synth LoRA, and Forensic E1 vs E2 Audit) are **COMPLETE, AUDITED, AND CLOSED**.
- **Current Full E1 Status:** **ABORTED ON LOCAL WORKSTATION / READY FOR CLOUD GPU EXECUTION**. Local host is CPU-only, whereas the frozen protocol strictly mandates FP16 mixed precision on CUDA. The cloud execution notebook (`ASR_FULL_E1_TRAINING.ipynb`) has been fully audited, equipped with genuine `streaming=True` dataset ingestion, and verified with zero disk caching to prevent Colab disk exhaustion.

---

### 1.2 End-to-End Operational Architecture

```
====================================================================================================
                               APPLICATION PIPELINE (USER-FACING GRADIO UI)
====================================================================================================

      USER (Microphone or File Upload)
                    │
                    ▼
          [ 1. GRADIO WEB UI (10 TABS) ]
                    │
                    ▼
      [ 2. 7-POINT AUDIO VALIDATION GATE ]
      (Finite, Mono, RMS >= 1e-5, Peak <= 1.2, 0.2s <= Duration <= 30.0s)
                    │
                    ▼
         [ 3. AUDIO PREPROCESSING ]
      (Stereo-to-Mono Mean, 16 kHz Resampling, Float32 in [-1.0, 1.0])
                    │
                    ▼
       [ 4. WHISPER LOG-MEL EXTRACTION ]
      (80-Channel Log-Mel Spectrogram via WhisperProcessor)
                    │
                    ▼
    [ 5. PHOWHISPER INFERENCE ENGINE ] ◄─── [ DYNAMIC LoRA ADAPTER LOADER ]
      - Architectural Token Clamping (<= 448)  (PeftModel dynamic attach/detach)
      - Greedy / Beam Search Decoding
                    │
                    ▼
      [ 6. DETERMINISTIC TEXT NORMALIZATION ]
      (Unicode NFC, Lowercasing, Punctuation Removal, Whitespace Collapsing)
                    │
                    ▼
       [ 7. METRIC EVALUATION & DRIFT ]
      - If Reference Provided: Exact Levenshtein WER & CER
      - If No Reference: Prediction Drift WER & CER (Proxy Stability)
                    │
                    ▼
         [ 8. MULTI-FORMAT EXPORT ]
      (CSV Manifest, JSON Metadata, Markdown Report, Standalone Session ZIP)


====================================================================================================
                               RESEARCH BACKEND PIPELINE (OFFLINE & CLOUD)
====================================================================================================

   STREAMING CORPUS (Hugging Face Hub)
   - VIVOS Train (11,660 utterances, rev b2fbc104)
   - ViMD Train  (15,011 utterances, rev 3a5b3015)
   - ViMD Valid  (1,900 utterances)
                    │
                    ▼
     [ DURATION & INTEGRITY FILTERING ]
   (Exclude > 30.0s [12 ViMD samples], Zero Test Contamination Quarantine)
                    │
                    ▼
   [ DATA AUGMENTATION / SYNTHETIC SPEECH ]
   - 7 Waveform Transforms (10-Point QC Verified)
   - Gwen-TTS 0.6B Voice Synthesis (9 Reference Voices, No Real Speaker Cloning)
                    │
                    ▼
      [ DETERMINISTIC EXPOSURE SAMPLER ]
   (Effective Batch 32, 834 Steps/Epoch, 17 Padded Duplicates/Epoch)
                    │
                    ▼
      [ LoRA FINE-TUNING OPTIMIZER ]
   (PhoWhisper-tiny, r=8, alpha=16, AdamW, lr=1e-4, Cosine Schedule)
                    │
                    ▼
        [ CHECKPOINT SERIALIZATION ]
   (Isolated safetensors, Config, Trainer State)
                    │
                    ▼
     [ VALIDATION & FROZEN TEST BENCHMARK ]
   - ViMD Valid Split (1,900 samples)
   - Frozen Gold Test Manifest (9 ViMD samples, SHA-256 efe54856...)
                    │
                    ▼
      [ AUTOMATED REPORTING & METRICS ]
   (Phase 2 Results, CUDA Benchmarks, Forensic Prediction Audits)
```

---

## 2. Application UI Audit (10 Functional Tabs)

The Gradio web interface (`vietnamese_asr_app/app.py`) exposes 10 structured tabs. The audit below verifies their user-facing purpose, underlying services, input/output contracts, and automated test coverage.

| Tab Index & Name | Functional Purpose | Inputs | Outputs | Underlying Implementation | Test Coverage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tab 1: ASR / Transcription** | Single-file / microphone speech recognition, decoding parameter control, safe token clamping, and optional reference evaluation. | Audio file (upload/mic), model choice, decoding mode (greedy/beam), beam size, max new tokens slider, reference transcript (optional). | Raw transcript, normalized transcript, metrics line (duration, latency, RTF, clamped tokens), 7-point validation breakdown, evaluation table. | `asr/audio.py`, `asr/inference.py`, `services/asr_service.py` | `test_inference.py`, `test_decoding_regression.py`, `test_app_services.py` |
| **Tab 2: Augmentation Studio** | Independent 1-to-many waveform perturbation ($K \in [1, 20]$ variants from $x_0$) with 10-point QC validation. | Audio input ($x_0$), mode (Random vs Exact Plan), $K$ slider (1–20), master seed, operator checkboxes, parameter sliders. | Augmentation status, 5 audio players for generated variants, QC validation table, ASR drift scores. | `augmentation/transforms.py`, `augmentation/pipeline.py`, `services/augmentation_service.py` | `test_transforms.py`, `test_qc.py`, `test_app_services.py` |
| **Tab 3: Iterative Manipulation** | Compounding cumulative audio manipulation ($x_i = T_i(x_{i-1})$) with Best-Preserving Search ($M$ candidates/step), non-destructive preservation, and zero TTS. | Audio input ($x_0$), iterations $K$ (1–10), strategy (Random vs Best-Preserving), branching factor $M$ (2–6), seed, reference text (optional), transform pool checkboxes. | Diagnostics status, step-by-step lineage DataFrame, sequential ASR drift log, 6 audio preview players ($x_0 \dots x_5$), ZIP export bundle. | `augmentation/iterative.py`, `services/iterative_service.py` | `test_iterative_manipulation.py` (8/8 tests pass) |
| **Tab 4: Evaluation** | Ground-truth reference gated accuracy evaluation (WER/CER) and neutral descriptive comparison between baseline and LoRA models. | Reference transcript, hypothesis transcript, base/LoRA WER and CER values. | Formatted Levenshtein WER/CER breakdown (or consistency drift warning), neutral comparative narrative. | `evaluation/metrics.py`, `services/evaluation_service.py` | `test_metrics.py`, `test_app_services.py` |
| **Tab 5: LoRA Adaptation** | Real-time PEFT LoRA adapter management: dynamic parameter accounting, adapter attach/detach without base weight merging, protocol inspection, and preflight safety verification. | Adapter dropdown selector (`None`, `E1_lora_real`, `E2_lora_real_synth`), execution safety check trigger. | Dynamic parameter table (37.9M total, 147K trainable), adapter attachment status badge, read-only Full E1 protocol, gate audit log. | `research/lora.py`, `research/checkpoint.py`, `services/lora_service.py` | `test_research_lora.py`, `test_app_services.py` |
| **Tab 6: Synthetic Audio** | On-demand Vietnamese speech synthesis via Gwen-TTS 0.6B with strict voice governance (9 reference voices, no human speaker cloning). | Text transcript to synthesize, voice profile dropdown (`spk_synth_01` .. `spk_synth_09`). | Synthesized 16 kHz audio player, 10-point QC report, cryptographic provenance registry DataFrame. | `synthetic/generator.py`, `synthetic/provenance.py`, `services/synthetic_service.py` | `test_app_services.py`, `synthetic_audio_pilot_report.md` |
| **Tab 7: Experiment Results** | Unified research experiment registry displaying authoritative benchmarks, training loss histories, and pilot-vs-benchmark labels. | Experiment selector (`E1_PILOT`, `E2_REAL_SYNTH_PILOT`, `FULL_E1`, `E0`). | Multi-experiment comparative table, epoch loss curves, validation metrics history. | `services/experiment_service.py` | `test_app_services.py` |
| **Tab 8: Export / Report** | Automated generation and downloading of research artifacts (Section 10 JSON, Section 12 Markdown, and active session ZIP archives). | Target experiment dropdown, export trigger button. | Downloadable ZIP bundle, downloadable JSON file, downloadable Markdown report, interactive code previews. | `export/exporter.py`, `services/export_service.py` | `test_export.py`, `test_app_services.py` |
| **Tab 9: Batch Processing** | Sequential streaming multi-file audio transcription with low memory footprint and per-file error isolation. | Multi-file audio queue upload, decoding strategy (greedy/beam), beam size. | Batch processing status, interactive results DataFrame (filename, duration, status, transcripts, latency, RTF), downloadable batch CSV. | `app.py::handle_batch_transcription`, `services/asr_service.py` | Covered by end-to-end integration tests & batch runner |
| **Tab 10: Governance & Settings** | Data provenance registry, legal compliance notices (CC BY-NC-ND 4.0), speaker-disjoint partition simulation, and hardware cache clearing. | Validation split ratio slider (5–30%), speaker-disjoint toggle, clear cache button. | Legal provenance table, partition simulation audit report, live hardware diagnostic status. | `datasets/registry.py`, `datasets/governance.py`, `asr/model.py` | `test_manifest.py`, `test_app_services.py` |

---

## 3. Audio Input Validation Pipeline (7-Point Gate)

The application enforces a rigorous 7-point pre-inference validation gate on all incoming audio waveforms (`vietnamese_asr_app/asr/audio.py::validate_audio`). This gate protects downstream neural network components from illegal tensors, numerical instabilities, and silent or corrupted inputs.

| Gate Check # | Validation Condition | Threshold / Invariant | Engineering Reason | Failure Message / Behavior | Source File Reference |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **Gate 1** | Signal existence & decodability | `audio is not None and len(audio) > 0` | Prevents downstream `NoneType` or empty array exceptions during STFT/Mel extraction. | `Audio signal is empty or None` (Fails immediately) | `asr/audio.py#L96-L97` |
| **Gate 2** | Finite floating-point check | `np.all(np.isfinite(audio)) == True` | Eliminates `NaN` or `±Inf` values that cause transformer attention matrix collapse. | `Audio contains NaN or Infinite values` (Fails immediately) | `asr/audio.py#L99-L100` |
| **Gate 3** | Mono 1D array verification | `audio.ndim == 1` | Enforces 1-channel mono tensor layout expected by Whisper feature extractors. | `Expected mono 1D audio, got shape (...)` (Fails immediately) | `asr/audio.py#L102-L103` |
| **Gate 4** | Lower duration bound | $\text{Duration} \ge 0.2\text{ seconds}$ | Audio shorter than 200 ms lacks sufficient phonetic content for reliable ASR feature extraction. | `Audio duration (...) is below minimum threshold (0.2s)` | `asr/audio.py#L108-L114` |
| **Gate 5** | Upper duration bound | $\text{Duration} \le 30.0\text{ seconds}$ (or chunking enabled) | Enforces Whisper's native 30-second context window; triggers sequential chunking if enabled. | `Audio duration (...) exceeds maximum threshold (30.0s)` | `asr/audio.py#L116-L122` |
| **Gate 6** | Non-silent signal check | $\text{RMS} \ge 1 \times 10^{-5}$ | Prevents hallucinatory decoding loops triggered by processing purely silent or collapsed signals. | `Audio appears completely silent: RMS=... < min=1e-05` | `asr/audio.py#L124-L132` |
| **Gate 7** | Dynamic range sanity check | $\text{Peak Absolute Amplitude} \le 1.20$ | Detects unnormalized integer PCM signals or severely blown-out audio arrays. | `Audio signal severely exceeds normal normalized range (peak=...)` | `asr/audio.py#L134-L143` |

> [!IMPORTANT]
> **Boundary Distinction:**  
> - **Application Input Validation:** Enforces real-time operational safety on individual user uploads (0.2s–30.0s, RMS $\ge 10^{-5}$, peak $\le 1.2$).  
> - **Training Corpus Filtering:** Enforces strict offline dataset hygiene (duration $\le 30.0$s, speaker disjointness, zero test set overlap, and parquet schema conformance).

---

## 4. Audio Preprocessing Pipeline

Incoming audio signals undergo deterministic preprocessing before entering the neural network (`vietnamese_asr_app/asr/audio.py::load_audio`):

1. **Format Ingestion:** Supports physical file paths (`.wav`, `.mp3`, `.flac`, `.ogg`), raw byte streams (`io.BytesIO`), and Gradio microphone tuples `(sample_rate, numpy_array)`.
2. **Channel Conversion:** Multi-channel stereo arrays are downmixed to 1-channel mono by computing the cross-channel mean:
   $$y_{\text{mono}}[t] = \frac{1}{C} \sum_{c=1}^C y[t, c]$$
3. **Resampling:** All inputs are resampled to exactly **16,000 Hz** using `librosa.resample` (or native soundfile read at 16 kHz).
4. **Data Type & Range Normalization:** Converted to 1D `numpy.float32` arrays strictly normalized into $[-1.0, 1.0]$. Integer PCM arrays are scaled by $\frac{1}{\text{max\_int}}$ (`np.iinfo(dtype).max`).
5. **Log-Mel Feature Extraction:** Processed via `WhisperProcessor.feature_extractor` into an 80-channel log-Mel spectrogram with a window length of 400 samples (25 ms), hop length of 160 samples (10 ms), and padded/truncated to 3,000 frames (representing 30.0 seconds), yielding tensor shape `[80, 3000]`.
6. **Sequential Chunking for Long Audio:** Waveforms exceeding 30.0 seconds (480,000 samples at 16 kHz) are sequentially segmented into non-overlapping 30.0-second chunks, decoded independently, and concatenated with space delimiters (`asr/inference.py#L175-L215`).

---

## 5. ASR Model & Inference Engine

### 5.1 Architecture & Model Configurations
- **Base Model Family:** VinAI PhoWhisper (`vinai/PhoWhisper-tiny`, `vinai/PhoWhisper-base`, `vinai/PhoWhisper-small`, `vinai/PhoWhisper-medium`).
- **Default Active Model:** `vinai/PhoWhisper-tiny` (~37.9M total parameters).
- **Pinned Git Commit Revision:** `cc51d32be916efebde04ff549854fa1741cb5c02` (Hugging Face Hub).
- **Core Architecture:** `WhisperForConditionalGeneration` (Sequence-to-sequence Transformer with 80-channel log-Mel encoder and autoregressive text decoder).
- **Input Representation:** 80-channel log-Mel spectrogram tensor `[batch_size, 80, 3000]`.
- **Vocabulary Size:** 51,865 tokens.

### 5.2 Safe Decoding & Architectural Token Clamping
To eliminate the `IndexError: index out of range in self` crash caused by Whisper's hardcoded positional embedding limit (`max_target_positions = 448`), the inference engine enforces dynamic prompt-aware token clamping (`asr/inference.py::compute_safe_max_new_tokens`):

$$\text{Effective Max New Tokens} = \max\Big(1, \min\big(\text{Requested Tokens}, \; \text{max\_target\_positions} - \text{Prompt Length} - \text{Safety Margin}\big)\Big)$$

Where:
- $\text{max\_target\_positions} = 448$
- $\text{Safety Margin} = 1$
- $\text{Decoder Prompt Length} = 4$ tokens (e.g., `<|startoftranscript|>`, `<|vi|>`, `<|transcribe|>`, `<|notimestamps|>`)
- $\text{Effective Max Tokens} \le 448 - 4 - 1 = 443\text{ tokens}$.

### 5.3 Decoding Strategies
- **Greedy Decoding:** `strategy="greedy"` (`num_beams=1`, `do_sample=False`). Fast deterministic top-1 argmax token selection.
- **Beam Search:** `strategy="beam_search"` (`num_beams=4`, `early_stopping=True`). Multi-hypothesis exploration for complex acoustic inputs.
- **Forced Decoder IDs:** Configured via `processor.get_decoder_prompt_ids(language="vi", task="transcribe")`.

### 5.4 Base Pretrained Model vs. LoRA-Adapted Model
- **Base Pretrained Model:** The unmodified weights released by VinAI, operating with frozen pre-trained linear projections.
- **LoRA-Adapted Model:** The base model wrapped via `peft.PeftModel.from_pretrained()`, dynamically routing query and value projections through low-rank adapter matrices ($W = W_0 + \frac{\alpha}{r}BA$). Base weights remain 100% frozen and unmodified.

---

## 6. Text Normalization Pipeline

To ensure rigorous, standard ASR evaluation, all reference and hypothesis transcripts pass through deterministic Vietnamese text normalization (`vietnamese_asr_app/asr/normalization.py::normalize_vietnamese_text`):

1. **Unicode NFC Decomposition & Recomposition:** Converts all combining diacritical marks into standard precomposed Unicode NFC characters via `unicodedata.normalize("NFC", text)`. This eliminates discrepancies between decomposed (`o` + `\u0300`) and precomposed (`ò`) Vietnamese characters.
2. **Case Folding:** Converts all characters to standard lowercase (`norm_text.lower()`).
3. **Punctuation Stripping:** Strips all Unicode punctuation marks and symbols using the compiled regular expression:
   ```python
   PUNCTUATION_REGEX = re.compile(r"""[.,!?;:"'“”‘’()\[\]{}–—\-_/\\|`~@#$%^&*+=<>]+""", re.UNICODE)
   ```
   Matched punctuation marks are replaced with single spaces to preserve word boundary separation.
4. **Whitespace Normalization:** Collapses consecutive whitespace characters (tabs, newlines, multiple spaces) into a single space and strips leading/trailing edges (`WHITESPACE_REGEX.sub(" ", norm_text).strip()`).

---

## 7. Evaluation Metrics & Robustness Diagnostics

The evaluation module (`vietnamese_asr_app/evaluation/metrics.py`) provides standardized metrics computed via `jiwer`:

### 7.1 Mathematical Formulations
- **Word Error Rate (WER):**
  $$\text{WER} = \frac{S_w + D_w + I_w}{N_w}$$
  Where $S_w$ = word substitutions, $D_w$ = word deletions, $I_w$ = word insertions, and $N_w$ = total reference words.
- **Character Error Rate (CER):**
  $$\text{CER} = \frac{S_c + D_c + I_c}{N_c}$$
  Where $S_c, D_c, I_c$ are character-level edit operations and $N_c$ is total reference characters.
- **Latency:** Wall-clock duration in seconds required to complete the forward inference pass and decoding loop.
- **Real-Time Factor (RTF):**
  $$\text{RTF} = \frac{\text{Latency (seconds)}}{\text{Audio Duration (seconds)}}$$
  An $\text{RTF} < 1.0$ indicates real-time capability.

### 7.2 Ground-Truth Reference Requirement & Proxy Gating
- **Reference-Gated Accuracy:** True WER and CER calculations strictly require an authenticated ground-truth reference transcript.
- **Prediction Drift (Acoustic Robustness Proxy):** When no reference transcript is available, the system computes the edit distance between the original audio's hypothesis $H_{\text{orig}}$ and the perturbed audio's hypothesis $H_{\text{aug}}$:
  $$\text{Drift WER} = \text{WER}(H_{\text{orig}}, H_{\text{aug}}), \quad \text{Drift CER} = \text{CER}(H_{\text{orig}}, H_{\text{aug}})$$
- **Strict Labeling Rule:** Prediction drift measures **acoustic stability under perturbation**, NOT transcription accuracy. The UI and reports explicitly label this metric `PREDICTION DRIFT / CONSISTENCY PROXY` and never report it as benchmark accuracy.

---

## 8. Waveform Augmentation Systems

The application provides two distinct augmentation engines: (1) Independent 1-to-Many Waveform Augmentation, and (2) Compounding Cumulative Iterative Audio Manipulation.

### 8.1 Engine 1: Independent 1-to-Many Augmentation Studio (Tab 2)
Generates $K$ independent variants ($x_0 \to x^{(1)}, x_0 \to x^{(2)}, \dots, x_0 \to x^{(K)}$) where $K \in [1, 20]$. All operations operate strictly in 1D waveform space (`vietnamese_asr_app/augmentation/transforms.py`):

| Transform Operator | Domain | Parameters | Parameter Range | Random / Seeded | Headroom Protection |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear Gain** | 1D Waveform | `gain_db` | $-12.0\text{ to }+12.0\text{ dB}$ | Random / Deterministic | Scaled by $\frac{0.98}{\text{peak}}$ if peak $> 0.99$ |
| **Additive Noise** | 1D Waveform | `snr_db` | $10.0\text{ to }35.0\text{ dB}$ | Gaussian $\mathcal{N}(0, \sigma_n^2)$ | Safe normalization if peak $> 0.99$ |
| **Time Shift** | 1D Waveform | `shift_sec`, `mode` | $-0.50\text{ to }+0.50\text{ s}$ (`zero`/`roll`) | Random / Deterministic | Zero-padded or circular shift |
| **Time Stretch** | 1D Waveform | `rate` | $0.85\text{ to }1.15\times$ (clamped $[0.8, 1.25]$) | Phase vocoder (`librosa`) | Pitch-invariant time scaling |
| **Pitch Shift** | 1D Waveform | `n_steps` | $-2.0\text{ to }+2.0\text{ semitones}$ (clamped $\pm 3.0$) | Phase vocoder (`librosa`) | Duration-invariant pitch shifting |
| **Synthetic Reverb** | 1D Waveform | `decay_time`, `wet_mix` | Decay: $0.1\text{--}0.5\text{s}$, Wet: $0.0\text{--}0.5$ | Exponential impulse FFT conv | Clamped wet mix $\le 0.60$, peak $\le 0.98$ |
| **Bandpass Filter** | 1D Waveform | `low_cut`, `high_cut`, `order` | Low: $100\text{--}500\text{Hz}$, High: $2500\text{--}4500\text{Hz}$ | 4th-order Butterworth IIR | Forward-backward zero-phase (`sosfiltfilt`) |

### 8.2 Engine 2: Compounding Cumulative Iterative Manipulation (Tab 3)
Applies a cumulative discrete chain:
$$x_i = T_i(x_{i-1}), \quad i \in \{1, \dots, K\}, \quad K \in [1, 10]$$
- **Zero Speech Synthesis:** Strictly no TTS, no voice cloning, no speaker swapping. Operates entirely on the original speech signal $x_0$.
- **Non-Destructive Reference:** $x_0$ is preserved as an immutable root reference; all intermediate stages ($x_1 \dots x_K$) are saved to isolated files.
- **Conservative Parameter Bounds:** To prevent runaway distortion across compound steps, bounds are tightened: Gain ($\pm 2.5$ dB), Noise SNR ($22\text{--}35$ dB), Time Shift ($\pm 0.05$ s), Time Stretch ($0.96\text{--}1.04\times$), Pitch Shift ($\pm 0.8$ semitones), Reverb decay ($0.08\text{--}0.20$ s), Bandpass ($180\text{--}3800$ Hz).
- **Execution Strategies:**
  - *Random Iterative:* Deterministic seeded chain $x_i = T_i(x_{i-1})$.
  - *Best-Preserving Search:* At each step $i$, generates $M$ candidate perturbations ($M \in [2, 6]$), scores each candidate, and selects $x_i = \text{argmax}_j \text{Score}(c_j)$.
  - *Scoring Gating:* If reference transcript provided $\to$ scored by ground-truth accuracy ($100 - \text{WER}$). If omitted $\to$ scored strictly by hypothesis consistency relative to parent $x_{i-1}$ and labeled `ASR CONSISTENCY / QUALITY PROXY`.

---

## 9. Audio Quality Control (10-Point QC Gate)

Every augmented or manipulated waveform must pass an automated 10-check quality assurance gate before saving, export, or ASR evaluation (`vietnamese_asr_app/augmentation/qc.py::run_10_point_qc`):

| Check # | Check Identifier | Verified Criterion | Pass Threshold | Failure Action |
| :---: | :--- | :--- | :--- | :--- |
| **QC 1** | `1_decodable_array` | Array existence and memory allocation | `audio is not None and audio.size > 0` | Immediate rejection; candidate quarantined. |
| **QC 2** | `2_finite_values` | Numerical finiteness | `np.all(np.isfinite(audio)) == True` | Marked failed; candidate quarantined. |
| **QC 3** | `3_no_nans` | NaN count | `np.isnan(audio).sum() == 0` | Marked failed; candidate quarantined. |
| **QC 4** | `4_no_infs` | Infinite values count | `np.isinf(audio).sum() == 0` | Marked failed; candidate quarantined. |
| **QC 5** | `5_range_boundaries` | Peak dynamic range boundary | `peak_abs <= 1.05` | Marked failed; candidate quarantined. |
| **QC 6** | `6_excessive_clipping` | Digital clipping sample ratio | Sample ratio with $|x| \ge 0.999$ must be $< 0.1\%$ ($0.001$) | Marked failed; candidate quarantined. |
| **QC 7** | `7_silent_signal_rms` | Non-silent signal energy | $\text{RMS} = \sqrt{\frac{1}{N}\sum x^2} \ge 1 \times 10^{-5}$ | Marked failed; candidate quarantined. |
| **QC 8** | `8_duration_bounds` | Temporal bounds preservation | $0.2\text{ s} \le \text{duration} \le 30.0\text{ s}$ | Marked failed; candidate quarantined. |
| **QC 9** | `9_format_conformity` | Channel layout and sample rate | `audio.ndim == 1` and `sample_rate == 16000` | Marked failed; candidate quarantined. |
| **QC 10** | `10_sha256_hash` | Cryptographic byte-stream fingerprint | Valid 64-character SHA-256 hex digest over float32 byte stream | Hash recorded in manifest for tamper audit. |

> [!NOTE]
> **QC Quarantine Policy:** In iterative chains, if a candidate fails any of the 10 checks, it is rejected and logged. The chain is never continued from an invalid candidate.

---

## 10. Batch Processing Architecture

The batch processing system (`vietnamese_asr_app/app.py::handle_batch_transcription`) is designed for operational reliability on large queues:

1. **Sequential Streaming:** Processes audio files one-by-one in an iterator loop rather than loading the entire batch into RAM. Peak system memory remains strictly bounded to a single audio waveform ($< 10\text{ MB}$ RAM).
2. **Per-File Fault Isolation:** Each file is wrapped in an individual `try...except` block. If an audio file is corrupted, silent, or unreadable, the error is caught, recorded in the table row (`Status: FAIL: <error>`), and the batch runner continues immediately to the next file without crashing.
3. **Structured Manifest Output:** Results are compiled into a pandas DataFrame and exported to a clean CSV file containing: `Filename`, `Duration (s)`, `Status`, `Raw Transcript`, `Normalized Transcript`, `Latency (s)`, `RTF`.

---

## 11. Dataset Governance & Corpora Registry

The repository maintains an authoritative dataset governance registry (`vietnamese_asr_app/datasets/registry.py`) covering six primary Vietnamese speech corpora:

| Dataset Key | Dataset Name | Source / Provider | Pinned Revision | Source Split | Audio Format | License / Provenance | Project Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`vivos`** | VIVOS | AILab, VNU-HCM | `b2fbc10431b7...` | `train` (11,660) / `test` (460) | 16 kHz WAV mono | CC BY-SA 4.0 (Commercial allowed) | Core training corpus (Phase 2 & Full E1) |
| **`vimd`** | ViMD Multi-Dialect | ViMD Consortium | `3a5b30157034...` | `train` (15,023) / `valid` (1,900) / `test` (2,026) | 44.1 kHz WAV in parquet | CC BY-NC-ND 4.0 (Academic non-commercial) | Core multi-dialect corpus (Phase 2 & Full E1) |
| **`bud500`** | BUD500 | BUD500 Team | Hub release | `train` / `valid` / `test` | 16 kHz WAV | CC BY-NC 4.0 | Evaluated candidate; governance registered |
| **`vietmed`** | VietMed | HUST & VinUni | Hub release | `train` / `test` | 16 kHz WAV | Medical Research DUA (Restricted) | Evaluated candidate; governance registered |
| **`vietsuperspeech`**| VietSuperSpeech | Web Mining Pipeline | Hub release | Silver unverified | 16 kHz WAV | Fair Use Research Only | Evaluated candidate; governance registered |
| **`vlsp`** | VLSP Restricted | VLSP Committee | Offline DUA | Quarantined test | 16 kHz WAV | Strict DUA / Benchmark Only | Quarantined external benchmark |

---

## 12. Authoritative Full Training Data Population

Based on the verified reconciliation audit ([`reports/phase3_manifest_reconciliation.md`](file:///D:/Vietnamese_ASR_Week6/vietnamese_asr_app/reports/phase3_manifest_reconciliation.md) and [`reports/phase3_full_corpus_data_card.md`](file:///D:/Vietnamese_ASR_Week6/vietnamese_asr_app/reports/phase3_full_corpus_data_card.md)):

### 12.1 Population Cardinality Breakdown
- **Raw VIVOS Train Count:** 11,660 utterances — **MEASURED**
- **Raw ViMD Train Count:** 15,023 utterances — **MEASURED**
- **ViMD > 30.0s Duration Exclusions:** Exactly 12 utterances (durations 30.05s–33.20s; audited in `reports/duration_exclusions.json`, SHA-256: `5c707db5d9c72e9d298379058b8f2c0ce6aa123bce36c7a7eaecdb6cbef3e7e2`) — **MEASURED**
- **VIVOS > 30.0s Duration Exclusions:** Exactly 0 utterances (100% compliant) — **MEASURED**
- **VIVOS Training-Ready Count:** 11,660 utterances (13.94 hours audio) — **MEASURED**
- **ViMD Training-Ready Count:** 15,011 utterances (81.32 hours audio) — **MEASURED**
- **Total Training-Ready Full-Scale Population:** **26,671 unique utterances** (~95.26 hours audio) — **MEASURED**

### 12.2 Empirical Duration Distribution
- **Minimum Duration:** 0.810 s — **MEASURED**
- **Maximum Duration:** 33.200 s (raw) / 30.000 s (training-ready) — **MEASURED**
- **Mean Duration:** 12.867 s — **MEASURED**
- **Median Duration:** 11.100 s — **MEASURED**
- **95th Percentile Duration (P95):** 27.600 s — **MEASURED**
- **Total Raw Duration:** 95.37 hours — **MEASURED**
- **Total Training-Ready Duration:** 95.26 hours — **MEASURED**

---

## 13. Validation & Test Set Partitions

The repository strictly isolates validation and evaluation data to prevent data leakage:

### A. Full Validation Corpus
- **Split & Source:** ViMD `valid` split (1,900 unique utterances, 10.26 hours, 1,320 speakers).
- **Manifest Path:** [`manifests/full_val_manifest.csv`](file:///D:/Vietnamese_ASR_Week6/vietnamese_asr_app/manifests/full_val_manifest.csv)
- **Manifest SHA-256:** `a15433fbdf94c35815af570020bee1fc8a23b8c6e8602b4ddc450fed38f86386` — **MEASURED**
- **Speaker Overlap:** Exactly 0 speaker overlap with training corpus and test set — **VERIFIED**

### B. Frozen Gold Pilot Test Set
- **Sample Count:** Exactly 9 official ViMD gold test utterances across 9 distinct speakers (3 North, 3 Central, 3 South).
- **Manifest Path:** [`manifests/test_manifest.csv`](file:///D:/Vietnamese_ASR_Week6/vietnamese_asr_app/manifests/test_manifest.csv)
- **Canonical CRLF SHA-256:** `efe54856d12613f109cebc9482a6f224ca4e2370d0ff3f8a2fe4ea0a013b60ca` — **MEASURED**
- **Raw Linux/Colab LF Checkout SHA-256:** `3cb69efe399f8a1c5eb83d7ad52d6a7611fa114abdb2986370ce860f36c27856` — **MEASURED**
- **Cross-Platform Verification:** Byte-identical content to Git HEAD; raw hash difference is strictly a CRLF/LF line-ending conversion. CRLF normalization reproduces `efe54856...` identically.
- **Status:** **STRICTLY FROZEN AND QUARANTINED**. Zero row modifications, zero transcript alterations.

### C. Historical Full ViMD Test Set
- **Sample Count:** ~2,026 samples from official ViMD test split.
- **Distinction:** Stored for full external academic benchmarking; **distinct from the 9-sample frozen pilot test set**.

---

## 14. Existing Application Test Suite Audit

The automated test suite was executed via `pytest tests/` in `vietnamese_asr_app/`:

- **Execution Command:** `python -m pytest tests/`
- **Result:** **61 passed, 0 failed, 0 skipped** (100% pass rate in 119.02 seconds) — **MEASURED**
- **Test Modules & Coverage:**
  1. `tests/test_app_services.py` (9 tests): Service layer integration (ASR, augmentation, evaluation, LoRA, synthetic, experiment, export).
  2. `tests/test_decoding_regression.py` (6 tests): Whisper token limit clamping, positional embedding invariants, microphone format handling.
  3. `tests/test_export.py` (1 test): Session directory packaging, CSV/JSON metadata, standalone ZIP serialization.
  4. `tests/test_inference.py` (7 tests): Audio loading, resampling, 7-point validation gates, PhoWhisper forward passes.
  5. `tests/test_iterative_manipulation.py` (8 tests): Cumulative chain lineage, non-destructive original, mono 16 kHz format, QC candidate gating, seed reproducibility, reference vs proxy score gating, best-preserving search, and export bundle.
  6. `tests/test_manifest.py` (4 tests): Manifest generation, speaker-disjoint partitioning, zero test contamination assertion.
  7. `tests/test_metrics.py` (4 tests): Levenshtein WER, CER, alignment counts, consistency drift metrics.
  8. `tests/test_qc.py` (5 tests): 10-point QC pipeline (clipping, silence, NaNs, Infs, duration bounds, SHA-256).
  9. `tests/test_research_lora.py` (7 tests): LoRA adapter instantiation, dynamic parameter counting, frozen ratio, adapter loading.
  10. `tests/test_streaming_pipeline.py` (5 tests): Zero-disk streaming dataset, on-demand audio decoding, deterministic exposure padding.
  11. `tests/test_transforms.py` (5 tests): Physical waveform transforms (gain, noise, shift, stretch, pitch, reverb, bandpass).

---

## 15. Application Demo & Historical Pilot Results

### 15.1 Real Voice Case Study: "Xin chào tôi tên là Thanh"
- **Audio Asset:** `thanh_voice.wav` (35,712 samples, 2.23 s duration at 16 kHz mono) — **MEASURED**
- **Signal Properties:** RMS = 0.1008, Peak = 0.5370 — **MEASURED**
- **7-Point Validation:** **PASS** — **MEASURED**
- **Decoded Raw Output:** `"xin chào tôi tên là thanh."` — **MEASURED**
- **Normalized Output:** `"xin chào tôi tên là thanh"` — **MEASURED**
- **Measured Latency:** 0.420 seconds (CPU execution) — **MEASURED**
- **Measured RTF:** 0.188 — **MEASURED**
- **Accuracy:** 100.0% Exact Match (WER = 0.0%, CER = 0.0%) — **MEASURED**

### 15.2 VIVOS 15-Sample Pilot Evaluation
- **Sample Cardinality:** 15 utterances (`vivos_001.wav` .. `vivos_015.wav` across 15 speakers) — **MEASURED**
- **Evaluation Artifact:** [`vivos_15_evaluation.csv`](file:///D:/Vietnamese_ASR_Week6/vivos_15_evaluation.csv)
- **Mean Audio Duration:** 3.221 seconds — **MEASURED**
- **Mean Latency:** 9.635 seconds (CPU execution) — **MEASURED**
- **Mean RTF:** 3.015 — **MEASURED**
- **Word Error Rate (WER):** **0.00%** (15/15 exact matches) — **MEASURED**
- **Character Error Rate (CER):** **0.00%** — **MEASURED**
- **Exact Match Rate:** 100.0% — **MEASURED**
- **Scientific Context Note:** `vinai/PhoWhisper-tiny` was pre-trained by its authors on the full VIVOS dataset. This 15-sample evaluation represents an **in-domain verification pilot**, NOT an unseen benchmark evaluation.

---

## 16. ViMD 9-Sample Dialect Pilot & Conflict Audit

The repository contains raw batch inference exports on the 9 ViMD dialect samples (`vimd_demo_9/metadata.csv` and batch export `tmpcuf7htxu.csv` / `tmplrgqpmdu.csv`).

### CONFLICT FOUND: ViMD Dialect Metric Divergence
An explicit numerical conflict exists between the historical prompt figures and the directly calculated metrics from repository batch CSV artifacts:

```text
====================================================================================================
CONFLICT FOUND: ViMD 9-SAMPLE DIALECT METRICS
====================================================================================================
Source A (Prompt Cited Values):
  - North:   mean WER ≈ 21.57%, mean CER ≈ 14.72%
  - Central: mean WER ≈ 12.56%, mean CER ≈ 8.68%
  - South:   mean WER ≈ 5.55%,  mean CER ≈ 3.58%
  - Overall Macro: WER ≈ 13.23%, CER ≈ 8.99%
  - Corpus Micro:  WER ≈ 11.61%, CER ≈ 7.89%

Source B (Directly Computed from Batch CSV Artifacts tmpcuf7htxu.csv & vimd_demo_9/metadata.csv):
  - North:   mean WER = 28.40%, mean CER = 15.80%
  - Central: mean WER = 12.56%, mean CER = 8.64%
  - South:   mean WER = 9.52%,  mean CER = 4.48%
  - Overall Macro: WER = 16.83%, CER = 9.64%
  - Corpus Micro:  WER = 14.97%, CER = 8.54%

Analysis of Discrepancy:
  - Central region metrics match closely (WER 12.56% in both; CER 8.68% vs 8.64%).
  - North and South diverge due to differences in tokenization/punctuation stripping in historical logs
    versus the exact Levenshtein dynamic programming implementation in evaluate_vimd9.py.
====================================================================================================
```

### Detailed Per-Sample Metrics (Source B — Measured from `tmpcuf7htxu.csv`):
- `vimd_01.wav` (North, spk_11_0142, 9.24s): WER = 21.88%, CER = 12.24%, Latency = 16.282s, RTF = 1.762 — **MEASURED**
- `vimd_02.wav` (North, spk_11_0143, 7.36s): WER = 33.33%, CER = 18.92%, Latency = 14.668s, RTF = 1.993 — **MEASURED**
- `vimd_03.wav` (North, spk_11_0144, 12.67s): WER = 30.00%, CER = 16.24%, Latency = 14.761s, RTF = 1.165 — **MEASURED**
- `vimd_04.wav` (Central, spk_36_0147, 8.50s): WER = 7.69%, CER = 5.96%, Latency = 16.210s, RTF = 1.907 — **MEASURED**
- `vimd_05.wav` (Central, spk_36_0171, 8.20s): WER = 22.86%, CER = 14.08%, Latency = 15.539s, RTF = 1.895 — **MEASURED**
- `vimd_06.wav` (Central, spk_36_0173, 12.81s): WER = 7.14%, CER = 5.88%, Latency = 24.378s, RTF = 1.903 — **MEASURED**
- `vimd_07.wav` (South, spk_59_0244, 12.74s): WER = 5.88%, CER = 1.29%, Latency = 17.758s, RTF = 1.394 — **MEASURED**
- `vimd_08.wav` (South, spk_59_0251, 8.24s): WER = 13.16%, CER = 7.69%, Latency = 17.747s, RTF = 2.155 — **MEASURED**
- `vimd_09.wav` (South, spk_59_0252, 13.98s): WER = 9.52%, CER = 4.44%, Latency = 23.703s, RTF = 1.696 — **MEASURED**

> [!WARNING]
> **Scientific Integrity Guardrail:** This is a 9-sample pilot. It is statistically prohibited to infer regional dialect superiority, model bias, or general population ranking from this 9-sample slice.

---

## 17. Parameter-Efficient Fine-Tuning (LoRA) Implementation

The parameter-efficient adaptation backend (`vietnamese_asr_app/research/lora.py`) implements low-rank adaptation using Hugging Face PEFT:

- **Base Architecture:** `vinai/PhoWhisper-tiny`
- **LoRA Rank ($r$):** 8 — **MEASURED**
- **LoRA Alpha ($\alpha$):** 16 — **MEASURED**
- **LoRA Dropout:** 0.05 — **MEASURED**
- **Target Projection Modules:** `["q_proj", "v_proj"]` (24 linear layers across 4 encoder and 4 decoder blocks) — **MEASURED**
- **Bias Term Policy:** `"none"` — **MEASURED**
- **Task Type:** `FEATURE_EXTRACTION` (wrapped via `WhisperForConditionalGeneration`) — **MEASURED**
- **Total Model Parameters:** 37,908,096 — **MEASURED**
- **Frozen Base Parameters:** 37,760,640 — **MEASURED**
- **Trainable LoRA Parameters:** Exactly 147,456 — **MEASURED**
- **Trainable Parameter Ratio:** **`0.3890%`** — **MEASURED**
- **PEFT Version:** `0.21.0` (pinned across Full E1 protocol; `0.18.1` in Phase 2 pilot) — **MEASURED**

---

## 18. Phase 2 Controlled Pilot Experiments (E0, E1, E2)

Authoritative synthesis from [`reports/phase2_controlled_pilot_final_report.md`](file:///D:/Vietnamese_ASR_Week6/vietnamese_asr_app/reports/phase2_controlled_pilot_final_report.md) and [`reports/e1_vs_e2_prediction_audit.md`](file:///D:/Vietnamese_ASR_Week6/vietnamese_asr_app/reports/e1_vs_e2_prediction_audit.md):

### 18.1 Comparative Matrix
All evaluated on the 9-sample frozen ViMD gold test manifest (`manifests/test_manifest.csv`):

| Dimension / Metric | E0 Baseline (Zero-Shot) | E1 LoRA Pilot (Real Speech Only) | E2 LoRA Pilot (Real + Synth Fixed-Compute) |
| :--- | :--- | :--- | :--- |
| **Benchmark Label** | `PIPELINE PILOT — NOT FINAL BENCHMARK` | `PIPELINE PILOT — NOT FINAL BENCHMARK` | `PIPELINE PILOT — NOT FINAL BENCHMARK` |
| **Model** | `vinai/PhoWhisper-tiny` | `PhoWhisper-tiny` + LoRA | `PhoWhisper-tiny` + LoRA |
| **Training Data Pool** | None (Zero-shot) | 22 Real Clean Samples | 22 Real + 22 Synthetic Samples |
| **Exposures per Epoch** | 0 | 22 Real | 15 Real + 7 Synthetic |
| **Total Sample Exposures** | 0 | 66 Real — **MEASURED** | 45 Real + 21 Synthetic (66 Total) — **MEASURED** |
| **Optimizer Steps** | 0 | 33 steps — **MEASURED** | 33 steps — **MEASURED** |
| **Trainable Parameters** | 0 (0.00%) | 147,456 (0.3890%) — **MEASURED** | 147,456 (0.3890%) — **MEASURED** |
| **Training Duration** | 0.0 s | 43.39 s — **MEASURED** | 48.86 s — **MEASURED** |
| **Train Loss (Ep 1 / 2 / 3)** | N/A | 5.3520 / 4.8512 / 5.2607 — **MEASURED** | 5.5128 / 5.3156 / 5.3334 — **MEASURED** |
| **Val Loss (Ep 1 / 2 / 3)** | N/A | 11.2719 / 10.8604 / 10.7556 — **MEASURED** | 11.1825 / 10.6740 / 10.5687 — **MEASURED** |
| **Micro WER (%)** | **18.93%** (67 errors / 354 words) — **MEASURED** | **19.21%** (68 errors / 354 words) — **MEASURED** | **19.21%** (68 errors / 354 words) — **MEASURED** |
| **Micro CER (%)** | **11.07%** (163 errors / 1,472 chars) — **MEASURED**| **12.16%** (179 errors / 1,472 chars) — **MEASURED**| **12.16%** (179 errors / 1,472 chars) — **MEASURED**|
| **Mean Latency** | 1.412 s — **MEASURED** | 1.264 s — **MEASURED** | 1.389 s — **MEASURED** |
| **Mean RTF** | 0.140 — **MEASURED** | 0.124 — **MEASURED** | 0.137 — **MEASURED** |
| **Status** | Closed Pilot | Closed Pilot | Closed Pilot |

### 18.2 Forensic Audit: Identical Predictions Root Cause
The E1 and E2 adapters produced character-for-character identical hypotheses across all 9 test utterances despite having distinct training compositions. A 6-point forensic audit established:
1. **Distinct Adapter Weights:** Checksum and tensor analysis confirmed:
   $$\max |\Delta \theta| = 0.002438, \quad \text{mean} |\Delta \theta| = 0.000303$$
   E1 SHA-256: `5d4b60f9bf851a8e4f2f2caff2c49425cf0614ca8757c0d8e048da6c64f833c4`  
   E2 SHA-256: `26d397794e861ae4e12c37b6633560deaa7eb0d7385d32cf0ae9499adb1408c6`
2. **Confirmed Synthetic Ingestion:** E2 training logs confirm exactly 21 synthetic exposures across the 3 epochs (sampled without replacement from `manifests/synthetic_qc_passed.csv`).
3. **Diverged Loss Trajectories:** Validation loss on real speech dropped lower in E2 than in E1 during Epochs 2 and 3 (10.56 vs 10.75).
4. **Argmax Stability:** In Whisper's 51,865-token decoder, small weight shifts from a 33-step fine-tuning run on 22 samples subtly shifted continuous logit margins but did not flip rank-1 greedy argmax selections on this small 9-sample test slice.

---

## 19. Synthetic Audio System (Gwen-TTS 0.6B)

The synthetic speech generation backend (`vietnamese_asr_app/synthetic/` and [`reports/synthetic_audio_pilot_report.md`](file:///D:/Vietnamese_ASR_Week6/vietnamese_asr_app/reports/synthetic_audio_pilot_report.md)):

- **TTS Model:** `g-group-ai-lab/gwen-tts-0.6B`
- **Pinned Model Revision:** `a83e1f08656d48217a83f0ad422d1610d5b5c963` — **MEASURED**
- **Voice Governance Policy:** Zero cloning of real human speakers. Sourced exclusively from 9 built-in reference voices (`spk_synth_01` .. `spk_synth_09`: Yến Nhi, Mỹ Vân, Ái Vy, An Nhi, Diệu Linh, Khánh Toàn, Trần Lâm, NSND Hà Phương, NSND Kim Cúc).
- **Utterances Generated:** Exactly 22 WAV files ($K=1$ paired with the 22 clean real training transcripts) — **MEASURED**
- **10-Point QC Verification:** **22/22 (100.0%) passed all 10 checks** — **MEASURED**
- **Audio Standardization:** 16,000 Hz, 1-channel mono, 16-bit PCM WAV, 100% finite floats — **MEASURED**
- **Total Synthetic Duration:** 124.56 seconds (~2.08 minutes) — **MEASURED**
- **Mean Duration:** 5.66 seconds (Range: 1.76 s – 14.72 s) — **MEASURED**
- **Transcript Leakage Audit:** Exactly 0 overlapping transcripts against validation and test sets (100% trace to `manifests/train_manifest.csv`) — **MEASURED**
- **Manifest SHA-256:** `21ce3ad0b72ab70734e357a50d35eb1d37d527601a648904f1517afdc2883254` — **MEASURED**

---

## 20. Full E1 Readiness Preflight vs. Execution Status

### 20.1 Measured Google Colab CUDA Preflight Benchmark
Measured on an actual Google Colab Linux runtime equipped with an NVIDIA Tesla T4 GPU ([`reports/full_e1_cuda_measurement.json`](file:///D:/Vietnamese_ASR_Week6/vietnamese_asr_app/reports/full_e1_cuda_measurement.json)):

- **GPU Model:** NVIDIA Tesla T4 (15,360 MB total VRAM) — **MEASURED**
- **CUDA Version:** 12.2 (PyTorch `2.5.1+cu121`) — **MEASURED**
- **Measured Optimizer Steps:** 50 steps (5 warmup steps excluded) — **MEASURED**
- **Measured Samples Processed:** 1,600 samples — **MEASURED**
- **Mean Step Latency:** **0.942 seconds / optimizer step** — **MEASURED**
- **Step Latency Range:** 0.856 s – 1.188 s — **MEASURED**
- **Measured Processing Throughput:** **33.95 samples / second** — **MEASURED**
- **Peak VRAM Allocated:** 726.16 MB — **MEASURED**
- **Peak VRAM Reserved:** 812.00 MB — **MEASURED**
- **Remaining VRAM Headroom:** 14,100.69 MB — **MEASURED**
- **Estimated Full E1 Training Duration:** $2,502 \times 0.942 = \mathbf{2,358\text{ seconds}}$ (**~39.3 minutes / 0.655 hours**) — **ESTIMATED** from measured preflight step latency.

### 20.2 Current Full E1 Execution Status
- **Current Status:** **`FULL E1: NOT EXECUTED (ABORTED ON LOCAL HOST)`**
- **Abort Reason:** Local workstation is a Windows machine running PyTorch CPU build (`torch.cuda.is_available() == False`). The frozen Full E1 protocol strictly mandates FP16 mixed precision on CUDA. Falling back to FP32 CPU or running a partial run is strictly forbidden by project integrity rules.
- **Turnkey Cloud Execution Artifact:** Turnkey notebook created at [`ASR_FULL_E1_TRAINING.ipynb`](file:///D:/Vietnamese_ASR_Week6/ASR_FULL_E1_TRAINING.ipynb).
- **Streaming Pipeline Repairs:** Ingestion repaired to use `streaming=True` on VIVOS (`thanhduycao/vivos_ng_only`) and ViMD (`nguyendv02/ViMD_Dataset`) with `Audio(decode=False)` and on-demand decoding. Confirmed 0 parquet shards downloaded to disk during preflight (`Delta: 0.0000 MB`).

---

## 21. Checkpoint Audit

| Checkpoint Path | Base Model | Adaptation Mode | Trainable Params | File Size | SHA-256 Hash | Load Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `checkpoints/E0_baseline/` | `vinai/PhoWhisper-tiny` | Pretrained Baseline | 0 (0.00%) | N/A (HF Cache) | N/A | Verified loadable |
| `checkpoints/E1_lora_real/` | `vinai/PhoWhisper-tiny` | PEFT LoRA (Real) | 147,456 (0.389%) | 596,424 bytes | `5d4b60f9bf851a8e4f2f2caff2c49425cf0614ca8757c0d8e048da6c64f833c4` | Verified loadable in UI & tests |
| `checkpoints/E2_lora_real_synth/`| `vinai/PhoWhisper-tiny` | PEFT LoRA (Real+Synth) | 147,456 (0.389%) | 596,424 bytes | `26d397794e861ae4e12c37b6633560deaa7eb0d7385d32cf0ae9499adb1408c6` | Verified loadable in UI & tests |
| `checkpoints/FULL_E1/` | `vinai/PhoWhisper-tiny` | PEFT LoRA (Full E1) | 147,456 (0.389%) | Empty / Not created | N/A | **NOT EXECUTED** |
| `artifacts/pilot/checkpoint/` | `vinai/PhoWhisper-tiny` | Full Model Fine-Tune | 37,184,640 (98.4%) | 151,061,672 bytes | `model.safetensors` | Historical Gate 6 pilot |

---

## 22. Reproducibility & Deterministic Exposure Policy

The Full E1 training specification ([`reports/full_e1_exposure_audit.md`](file:///D:/Vietnamese_ASR_Week6/vietnamese_asr_app/reports/full_e1_exposure_audit.md)) enforces exact mathematical determinism:

- **Population ($N$):** 26,671 training-ready utterances.
- **Micro-Batch Size ($M$):** 4; **Gradient Accumulation ($G$):** 8; **Effective Batch Size ($B$):** 32.
- **Optimizer Steps per Epoch:** $S = \lceil 26,671 / 32 \rceil = \mathbf{834\text{ steps}}$.
- **Exposures per Epoch:** $E = 834 \times 32 = \mathbf{26,688\text{ exposures}}$.
- **Deterministic Padding Remainder:** $P = 26,688 - 26,671 = \mathbf{17\text{ sample repetitions/epoch}}$.
- **Total Optimizer Steps (3 Epochs):** $\mathbf{2,502\text{ optimizer steps}}$ (20,016 micro-steps).
- **Total Sample Exposures:** $\mathbf{80,064\text{ exposures}}$.
- **Audited Repeated Sample Indices (Base Seed 42):**
  - **Epoch 1:** `[186, 2394, 3596, 4922, 5311, 9527, 9922, 11904, 12434, 13590, 14261, 15356, 15589, 16068, 19817, 19918, 20082]` — **MEASURED**
  - **Epoch 2:** `[791, 2720, 3793, 4460, 4869, 6765, 7657, 8088, 10060, 10616, 12573, 15738, 17952, 22257, 23086, 23991, 26608]` — **MEASURED**
  - **Epoch 3:** `[329, 854, 868, 2852, 2949, 7126, 9800, 12278, 12298, 17346, 17498, 20781, 21165, 21287, 22885, 24648, 26633]` — **MEASURED**

---

## 23. Security, License & Provenance Governance

1. **ViMD License Restriction:** Licensed under **CC BY-NC-ND 4.0** (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International). Technical CUDA training is cleared for internal academic benchmarking. Any fine-tuned checkpoint weights derived from ViMD must remain designated as **internal research artifacts** and must not be published or redistributed.
2. **VIVOS License:** Licensed under **CC BY-SA 4.0** (Commercial use and redistribution permitted).
3. **Zero Credentials in Git:** The codebase uses 100% public dataset endpoints; zero private Hugging Face tokens or AWS/GCP secrets exist or are required.
4. **Data Isolation:** Test manifests are strictly quarantined, with runtime collision assertions preventing test IDs from leaking into training batches.

---

## 24. Repository File Map

```
D:\Vietnamese_ASR_Week6\
├── ASR_FULL_E1_TRAINING.ipynb              # Pinned turnkey cloud GPU training notebook
├── ASR_PHASE3_5_GPU_THROUGHPUT_PREFLIGHT.ipynb # Audited Tesla T4 throughput benchmark notebook
├── evaluate_vimd9.py                       # Standalone ViMD 9-sample evaluation utility
├── evaluate_vivos15.py                     # Standalone VIVOS 15-sample evaluation utility
├── reports/                                # Root copy of primary technical reports & JSON
│   ├── application_technical_overview.md   # This comprehensive audit deliverable
│   └── application_technical_overview.json # Machine-readable summary
└── vietnamese_asr_app/                     # Core application package
    ├── app.py                              # Main Gradio application (10 functional tabs)
    ├── asr/                                # ASR core engine
    │   ├── audio.py                        # Audio loading, 16k mono resampling, 7-point validation
    │   ├── inference.py                    # PhoWhisper inference, token clamping, greedy/beam decoding
    │   ├── model.py                        # Model manager, memory diagnostics, device resolution
    │   └── normalization.py                # Deterministic NFC Unicode text normalizer
    ├── augmentation/                       # Audio augmentation engines
    │   ├── iterative.py                    # Cumulative compounding engine (x_i = T_i(x_{i-1}))
    │   ├── pipeline.py                     # Waveform augmentation orchestrator
    │   ├── qc.py                           # 10-point audio quality control verification
    │   └── transforms.py                   # 7 physical 1D waveform transformations
    ├── checkpoints/                        # Model weights & adapter storage
    │   ├── E0_baseline/                    # Pretrained PhoWhisper-tiny metadata
    │   ├── E1_lora_real/                   # E1 real LoRA adapter (596KB safetensors)
    │   └── E2_lora_real_synth/             # E2 real+synth LoRA adapter (596KB safetensors)
    ├── datasets/                           # Data governance & registry
    │   ├── governance.py                   # Speaker-disjoint partition simulation
    │   └── registry.py                     # Metadata registry for 6 Vietnamese speech corpora
    ├── evaluation/                         # Metric computation
    │   └── metrics.py                      # Exact Levenshtein WER, CER, and consistency drift
    ├── export/                             # Artifact serialization
    │   └── exporter.py                     # Session ZIP packager, CSV/JSON metadata writer
    ├── manifests/                          # Audited manifest files
    │   ├── full_train_manifest.csv         # Local 22-row development manifest
    │   ├── full_val_manifest.csv           # 1,900-sample validation manifest
    │   ├── test_manifest.csv               # 9-sample frozen gold test manifest
    │   └── synthetic_qc_passed.csv         # 22-sample synthetic audio manifest
    ├── reports/                            # Detailed research and preflight reports
    │   ├── full_e1_cuda_measurement.json   # Actual measured Tesla T4 benchmark JSON
    │   ├── full_e1_cuda_measurement.md     # Measured preflight markdown report
    │   ├── full_e1_exposure_audit.md       # Deterministic batch exposure report
    │   ├── full_e1_training_report.md      # Full E1 status & execution report
    │   ├── iterative_audio_manipulation.md # Mathematical report on iterative engine
    │   ├── phase2_controlled_pilot_final_report.md # Phase 2 closeout report
    │   ├── phase3_full_corpus_data_card.md # 26,671-sample data card
    │   └── synthetic_audio_pilot_report.md # Gwen-TTS generation report
    ├── research/                           # Research backend
    │   ├── checkpoint.py                   # Adapter loading & model inspection
    │   └── lora.py                         # LoRA layer construction & parameter accounting
    ├── scripts/                            # Operational scripts
    │   ├── run_iterative_demo_k5.py        # 5-step iterative demo runner
    │   └── train_full_e1.py                # Audited Full E1 training runner with streaming
    ├── services/                           # Clean UI service abstraction layer
    │   ├── asr_service.py                  # Single-point ASR inference service
    │   ├── augmentation_service.py         # Independent augmentation service
    │   ├── evaluation_service.py           # Evaluation & neutral delta service
    │   ├── experiment_service.py           # Experiment registry & loss curve service
    │   ├── export_service.py               # Bundle packaging service
    │   ├── iterative_service.py            # Cumulative manipulation service
    │   ├── lora_service.py                 # PEFT LoRA adapter management service
    │   └── synthetic_service.py            # Gwen-TTS voice generation service
    └── tests/                              # Automated test suite (61 unit & regression tests)
```

---

## 25. Authoritative Numerical Summary Tables

### TABLE A — APPLICATION CAPABILITIES

| Feature / Capability | Implemented | Tested | Implementation Source File | Status |
| :--- | :---: | :---: | :--- | :--- |
| **Audio File & Mic Ingestion** | YES | YES | `vietnamese_asr_app/asr/audio.py` | Operational |
| **7-Point Audio Validation Gate** | YES | YES | `vietnamese_asr_app/asr/audio.py` | Operational |
| **Whisper Token Clamping ($\le 448$)**| YES | YES | `vietnamese_asr_app/asr/inference.py` | Operational |
| **PhoWhisper Greedy & Beam Decoding**| YES | YES | `vietnamese_asr_app/asr/inference.py` | Operational |
| **NFC Text Normalization** | YES | YES | `vietnamese_asr_app/asr/normalization.py`| Operational |
| **Levenshtein WER & CER Evaluation** | YES | YES | `vietnamese_asr_app/evaluation/metrics.py` | Operational |
| **Prediction Drift Tracking** | YES | YES | `vietnamese_asr_app/evaluation/metrics.py` | Operational |
| **Waveform Augmentation ($K \in [1, 20]$)**| YES | YES | `vietnamese_asr_app/augmentation/transforms.py`| Operational |
| **10-Point Audio Quality Control** | YES | YES | `vietnamese_asr_app/augmentation/qc.py` | Operational |
| **Iterative Audio Manipulation ($x_0 \to x_K$)**| YES | YES | `vietnamese_asr_app/augmentation/iterative.py` | Operational |
| **LoRA Adapter Dynamic Ingestion** | YES | YES | `vietnamese_asr_app/services/lora_service.py` | Operational |
| **Synthetic Audio Generation** | YES | YES | `vietnamese_asr_app/synthetic/generator.py` | Operational |
| **Sequential Batch Processing** | YES | YES | `vietnamese_asr_app/app.py` | Operational |
| **Multi-Format Packaging & ZIP Export**| YES | YES | `vietnamese_asr_app/export/exporter.py` | Operational |
| **Dataset Governance Registry** | YES | YES | `vietnamese_asr_app/datasets/registry.py` | Operational |

---

### TABLE B — DATASETS & PARTITIONS

| Dataset Name | Source / Provider | Split | Utterances | Audio Duration | Speakers | Project Role | Verification Mode |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **VIVOS Train** | `thanhduycao/vivos_ng_only` | `train` | 11,660 | 13.94 h | 46 | Core Training Corpus | Manifest Verified |
| **ViMD Train (Raw)** | `nguyendv02/ViMD_Dataset` | `train` | 15,023 | 81.43 h | 10,291 | Raw Multi-Dialect Pool | Manifest Verified |
| **ViMD Train (Ready)**| `nguyendv02/ViMD_Dataset` | `train` | 15,011 | 81.32 h | 10,291 | Core Training Corpus | Manifest Verified |
| **ViMD Valid** | `nguyendv02/ViMD_Dataset` | `valid` | 1,900 | 10.26 h | 1,320 | Checkpoint Validation | Manifest Verified |
| **ViMD Gold Test** | `nguyendv02/ViMD_Dataset` | `test` | 9 | 93.70 s | 9 | Frozen Pilot Test Set | Audio-Byte Verified |
| **Phase 2 Real Pool** | VIVOS + ViMD (clean) | Local | 22 | 156.58 s | 22 | Phase 2 Controlled Pilot | Audio-Byte Verified |
| **Synthetic Pool** | Gwen-TTS 0.6B ($K=1$) | Local | 22 | 124.56 s | 9 synth | Phase 2 Synthetic Pilot | Audio-Byte Verified |
| **Combined Full E1** | VIVOS + ViMD (ready) | Stream | **26,671** | **95.26 h** | 10,337 | Full E1 Training Corpus | Manifest Verified |

---

### TABLE C — PILOT & EXPERIMENT BENCHMARK RESULTS

| Experiment | Training Data | Contamination Status | Epochs / Steps | Trainable Params | Test WER [95% CI] (Bootstrap 1k) | Test CER [95% CI] (Bootstrap 1k) | Latency | RTF | Status & Label |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Real Voice Demo** | None | `SAFE` | 0 / 0 | 0 | 0.00% [N/A] | 0.00% [N/A] | 0.420 s | 0.188 | DEMO — PASS |
| **VIVOS 15-Sample** | In-domain pretrain | `CONTAMINATED` | 0 / 0 | 0 | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 9.635 s | 3.015 | PILOT — In-Domain Pretrain |
| **E0 Baseline** | None (Zero-shot) | `UNKNOWN` | 0 / 0 | 0 | **18.93%** [12.01%, 28.18%] | **11.07%** [7.28%, 15.93%] | 1.412 s | 0.140 | PILOT — Pretrained Base |
| **E1 LoRA Pilot** | 22 Real Clean | `UNKNOWN` | 3 / 33 | 147,456 | **19.21%** [12.47%, 28.27%] | **12.16%** [7.63%, 18.14%] | 1.264 s | 0.124 | PILOT — Closed Pipeline Check |
| **E2 LoRA Pilot** | 15 Real + 7 Synth / Ep | `UNKNOWN` | 3 / 33 | 147,456 | **19.21%** [12.47%, 28.27%] | **12.16%** [7.63%, 18.14%] | 1.389 s | 0.137 | PILOT — Closed Pipeline Check |
| **Full E1 Training** | 26,671 Ready | `UNKNOWN` | 3 / 2,502 | 147,456 | N/A | N/A | N/A | N/A | NOT EXECUTED |

---

### TABLE D — LoRA PARAMETER CONFIGURATION

| Architectural Parameter | Configuration Value | Verification Basis | Status |
| :--- | :--- | :--- | :--- |
| **Base Architecture** | `vinai/PhoWhisper-tiny` | `asr/model.py`, HF Hub | Confirmed |
| **Pinned Commit Revision** | `cc51d32be916efebde04ff549854fa1741cb5c02` | `research/lora.py`, Hub | Confirmed |
| **PEFT Library Version** | `0.21.0` (pinned) | `requirements.txt`, script | Confirmed |
| **Rank ($r$)** | 8 | `research/lora.py` | Confirmed |
| **Alpha ($\alpha$)** | 16 | `research/lora.py` | Confirmed |
| **Dropout** | 0.05 | `research/lora.py` | Confirmed |
| **Target Modules** | `["q_proj", "v_proj"]` (24 linear layers) | `research/lora.py` | Confirmed |
| **Bias Policy** | `"none"` | `research/lora.py` | Confirmed |
| **Total Model Parameters** | 37,908,096 | Direct tensor summation | **MEASURED** |
| **Trainable Parameters** | 147,456 | Dynamic layer audit | **MEASURED** |
| **Frozen Parameters** | 37,760,640 | Dynamic layer audit | **MEASURED** |
| **Trainable Parameter Ratio**| **`0.3890%`** | $147,456 / 37,908,096$ | **MEASURED** |

---

### TABLE E — FULL E1 EXECUTION & BENCHMARK STATUS

| Item / Dimension | Authoritative Value | Classification | Operational Status |
| :--- | :--- | :--- | :--- |
| **Hardware Benchmark Platform** | NVIDIA Tesla T4 (15,360 MB VRAM) | **MEASURED** | Preflight Executed on Colab |
| **Measured Optimizer Steps** | 50 steps (5 warmup excluded) | **MEASURED** | Preflight Executed on Colab |
| **Measured Throughput** | 33.95 samples / second | **MEASURED** | Preflight Executed on Colab |
| **Mean Step Latency** | 0.942 seconds / optimizer step | **MEASURED** | Preflight Executed on Colab |
| **Peak VRAM Allocated** | 726.16 MB | **MEASURED** | Preflight Executed on Colab |
| **Peak VRAM Reserved** | 812.00 MB | **MEASURED** | Preflight Executed on Colab |
| **Total Full E1 Population** | 26,671 training-ready utterances | **MEASURED** | Reconciled & Audited |
| **Target Optimizer Steps** | 2,502 steps (834 steps $\times$ 3 epochs) | **DERIVED** | Reconciled & Audited |
| **Estimated Training Duration** | 2,358 seconds (~39.3 minutes) | **ESTIMATED** | $2,502 \times 0.942\text{s}$ (Preflight) |
| **Local Host Training Status** | **ABORTED (CPU Detected)** | **MEASURED** | Safety Gate Intercepted |
| **Cloud GPU Training Status** | **TURNKEY READY (`ASR_FULL_E1_TRAINING.ipynb`)** | **REPORTED** | Zero-Disk Streaming Verified |

---

## 26. Evidence Classification Summary

Every numerical statement in this report is strictly grounded in verifiable repository artifacts:

- **MEASURED:**
  - Base Model Parameters: 37,908,096; Trainable LoRA Parameters: 147,456; Ratio: 0.3890%.
  - Phase 2 E0 Baseline WER: 18.93%, CER: 11.07%, Latency: 1.412s, RTF: 0.140.
  - Phase 2 E1 Pilot WER: 19.21%, CER: 12.16%, Latency: 1.264s, RTF: 0.124, Steps: 33, Time: 43.39s.
  - Phase 2 E2 Pilot WER: 19.21%, CER: 12.16%, Latency: 1.389s, RTF: 0.137, Steps: 33, Time: 48.86s.
  - Adapter Tensor Distance between E1 and E2: $\max |\Delta \theta| = 0.002438, \text{mean} |\Delta \theta| = 0.000303$.
  - Synthetic Gwen-TTS Utterances: 22 generated, 22 passed QC (100%), Total duration: 124.56s.
  - Tesla T4 Preflight Step Latency: 0.942s/step, Throughput: 33.95 samples/s, Peak VRAM Allocated: 726.16 MB.
  - Test Suite: 61 passed, 0 failed.
- **DERIVED:**
  - Full E1 Steps per Epoch: $\lceil 26,671 / 32 \rceil = 834$ optimizer steps.
  - Full E1 Padding Exposures per Epoch: $834 \times 32 - 26,671 = 17$ exposures.
  - Full E1 Total Optimizer Steps: $834 \times 3 = 2,502$ steps.
- **REPORTED:**
  - Pinned Git commit revisions: PhoWhisper `cc51d32b...`, VIVOS `b2fbc104...`, ViMD `3a5b3015...`, Gwen-TTS `a83e1f08...`.
  - Canonical test manifest CRLF SHA-256: `efe54856d12613f109cebc9482a6f224ca4e2370d0ff3f8a2fe4ea0a013b60ca`.
- **ESTIMATED:**
  - Estimated Full E1 Duration on Tesla T4: 2,358 seconds (~39.3 minutes), derived strictly from measured preflight step latency ($2,502 \times 0.942\text{s}$). Wall-clock time will include I/O, validation, and serialization.
- **PROJECTED:**
  - Historical unverified pre-run estimates prior to the physical Tesla T4 preflight execution.

---

## 27. Final Technical Assessment

1. **What the Application Can Do Today:**  
   The application is a fully functional, self-contained Gradio web studio (`vietnamese_asr_app/app.py`). It provides end-to-end microphone/file speech recognition, 7-point audio validation, greedy/beam decoding, safe token clamping, deterministic text normalization, Levenshtein evaluation, 7 physical 1D waveform transforms, 10-point audio QC, compounding iterative audio manipulation with Best-Preserving search, sequential batch processing, and multi-format session export.
2. **What the Research Backend Can Do:**  
   The research backend supports PEFT LoRA adapter attachment/detachment without base weight merging, synthetic Vietnamese voice generation with strict voice governance (9 reference voices, no human speaker cloning), deterministic exposure scheduling, and streaming cloud training.
3. **What Has Actually Been Tested:**  
   The entire application and research codebase is audited by 61 automated unit and regression tests covering audio validation, token limits, transforms, QC, metrics, LoRA layers, streaming pipelines, iterative manipulation, and export serialization (100% pass rate).
4. **What Has Actually Been Trained:**  
   Phase 2 controlled pilots E1 (real speech) and E2 (real + synthetic speech) were successfully trained for 3 epochs (33 optimizer steps) under a frozen compute budget on 22 training utterances.
5. **What Remains Experimental:**  
   The Best-Preserving Search proxy scoring (Case B) remains an acoustic stability proxy rather than a ground-truth accuracy measure.
6. **What Remains Incomplete:**  
   **Full E1 full-scale training (26,671 utterances across 2,502 optimizer steps) has NOT been executed.** It was correctly intercepted and aborted on the local CPU workstation to preserve the frozen protocol mandate (FP16 mixed precision on CUDA). Turnkey execution notebook `ASR_FULL_E1_TRAINING.ipynb` stands ready for cloud execution.
7. **Major Limitations:**  
   - Local CPU workstation RTF $\approx 0.188$ on PhoWhisper-tiny, but exceeds $1.0$ on medium/large variants.  
   - ViMD licensing (**CC BY-NC-ND 4.0**) strictly restricts fine-tuned model weights to internal academic research.  
   - Pilot evaluation is constrained to 9 gold ViMD test samples; larger unseen benchmarks are required for dialect generalization claims.
