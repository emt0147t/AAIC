# 11 — Final Experimental ASR Protocol

> **PROTOCOL STATUS**: **PROVISIONAL (ENGINEERING-VERIFIED)**  
> This protocol has been verified through a 1-epoch CPU pilot (Gate 6). It remains labeled **PROVISIONAL** rather than fully frozen due to:
> 1. ViMD license constraint (`CC BY-NC-ND 4.0` NoDerivatives clause regarding fine-tuned model redistribution).
> 2. Full-dataset download bandwidth vs streaming architecture selection.
>
> Neither test set has been touched for tuning or checkpoint selection.

---

## 1. Master Dataset Audit & Status Summary

All metrics below reflect authoritative, verified sources.

### GOLD / HUMAN-VERIFIED TRANSCRIPTS

| Dataset | Role | Hours | Transcript Type | Speaker Metadata | Split Structure | Access Status | Key Notes |
|:---|:---|:---:|:---|:---|:---|:---|:---|
| **VIVOS** | Train baseline + Reference test | 15.67h | Gold (scripted read) | 65 speakers (46 train / 19 test) | Official train (11,660) / test (760) | Globally Public (Zenodo 7068130) | 16 kHz WAV; CC BY-NC-SA 4.0; Test set contaminated for PhoWhisper & Zipformer |
| **ViMD** | Train dialect + Valid + Primary test | 102.56h | Gold (provincial news) | 12,955 speakers across 63 provinces | Paper train (15,023) / val (1,900) / test (2,026) | Globally Public (`nguyendv02/ViMD_Dataset`) | Native 44.1 kHz WAV (must resample to 16 kHz); CC BY-NC-ND 4.0; Uncontaminated (NEF) across all models |
| **FOSD** | Excluded | ~30.0h | Gold (read speech) | Inconsistent / limited | None (single pool of 25,921 recordings) | Public Original (`Mendeley k9sxg2twv4/4`) | MP3 format (requires conversion); FPT Public License (not CC BY 4.0); excluded due to lack of official splits |
| **VietMed (labeled)** | Excluded | ~16.0h | Gold (medical doctor-patient) | Limited | Paper-defined train / val / test | Public Original (`leduckhai/VietMed`) | OGG format; medical domain mismatch; contaminated for Zipformer |
| **VLSP 2021 (transcribed)** | Excluded | ~280.0h | Gold (competition speech) | Unknown | Train / dev / eval | Registration Required (vlsp.org.vn) | Access pending registration; competition agreement required |

### SILVER / PSEUDO-LABELED DATA

| Dataset | Role | Hours | Transcript Type | Speaker Metadata | Split Structure | Access Status | Key Notes |
|:---|:---|:---:|:---|:---|:---|:---|:---|
| **VietSuperSpeech** | Excluded | 103h–300h+ | Silver (Zipformer pseudo-labels) | None | Train / dev (no test) | Globally Public (`thanhnew2001/VietSuperSpeech`) | VERSION_UNRESOLVED (32k vs 52k vs 67k); circular evaluation with Zipformer; excluded from protocol |

### UNLABELED DATA POOLS

| Dataset | Role | Hours | Transcript Type | Speaker Metadata | Access Status | Key Notes |
|:---|:---|:---:|:---|:---|:---|:---|
| **VLSP 2021 (untranscribed)** | Standby (SSL) | ~400.0h | None | Unknown | Registration Required | Intended for self-supervised pretraining |
| **VietMed (unlabeled)** | Standby (SSL) | ~2,200.0h | None | Unknown | Public Original | 1,000h medical + 1,200h general domain |

---

## 2. Experimental Pipeline Protocol Specification

### 1. Task Definition
- **Task**: Vietnamese Automatic Speech Recognition (ASR)
- **Objective**: Acoustic adaptation and fine-tuning on diverse Vietnamese read and provincial dialect speech
- **Target Language**: Vietnamese (`vi`)
- **Primary Metric**: Word Error Rate (WER)
- **Secondary Metric**: Character Error Rate (CER)

### 2. Dataset Composition (Provisional Mixture: Config B)
- **Total Labeled Data**: ~95.37 hours of Gold-standard human transcripts.
- **Composition**:
  - Academic scripted read speech: VIVOS train (~13.94h audio, 11,660 utterances).
  - Multi-dialect spontaneous broadcast speech: ViMD train (81.43h audio, 15,023 utterances).
- **Proportion**: ~15% read speech baseline / ~85% provincial multi-dialect spontaneous speech.

### 3. Training Split
- **VIVOS train**: 11,660 utterances, 46 disjoint speakers.
- **ViMD train**: 15,023 utterances, 10,291 disjoint speakers across 63 provinces.
- **Total Training Set**: 26,683 utterances.

### 4. Validation Split
- **ViMD valid**: 1,900 utterances (10.26 hours), 1,320 disjoint speakers.
- **Role**: Hyperparameter selection, learning rate scheduling, early stopping checkpoint selection.
- **Note**: VIVOS has no official validation set; carving from VIVOS train is avoided to preserve training volume.

### 5. Primary Test Benchmark
- **ViMD test**: 2,026 utterances (10.87 hours), 1,344 disjoint speakers across 63 provinces.
- **Contamination Status**: **NO_EVIDENCE_FOUND** across all examined model candidates (PhoWhisper, Whisper, MMS, Zipformer).
- **Benchmark Rule**: FROZEN. Untouched during development, pilot runs, and hyperparameter tuning.

### 6. Secondary Reference Test Set
- **VIVOS test**: 760 utterances (0.75 hours), 19 disjoint speakers.
- **Contamination Status**: **CONFIRMED_CONTAMINATED** for PhoWhisper-tiny, PhoWhisper-base, and Zipformer.
- **Benchmark Rule**: REFERENCE ONLY. Used solely to establish an in-domain performance ceiling against published literature; **never** cited as an out-of-domain generalization metric.

### 7. Speaker and Source Disjointness
- **ViMD Splits**: Strictly speaker-disjoint by design (train: 10,291 spk; valid: 1,320 spk; test: 1,344 spk).
- **VIVOS Splits**: Strictly speaker-disjoint (46 train spk vs 19 test spk).
- **Cross-Dataset Leakage**: Zero speaker or recording overlap between VIVOS (AILAB university lab) and ViMD (provincial broadcast stations).

### 8. Transcript Preprocessing Policy
- **Raw Storage**: Source transcripts (`sentence_raw`) are preserved unaltered in all manifests.
- **Deterministic Normalization (`sentence_norm`)**:
  1. Unicode NFC normalization.
  2. Lowercase all text.
  3. Strip all punctuation marks (`.,!?;:"'()[]-` etc.) via standard regex.
  4. Preserve all Vietnamese diacritics and tone markers (`à, á, ả, ã, ạ, ă, â, đ, ê, ô, ơ, ư, etc.`).
  5. Collapse multiple whitespaces into a single space; trim outer padding.
- **VIVOS Special Case**: VIVOS source text is 100% UPPERCASE; normalized deterministically via lowercasing.

### 9. Audio Preprocessing Pipeline
- **Channels**: Single-channel mono. Multi-channel audio is averaged across the channel axis.
- **Target Sample Rate**: **16,000 Hz** (16 kHz) mono PCM float32, normalized to $[-1.0, 1.0]$.
- **Resampling Policy**:
  - VIVOS native audio (16,000 Hz): No resampling applied.
  - ViMD native audio (44,100 Hz): Resampled via `librosa.resample` ($44.1\text{ kHz} \rightarrow 16.0\text{ kHz}$).
- **Feature Extraction**: 80-channel log-mel spectrogram calculated by WhisperProcessor (window length 400, hop length 160).
- **Tensor Input Shape**: `[80 mel bins, 3000 frames]` (padded/truncated to 30.0 seconds maximum duration).

### 10. Model Specification
- **Primary Model**: `vinai/PhoWhisper-tiny` (37.8M parameters; 37.2M trainable).
- **Fallback Model**: `vinai/PhoWhisper-base` (74.0M parameters).
- **Architecture**: Sequence-to-sequence Transformer encoder-decoder with specialized Vietnamese byte-pair tokenizer.

### 11. Processor and Tokenizer Configuration
- **Processor**: `WhisperProcessor.from_pretrained("vinai/PhoWhisper-tiny")`.
- **Target Tokens**: Vietnamese text (`language="vi"`, `task="transcribe"`).
- **Label Masking**: Padding tokens in decoder target sequences are set to `-100` to be ignored by cross-entropy loss computation.

### 12. Optimization Protocol
- **Optimizer**: AdamW.
- **Learning Rate**: $1 \times 10^{-4}$ for PhoWhisper-tiny with linear decay schedule.
- **Weight Decay**: 0.01.
- **Gradient Clipping**: Maximum gradient norm 1.0.
- **Batching**: Micro-batch size 8–16 (depending on VRAM) with gradient accumulation to achieve effective batch size 32.
- **Precision**: fp16 on CUDA GPU; fp32 on CPU.

### 13. Evaluation Metrics
- **Word Error Rate (WER)**: Standard Levenshtein distance on whitespace-separated words:
  $$\text{WER} = \frac{S + D + I}{N_{\text{ref}}}$$
- **Character Error Rate (CER)**: Levenshtein distance on characters excluding whitespace.
- **Evaluation Normalization**: Identical deterministic normalization applied to both reference and hypothesis strings prior to Levenshtein calculation.

### 14. Model Contamination Policy
- Contamination status must be stated alongside every reported metric:
  - ViMD test: `NO_EVIDENCE_FOUND` (genuine generalization test).
  - VIVOS test: `CONFIRMED_CONTAMINATED` (reference ceiling).

### 15. Compute & Budget Constraints
- Full training requires ~95 hours of audio ingestion (~26,683 utterances).
- Estimated full fine-tuning compute: ~2–4 GPU hours on a single modern GPU (e.g. RTX 3090 / A100).
- No GPU training permitted without explicit authorization.

### 16. Pilot Verification Scope
- Verified in Gate 6: 100 utterances (50 VIVOS + 50 ViMD train), 20 ViMD valid utterances, 1 epoch CPU run in 68.26 seconds.
- Confirmed finite loss ($1.15 \rightarrow 0.37$), checkpoint serialization/deserialization, and qualitative Vietnamese inference.
