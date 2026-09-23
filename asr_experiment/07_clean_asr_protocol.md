# 07 — Clean ASR Protocol (Gate 4 — PROVISIONAL)

> STATUS: PROVISIONAL — pending small-sample verification results
> Do NOT run test set through model during development
> Do NOT tune on test data

---

## Task Definition

- **Task**: Vietnamese Automatic Speech Recognition (ASR)
- **Goal**: Establish a clean, reproducible baseline with limited compute
- **Metric**: Word Error Rate (WER), Character Error Rate (CER)
- **Constraint**: No full training in this phase — protocol freeze only

---

## Selected Training Configuration: CONFIG B (General + Dialect)

### Training Datasets

| Dataset | Hours | Utterances | Transcript | Role |
|---------|-------|------------|-----------|------|
| VIVOS train | ~14h | ~11,660 | Gold (human, read) | Baseline read speech |
| ViMD train | 81.43h | 15,023 | Gold (human, spontaneous) | Dialect diversity |
| **Total gold** | **~95h** | **~26,683** | | |

### Validation Dataset

| Dataset | Hours | Utterances | Transcript |
|---------|-------|------------|-----------|
| ViMD valid | 10.26h | 1,900 | Gold (human) |

> VIVOS has no official validation split. Carving one from train would reduce training data.
> ViMD valid is the official paper-defined validation split with speaker-disjoint property.

### Primary Test Candidate

| Dataset | Hours | Utterances | Transcript | Contamination |
|---------|-------|------------|-----------|--------------|
| ViMD test | 10.87h | 2,026 | Gold (human) | NO_EVIDENCE_FOUND for all examined models |

> ViMD test is the ONLY candidate with NO_EVIDENCE_FOUND across PhoWhisper, Whisper, MMS, Zipformer, wav2vec2 variants.

### Secondary Test Candidate

| Dataset | Hours | Utterances | Transcript | Contamination |
|---------|-------|------------|-----------|--------------|
| VIVOS test | ~0.75h | 760 | Gold (human) | CONFIRMED_CONTAMINATED for PhoWhisper, Zipformer |

> VIVOS test is useful ONLY as a reference for comparing against published results.
> It is NOT a valid generalization test for PhoWhisper or Zipformer.

---

## Excluded Datasets

| Dataset | Reason |
|---------|--------|
| FOSD | No official splits; MP3 format requires conversion; weak speaker metadata; adds preprocessing complexity without clear benefit over ViMD |
| VietSuperSpeech | VERSION_UNRESOLVED; pseudo-labeled (silver); no speaker IDs; circular contamination with Zipformer |
| VietMed labeled | Medical domain only; small (16h); adds domain mismatch |
| VLSP2021 | Access not obtained (registration required) |

---

## Transcript Policy

- **Gold transcripts ONLY** in training and evaluation
- **No pseudo-labeled data** in frozen protocol
- VietSuperSpeech (silver) excluded until version resolved
- Transcript normalization:
  - Lowercase all text
  - Remove punctuation (.,!?;:—""''…)
  - Preserve Vietnamese diacritics (tones + vowel marks)
  - Normalize whitespace (single space between words)
  - VIVOS transcripts are UPPERCASE — must lowercase

---

## Speaker Split Policy

- **ViMD**: Official paper-defined train/valid/test splits are speaker-disjoint by design
  - Train: 10,291 speakers, Valid: 1,320 speakers, Test: 1,344 speakers
  - Province metadata ensures geographic diversity
- **VIVOS**: Official train/test are speaker-disjoint (46 train / 19 test speakers)
- **Cross-dataset**: No speaker overlap between VIVOS and ViMD (different projects, sources, time periods)

---

## Source Split Policy

- **ViMD**: Source = 63 provincial broadcasting stations (news programs)
- **VIVOS**: Source = AILAB VNUHCM recordings
- No source overlap between datasets

---

## Audio Preprocessing

| Parameter | Value |
|-----------|-------|
| Sample rate | 16,000 Hz |
| Channels | Mono |
| Format | WAV PCM 16-bit or float32 |
| Maximum duration | 30 seconds |
| Minimum duration | 0.5 seconds |
| Silence trimming | Optional (VAD-based) |
| Normalization | Peak normalization to -1.0 dB |

> VIVOS is confirmed 16kHz WAV
> ViMD sample rate: VERIFIED 44,100 Hz native; resampled to 16,000 Hz via librosa.resample
> ViMD audio files named `{province_code}_{sequence}.wav` — WAV embedded in parquet
> ViMD transcript field: `text` (Vietnamese with diacritics, lowercase)
> ViMD schema verified: `audio`, `filename`, `gender`, `province_code`, `province_name`, `region`, `speakerID`, `text`

### License Constraints
- **VIVOS**: CC BY-NC-SA 4.0 (NonCommercial, ShareAlike)
- **ViMD**: CC BY-NC-ND 4.0 (NonCommercial, **NoDerivatives**)
  - NoDerivatives: fine-tuning models on this data and distributing the resulting model may constitute a "derivative work" — must verify license interpretation for research use
  - For internal research/evaluation: generally considered permitted
  - For distributing fine-tuned model weights: legal review recommended

---

## Evaluation Protocol

| Metric | Definition |
|--------|-----------|
| WER | Word Error Rate (standard: substitutions + insertions + deletions / reference words) |
| CER | Character Error Rate (same formula at character level) |
| Decoding | Greedy decoding (no beam search for baseline) |
| Seed | 42 |

### Evaluation Rules
1. Test set is FROZEN — no model selection or hyperparameter tuning on test data
2. Validation set (ViMD valid) used for early stopping and model selection
3. Report WER and CER on both primary test (ViMD test) and secondary reference (VIVOS test)
4. Report contamination status alongside all results

---

## Model Candidates

### Primary: PhoWhisper-tiny (vinai/PhoWhisper-tiny)
- Architecture: Whisper-tiny fine-tuned for Vietnamese
- Parameters: ~39M
- Training data: 843.79h (includes VIVOS + VLSP2020 — noted as contamination)
- License: Apache 2.0
- Compute: Lowest among PhoWhisper variants
- **Contamination**: CONFIRMED for VIVOS, NO_EVIDENCE_FOUND for ViMD

### Fallback: PhoWhisper-base (vinai/PhoWhisper-base)
- Architecture: Whisper-base fine-tuned for Vietnamese
- Parameters: ~74M
- Same training data as tiny
- Use if tiny produces unusable results

### Baseline Reference: Whisper-small multilingual (openai/whisper-small)
- Zero-shot Vietnamese capability
- No documented Vietnamese-specific fine-tuning
- **Contamination**: NO_EVIDENCE_FOUND for all candidates

---

## Contamination Disclosure

| Model | VIVOS test | ViMD test |
|-------|-----------|----------|
| PhoWhisper-tiny | CONFIRMED_CONTAMINATED (trained on VIVOS) | NO_EVIDENCE_FOUND |
| PhoWhisper-base | CONFIRMED_CONTAMINATED (trained on VIVOS) | NO_EVIDENCE_FOUND |
| Whisper-small | NO_EVIDENCE_FOUND | NO_EVIDENCE_FOUND |

---

## Download / Compute Budget

| Item | Size | Notes |
|------|------|-------|
| VIVOS (full) | ~1.5 GB | 16kHz WAV, via Zenodo |
| ViMD (full parquet) | ~74 GB | Parquet with embedded audio; or HF streaming |
| PhoWhisper-tiny | ~150 MB | Model weights |
| PhoWhisper-base | ~290 MB | Model weights |
| Whisper-small | ~460 MB | Model weights |
| **Total minimum** | **~76 GB** | **ViMD dominates** |

### Streaming Alternative
- ViMD can be loaded via HuggingFace streaming (no full download needed)
- This reduces immediate disk requirement to ~1.5 GB (VIVOS only) + model weights
- Streaming adds training latency but avoids 74 GB download

### Approximate Training Compute
- ~95h of labeled audio
- ~26,683 utterances
- PhoWhisper-tiny fine-tuning: ~2-4 GPU hours on a single consumer GPU (estimated)
- No GPU needed until protocol is frozen and approved

---

## Pre-Download Checklist (Pilot Verification Status)
 
- [x] Verify ViMD sample rate from streaming (VERIFIED in Gate 5: 44,100 Hz, requires resampling)
- [x] Verify ViMD audio quality on 5 random samples (VERIFIED in Gate 4 & Gate 5)
- [x] Verify VIVOS transcript format (UPPERCASE → lowercase normalization, VERIFIED in Gate 6)
- [x] Verify ViMD transcript format (casing, diacritics, VERIFIED in Gate 6)
- [x] Test ViMD streaming pipeline (VERIFIED in Gate 6)
- [x] Validate preprocessing pipeline on 100 train / 20 val samples (VERIFIED in Gate 6)
- [x] Execute 1-epoch engineering training & verify checkpoint (VERIFIED in Gate 6)
- [ ] Large-scale training run (pending user approval; NOT executed in pilot)
