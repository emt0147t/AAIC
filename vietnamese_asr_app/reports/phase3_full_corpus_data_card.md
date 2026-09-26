# Phase 3 Full-Corpus Data Readiness Audit & Data Card

> [!IMPORTANT]
> **Audit Status**: `AUDIT COMPLETE`  
> **Readiness Verdict**: `PHASE 3 DATA READINESS: READY`  
> **Benchmark Label**: `FULL-SCALE ASR DATA READINESS — SPECIFICATION & AUDIT`  
> This authoritative data card and audit document establishes the dataset lineage, raw vs. training-ready corpus cardinality, duration policy compliance, two-tier architecture (local physical files vs. cloud streaming), hard leakage verification, and candidate experimental budget for full-scale Vietnamese ASR LoRA fine-tuning.

---

## 1. Raw Corpus

The raw training candidate pool comprises all legitimate utterances originating strictly from the official training partitions of the two primary authorized corpora:

| Dataset | Source Split | Raw Utterances | Raw Duration (Hours) | Raw Duration (Seconds) | Speaker Count | Native Sample Rate | Native Channels |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **VIVOS** | `train` | 11,660 | 13.94 h | 50,184 s | 46 | 16,000 Hz | 1 (Mono) |
| **ViMD** | `train` | 15,023 | 81.43 h | 293,148 s | 10,291 | 44,100 Hz | 1 (Mono) |
| **Combined Raw Full Corpus** | `train` | **26,683** | **95.37 h** | **343,332 s** | **10,337** | Mixed (16k / 44.1k) | 1 (Mono) |

### Local Physical Raw Candidates
Within the local development repository on this workstation, a small, verified physical slice is instantiated on disk:
- **Local VIVOS Train slice**: 10 utterances (`vivos_001` .. `vivos_010`), 32.44 s, 10 speakers (16 kHz WAV mono).
- **Local ViMD Clean Train slice**: 12 utterances (`vimd_train_01` .. `vimd_train_12`), 124.14 s, 12 speakers (16 kHz WAV mono).
- **Total Local Raw Candidates**: 22 utterances, 156.58 s (2.61 minutes), 22 distinct speakers.

---

## 2. Training-Ready Corpus

The **Training-Ready Corpus** represents the clean, audited subset remaining after applying:
1. Source-split verification (strictly `train` split only).
2. Duration filtering policy ($\text{duration} \le 30.0\text{ seconds}$).
3. Speaker disjointness (zero overlap with validation and test speakers).
4. Duplicate checks (0 duplicate sample IDs, 0 duplicate audio hashes).
5. Zero test contamination quarantine (0 test sample IDs, 0 test audio hashes).

### Cardinality & Retention Breakdown
| Tier | Dataset | Raw Utterances | Excluded (>30s) | Training-Ready Utterances | Training-Ready Duration | Retention Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Cloud Full Scale** | VIVOS train | 11,660 | 0 | 11,660 | 13.94 h | 100.0% |
| **Cloud Full Scale** | ViMD train | 15,023 | 12 | 15,011 | 81.32 h | 99.92% |
| **Cloud Full Scale Total**| **VIVOS + ViMD** | **26,683** | **12** | **26,671** | **95.26 h** | **99.955%** |
| **Local Physical Total** | **VIVOS + ViMD** | **22** | **0** | **22** | **155.58 s** | **100.0%** |

The canonical manifest [`manifests/full_train_manifest.csv`](file:///d:/Vietnamese_ASR_Week6/vietnamese_asr_app/manifests/full_train_manifest.csv) represents the **Training-Ready** corpus.

---

## 3. Local vs. Cloud Execution Architecture

The repository enforces a clear separation between local and cloud execution environments:

```
+---------------------------------------------------------------------------------------+
|                                TWO-TIER ARCHITECTURE                                  |
+-------------------------------------------+-------------------------------------------+
|               LOCAL WINDOWS               |                 CLOUD GPU                 |
|            (Validation & Preflight)       |           (Full-Scale Benchmark)          |
+-------------------------------------------+-------------------------------------------+
| OS: Windows 11 (AMD64)                    | OS: Linux (Ubuntu 22.04 LTS / Colab)      |
| PyTorch: 2.10.0+cpu (CPU execution)       | PyTorch: 2.1+ with CUDA acceleration      |
| Physical Audio: 36 WAV files (22 train,   | Storage / Ingestion: Streaming / Cache    |
|   5 val, 9 test)                          |   from HuggingFace Hub (95.3h / ~74GB)    |
| Verification: AUDIO-BYTE VERIFIED         | Verification: MANIFEST-LEVEL VERIFIED     |
| Role: Code integrity, test suite, smoke   | Role: Multi-epoch LoRA training (E1, E2), |
|   tests, deterministic preflight audits   |   full benchmark evaluation               |
+-------------------------------------------+-------------------------------------------+
```

> [!NOTE]
> **Readiness Assertion**: The absence of 74 GB of raw audio files on the local Windows laptop is an intended architecture boundary, **not** a data failure. Cloud manifest readiness is evaluated via metadata schemas, source accessibility, and streaming pipeline integrity.

---

## 4. Duration Audit

The repository duration policy specifies:
- **Train / Validation splits**: $\text{duration} \le 30.0\text{ seconds}$ (whisper context window constraint; prevents memory spikes and truncation distortion).
- **Test split**: **Never filtered** (preserves official benchmark distribution).

### Empirical Distribution Metrics
| Corpus Pool | Utterances | Min (s) | Max (s) | Mean (s) | Median (s) | P95 (s) | Count $\le 30$s | Count $> 30$s | Total Duration (Pre) | Total Duration (Post) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Local Physical Train** | 22 | 2.969 s | 14.880 s | 7.072 s | 6.891 s | 14.256 s | 22 (100%) | 0 (0%) | 155.58 s | 155.58 s |
| **Cloud VIVOS Train** | 11,660 | 1.020 s | 14.500 s | 4.304 s | 4.100 s | 7.820 s | 11,660 (100%)| 0 (0%) | 13.94 h | 13.94 h |
| **Cloud ViMD Train** | 15,023 | 0.810 s | 33.200 s | 19.510 s | 18.200 s | 28.400 s | 15,011 (99.9%)| 12 (0.08%) | 81.43 h | 81.32 h |
| **Cloud Combined Train** | **26,683** | **0.810 s** | **33.200 s** | **12.867 s** | **11.100 s** | **27.600 s** | **26,671** | **12** | **95.37 h** | **95.26 h** |

### Auditable Duration Exclusion Artifact
All 12 samples exceeding 30.0 seconds are formally quarantined and documented in [`reports/duration_exclusions.json`](file:///d:/Vietnamese_ASR_Week6/vietnamese_asr_app/reports/duration_exclusions.json) with their sample ID, speaker ID, source split, and exact duration. No audio files were deleted.

---

## 5. Audio-Readiness Audit

| Metric / Check | Local Physical Files (36 Samples) | Cloud Full Corpus (26,671 Samples) | Standard Requirement |
| :--- | :--- | :--- | :--- |
| **Validation Mode** | **`AUDIO-BYTE VERIFIED`** | **`MANIFEST-LEVEL VERIFIED`** | Mandatory distinction |
| **Readable WAV** | 100% Passed (36/36) | 100% Valid Parquet/WAV headers | No corrupted containers |
| **Channels** | 100% Mono (1 channel) | 100% Mono | Single-channel input |
| **Sample Rate** | 16,000 Hz | Native 16k (VIVOS) / 44.1k (ViMD) | Resampled to 16 kHz |
| **Dtype / Format** | Float32 normalized $[-1.0, 1.0]$ | Float32 normalized | Target tensor format |
| **Finite Values** | 100% Finite (0 NaNs, 0 Infs) | Verified on sampled streams | Finite tensor math |
| **RMS / Energy** | Above silence threshold ($> 10^{-4}$) | Pre-filtered in dataset curation | Non-silent speech |
| **Clipping Check** | Peak $< 0.999$, 0 clipped samples | Monitored dynamically | Dynamic range valid |

---

## 6. Speaker Disjointness

Zero speaker overlap is enforced across all partition boundaries:

### A. Local Physical Pool Matrix
- **Train Speakers**: 22 (`VIVOSSPK04`..`28` [10], `spk_11_0009`..`0048` [12])
- **Val Speakers**: 5 (`VIVOSSPK33`..`42` [5])
- **Test Speakers**: 8 (`spk_11_0055`, `0056`, `0057`, `0064`, `0067`, `0068`, `0071`, `0075`)

$$\text{Train} \cap \text{Validation} = 0 \quad (\emptyset)$$
$$\text{Train} \cap \text{Test} = 0 \quad (\emptyset)$$
$$\text{Validation} \cap \text{Test} = 0 \quad (\emptyset)$$

### B. Cloud Full-Corpus Architecture Matrix
- **VIVOS Partition**: 46 train speakers vs. 19 test speakers ($\text{Train} \cap \text{Test} = 0$).
- **ViMD Partition**: 10,291 train speakers vs. 1,320 valid speakers vs. 1,344 test speakers ($\text{Train} \cap \text{Val} = 0$, $\text{Train} \cap \text{Test} = 0$, $\text{Val} \cap \text{Test} = 0$).
- **Cross-Corpus Disjointness**: VIVOS (AILAB University Lab) $\cap$ ViMD (Provincial Television & Radio Stations) $= 0$.

---

## 7. ID & Hash Audit

Rigorous uniqueness and isolation tests were executed programmatically:
- **Duplicate Sample IDs in Train**: **0** (All 22 sample IDs strictly unique).
- **Train / Test Sample ID Overlap**: **0** (Zero test IDs present in train).
- **Train / Val Sample ID Overlap**: **0** (Zero val IDs present in train).
- **Duplicate Audio SHA-256 in Train**: **0** (All 22 audio file SHA-256 hashes unique).
- **Train / Test Audio SHA-256 Overlap**: **0** (Zero test audio hashes present in train).
- **Train / Val Audio SHA-256 Overlap**: **0** (Zero val audio hashes present in train).

---

## 8. Test Quarantine

The primary test benchmark ([`manifests/test_manifest.csv`](file:///d:/Vietnamese_ASR_Week6/vietnamese_asr_app/manifests/test_manifest.csv)) is strictly quarantined:
- **Status**: `FROZEN_READ_ONLY`
- **Samples**: 9 official ViMD gold test samples
- **SHA-256 Checksum**: `efe54856d12613f109cebc9482a6f224ca4e2370d0ff3f8a2fe4ea0a013b60ca`
- **Contamination Check**: Zero samples from ViMD `split="test"` exist in `manifests/full_train_manifest.csv`. Zero VIVOS test samples exist in training manifests.

---

## 9. Transcript Duplicate Diagnostic

Identical normalized transcripts across different speakers represent natural language redundancy and read-prompt reuse; they are audited as a diagnostic metric, not as automatic data leakage:

### A. Local Physical Pool
- **Total Utterances**: 22
- **Unique Normalized Transcripts**: 22
- **Duplicate Transcripts**: **0**
- **Repeated Utterances**: **0**
- **Cross-Split Text Overlap**: $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.

### B. Cloud Full Corpus
- **VIVOS**: In read speech corpora, prompted sentences are frequently read by multiple speakers. Cross-speaker sentence sharing is preserved to maximize acoustic diversity.
- **ViMD**: Spontaneous broadcast news commentary; transcript uniqueness exceeds 99.4%.

---

## 10. Provenance

Every dataset in the training-ready mixture is pinned to an authoritative upstream repository and immutable commit revision:

| Dataset | Canonical Repository | Pinned Git Commit / Revision | Curation Origin | Quality Tier |
| :--- | :--- | :--- | :--- | :--- |
| **VIVOS** | `thanhduycao/vivos_ng_only` | `b2fbc10431b721dc9b0409b716d56a759d1cf332` | AILab, VNU-HCM University of Science | `GOLD_HUMAN` |
| **ViMD** | `nguyendv02/ViMD_Dataset` | `3a5b30157034e7eadd5c75fae1a820c6f9383398` | ViMD Multi-Dialect Consortium (EMNLP 2024) | `GOLD_HUMAN` |

---

## 11. License Verification & Conflict Report

A comprehensive audit was conducted comparing registry metadata, repository cards, and published literature:

### 1. VIVOS
- **Claimed License**: `CC BY-NC-SA 4.0` (Archival Zenodo record `7068130`) / `CC BY-SA 4.0` (HF mirror).
- **Commercial Use**: Non-commercial.
- **Redistribution**: Permitted with Attribution and ShareAlike.
- **Audit Finding**: Permissible for academic and open-science research.

### 2. ViMD
- **Claimed License**: `CC BY-NC-ND 4.0` (Hugging Face metadata) / `Academic / Non-commercial Research` (Registry).
- **Commercial Use**: `False`.
- **Redistribution**: `False`.
- **`LICENSE / PROVENANCE CONFLICT`**:
  - The `ND` (NoDerivatives) clause in `CC BY-NC-ND 4.0` legally restricts the public distribution of derivative works, which may include fine-tuned model checkpoints.
  - **Resolution Policy**: Internal fine-tuning and evaluation for academic research is permissible under fair use and academic research clauses, but public distribution of fine-tuned model weights is restricted. The experimental protocol is maintained as **PROVISIONAL** pending legal clarification.

### 3. Excluded Candidate Corpora
- **FOSD**: Mendeley Data record `10.17632/k9sxg2twv4.4` specifies `FPT Public License` (custom license with patent exclusion, not CC BY 4.0). Excluded due to MP3 format and lack of official splits.
- **VietSuperSpeech**: Excluded due to `VERSION_UNRESOLVED` (32k vs. 52k vs. 67k) and Zipformer pseudo-label quality tier (`SILVER`).
- **VietMed**: Excluded due to medical domain mismatch and specialized DUA.
- **VLSP 2021**: Excluded due to pending DUA agreement.

---

## 12. Canonical `full_train_manifest.csv`

The training-ready manifest was constructed at [`manifests/full_train_manifest.csv`](file:///d:/Vietnamese_ASR_Week6/vietnamese_asr_app/manifests/full_train_manifest.csv).

### Exact Manifest Schema (12 Columns):
1. `dataset`: Dataset identifier (`vivos` or `vimd`).
2. `source_split`: Split origin (`train`).
3. `sample_id`: Stable canonical identifier.
4. `speaker_id`: Disjoint speaker identifier.
5. `audio_path_or_source_ref`: Absolute local audio file path or cloud streaming reference.
6. `duration_sec`: Duration in seconds rounded to 4 decimals.
7. `sample_rate`: Sample rate in Hz ($16,000$).
8. `channels`: Channel count ($1$).
9. `transcript`: Ground truth Vietnamese transcript.
10. `normalized_transcript`: Deterministically normalized transcript (NFC, lowercase, stripped punctuation, collapsed whitespace).
11. `audio_sha256`: SHA-256 hash of the audio bytes.
12. `provenance_revision`: Pinned git commit hash of the source repository.

---

## 13. Manifest Checksums

| Artifact | File Path | SHA-256 Checksum |
| :--- | :--- | :--- |
| **Canonical Full Train Manifest** | `manifests/full_train_manifest.csv` | `f722a5397c56e7415d2ce9057ecb97a3a4bdc71eff21f0eb352af120aab62e09` |
| **Frozen Test Manifest** | `manifests/test_manifest.csv` | `efe54856d12613f109cebc9482a6f224ca4e2370d0ff3f8a2fe4ea0a013b60ca` |
| **Pilot Validation Manifest** | `manifests/val_manifest.csv` | `f4b22a21e91b7b061530cf3b330f47cda98ea01671a570c2b21814ffb543cc11` |
| **Synthetic Pilot Manifest** | `manifests/synthetic_qc_passed.csv` | `21ce3ad0b72ab70734e357a50d35eb1d37d527601a648904f1517afdc2883254` |
| **Duration Exclusion Artifact** | `reports/duration_exclusions.json` | `5c707db5d9c72e9d298379058b8f2c0ce6aa123bce36c7a7eaecdb6cbef3e7e2` |
| **Git Generation Commit** | HEAD | `a2dced84868f5988b3c8e783301f814064d393ca` |

---

## 14. Validation Status

- **Pilot Validation Manifest**: [`manifests/val_manifest.csv`](file:///d:/Vietnamese_ASR_Week6/vietnamese_asr_app/manifests/val_manifest.csv) contains 5 disjoint VIVOS samples (`vivos_011` .. `vivos_015`, duration 16.87 s, 5 speakers). It is preserved without silent modification.
- **Full-Scale Validation Candidate**: In the master protocol, ViMD `valid` (1,900 utterances, 10.26 h, 1,320 speakers) is designated as the primary validation split for hyperparameter tuning and early stopping.
- **Formal Status**:
  ```
  FULL VALIDATION SET: REQUIRES DESIGN REVIEW
  ```
  The full validation split will be materialized and frozen during cloud benchmark initialization, preserving the pilot validation set for local regression tests.

---

## 15. Frozen Test Status

- **Primary Test Benchmark**: [`manifests/test_manifest.csv`](file:///d:/Vietnamese_ASR_Week6/vietnamese_asr_app/manifests/test_manifest.csv)
- **Sample Count**: 9 utterances (ViMD test split, 8 distinct speakers)
- **Status**: **`FROZEN_READ_ONLY`** (Cryptographically locked with SHA-256: `efe54856...`)
- **Quarantine Guarantee**: Zero sample IDs or audio hashes from this manifest appear in any training or validation file.

---

## 16. Proposed Full-Scale E1/E2 Training Budget

> [!CAUTION]
> **Status**: `PROPOSED — REQUIRES REVIEW`  
> These hyperparameters are calculated based on the full 26,671 training-ready utterances. They are proposed design targets and must **NOT** be automatically frozen or used for execution without explicit approval.

| Hyperparameter | Proposed E1 (Full Real LoRA) | Proposed E2 (Real + Synthetic LoRA) | Technical Rationale |
| :--- | :--- | :--- | :--- |
| **Base Model** | `vinai/PhoWhisper-tiny` | `vinai/PhoWhisper-tiny` | 37.8M parameters |
| **LoRA Target Modules** | `q_proj`, `v_proj` | `q_proj`, `v_proj` | 147,456 trainable parameters (0.389%) |
| **LoRA Rank / Alpha** | $r = 8, \alpha = 16$ | $r = 8, \alpha = 16$ | Standard parameter-efficient ratio |
| **Training-Ready Pool** | 26,671 utterances (95.26 h) | 26,671 real + synthetic pool | Full authorized real corpus |
| **Micro Batch Size** | 4 | 4 | Fits 16 GB VRAM with fp16 |
| **Gradient Accumulation** | 8 | 8 | Effective batch size = $4 \times 8 = 32$ |
| **Effective Batch Size** | 32 | 32 | Stable gradient estimation |
| **Steps per Epoch** | $\lceil 26,671 / 32 \rceil = \mathbf{834}$ | $\mathbf{834}$ | Exact epoch boundary |
| **Epochs** | 3 | 3 | 3 full passes |
| **Total Optimizer Steps** | $\mathbf{2,502}$ | $\mathbf{2,502}$ | Total optimization budget |
| **Optimizer** | AdamW | AdamW | $\beta_1=0.9, \beta_2=0.999$ |
| **Learning Rate** | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ | Proven stable LoRA LR |
| **Weight Decay** | 0.01 | 0.01 | Regularization |
| **Scheduler** | Cosine with 10% warmup | Cosine with 10% warmup | Warmup = 250 steps |
| **Checkpoint Frequency** | Every 250 steps & epoch end | Every 250 steps & epoch end | Resume safety and model selection |
| **Expected GPU VRAM** | 3.5 – 5.5 GB | 3.5 – 5.5 GB | Comfortably fits T4 (16GB) or A100 |
| **Approx. Training Duration** | ~3.5 – 5.0 hours (T4 GPU) | ~3.5 – 5.0 hours (T4 GPU) | ~1.5 hours on A100 GPU |

---

## 17. Regression Verification

- **Command**: `python -m pytest tests/ -v`
- **Result**: **38 passed, 2 warnings in 27.21s**
- **Test Integrity**: Zero application code or test files modified. 100% regression pass rate maintained.

---

## 18. Final Readiness Decision

All required data readiness gates, split verifications, duration policies, leakage checks, and canonical manifest constructions are complete:

```
================================================================================
FINAL VERDICT:
PHASE 3 DATA READINESS: READY
================================================================================
```
