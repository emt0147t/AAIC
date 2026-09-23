# 14 — Full ASR Benchmark Preflight & Protocol Verification Report

> **EXPERIMENT STATUS**: **PREFLIGHT COMPLETE — STOPPED AT COMPUTE GATE**  
> **Preflight Result**: All manifest checksums, split disjointness checks, transcript normalization rules, and audio batch execution pipelines passed 100%.  
> **Stopping Trigger**: In accordance with Section 14 & 15 ("If any value conflicts with the saved protocol, STOP and report the conflict"), execution halted due to a hardware runtime conflict: while an `NVIDIA RTX A2000 Laptop GPU` (4 GB VRAM) is physically present, the active Python 3.14 Windows environment only provides `torch 2.10.0+cpu` (`torch.cuda.is_available() == False`). Running the full 95.4-hour (26,683 utterances) 5-epoch training job on CPU requires ~28.0 hours of uninterrupted execution and ~74 GB parquet download.
> **Test Integrity**: Test manifests remain **FROZEN & READ-ONLY** with cryptographic SHA-256 hashes locked in `artifacts/full_benchmark/manifest_hashes.json`. Zero test samples were touched.

---

## 1. Experiment Objective

The primary objective of this phase was to initiate the first full-scale Vietnamese Automatic Speech Recognition (ASR) benchmark under the frozen protocol established in Gates 0–7:
- Model: `vinai/PhoWhisper-tiny` (37.8M parameters).
- Training Set: Full ~95.4-hour Gold human-transcribed corpus (VIVOS train + ViMD train = 26,683 utterances).
- Validation Set: Official ViMD valid split (1,900 utterances, 10.26 hours, 1,320 disjoint speakers).
- Primary Test Benchmark: Official ViMD test split (2,026 utterances, 10.87 hours, 1,344 disjoint speakers).
- Secondary Reference Test: Official VIVOS test split (760 utterances, 0.75 hours, 19 disjoint speakers).

---

## 2. Frozen Dataset Protocol & Manifest Integrity

### Manifest Verification & Disjointness Check
Prior to launching any compute-intensive tasks, an end-to-end manifest audit was conducted across all splits:

| Split | Dataset | Verified Utterances | Verified Hours | Speaker Count | Disjointness Verification | SHA-256 Manifest Segment Hash |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **Train** | VIVOS train | 11,660 | 13.94h | 46 speakers | Disjoint from test (0 overlaps) | `985f49c7e168bec118229146f8eb...` |
| **Train** | ViMD train | 15,023 | 81.43h | 10,291 speakers | Disjoint from valid/test (0 overlaps) | `7bd2cc6eefffabe20a74bb8e5a09...` |
| **Total Train** | **VIVOS + ViMD** | **26,683** | **95.37h** | **10,337 speakers** | **Strictly Disjoint** | — |
| **Validation** | ViMD valid | 1,900 | 10.26h | 1,320 speakers | Disjoint from train/test (0 overlaps) | `923f998b1f79af3fcb31cef27617...` |
| **Primary Test** | ViMD test | 2,026 | 10.87h | 1,344 speakers | Disjoint from train/val (0 overlaps) | `c4919f0cf23db807188aabfbba12...` |
| **Secondary Ref**| VIVOS test | 760 | 0.75h | 19 speakers | Disjoint from train (0 overlaps) | `868de441db75977bd781c498ab31...` |

- **ID Overlap Result**: Exactly **0 overlapping sample IDs** and **0 overlapping speaker IDs** between training, validation, and test splits.
- **Manifest Checksums**: Stored permanently in `artifacts/full_benchmark/manifest_hashes.json`.
- **Test Manifest Status**: Locked as **FROZEN_READ_ONLY**.

---

## 3. Preflight Execution Verification

### Audio & Pipeline Preflight
1. **Audio Ingestion**:
   - VIVOS: Native 16,000 Hz mono PCM float32 $\rightarrow$ direct ingestion.
   - ViMD: Native 44,100 Hz mono PCM float32 $\rightarrow$ resampled on-the-fly to 16,000 Hz via `librosa.resample`.
   - Durations preserved exactly without truncation distortion.
2. **Batch Execution Test**:
   - **Training Batch (Size 2)**: Input feature tensor shape `torch.Size([2, 80, 3000])`. Forward pass computed cross-entropy loss = **1.5962** (strictly finite: `True`). Backward gradient pass completed cleanly in **1.51 seconds**.
   - **Validation Batch Generation (Size 2)**: Autoregressive greedy decoding completed in **1.95 seconds** generating legitimate Vietnamese text (`tôi đọc ờ đọc và những cái tư liệu đầu tiên để chi...`).
   - **Pipeline Status**: **PASS**.

### Transcript Normalization Policy
Configuration saved to `artifacts/full_benchmark/transcript_normalization.json`:
- Unicode NFC normalization (`unicodedata.normalize('NFC', text)`).
- Lowercase conversion (`text.lower()`).
- Punctuation stripping (`re.sub(r'[^\w\s]', '', text)`).
- Whitespace collapsing (`re.sub(r'\s+', ' ', text).strip()`).
- Applied identically to ground-truth references and model hypotheses prior to Levenshtein distance calculation.

---

## 4. Hardware Detection & Stopping Condition Report

In accordance with Section 15 ("perform a FINAL PREFLIGHT and print... If any value conflicts with the saved protocol, STOP and report the conflict"):

```
================================================================================
                       FINAL PREFLIGHT AUDIT SUMMARY
================================================================================
TRAIN UTTERANCES       = 26,683
TRAIN HOURS            = 95.37
VALID UTTERANCES       = 1,900
VALID HOURS            = 10.26
TEST UTTERANCES        = 2,026 (ViMD test) + 760 (VIVOS test)
TEST HOURS             = 10.87 (ViMD test) + 0.75 (VIVOS test)
MODEL                  = vinai/PhoWhisper-tiny
PARAMETERS             = 37,760,640 (37,184,640 trainable)
GPU                    = NVIDIA RTX A2000 Laptop GPU (Physical) / None (PyTorch)
VRAM                   = 4,096 MiB (Physical) / 0 MiB (Accessible to PyTorch)
MICRO_BATCH            = 2
GRAD_ACCUM             = 16
EFFECTIVE_BATCH        = 32
MAX_EPOCHS             = 5
EXPECTED TRAINING COST = ~28.0 hours CPU compute | ~74 GB parquet network download
TEST STATUS            = FROZEN
================================================================================
```

### Conflict & Blocker Analysis
1. **PyTorch CUDA Incompatibility**:
   - Physical GPU: `NVIDIA RTX A2000 Laptop GPU` with 4,096 MiB VRAM and NVIDIA Driver 610.60.
   - Runtime Environment: Python 3.14 on Windows 11.
   - Installed Framework: `torch 2.10.0+cpu` (`torch.cuda.is_available() == False`).
   - Root Cause: Official PyTorch CUDA binary wheels for Windows are currently built only for Python $\le$ 3.12 / 3.13 preview. No CUDA wheel exists on PyTorch indices for Python 3.14.
2. **Execution Time & Storage Feasibility on CPU**:
   - Training 26,683 utterances for 5 epochs on CPU requires:
     $$5 \text{ epochs} \times 13,341.5 \text{ batches} \times 1.51\text{ s} \approx 100,728\text{ seconds} \approx \mathbf{28.0\text{ hours}}$$
   - Ingesting all 130 ViMD parquet shards over the network requires downloading **~74 GB**.
3. **Mandatory Stop Protocol**:
   - Section 14 explicitly commands:  
     *"If a blocking issue occurs: DO NOT improvise a new research protocol. Document the issue and stop."*
   - Section 15 explicitly commands:  
     *"If any value conflicts with the saved protocol, STOP and report the conflict."*
   - Execution was therefore halted cleanly after verifying all manifests and pipeline stages.

---

## 5. Contamination Disclosure & Test Evaluation Roles

| Benchmark Role | Dataset | Size | Contamination Classification | Evaluation Protocol Rule |
|:---|:---|:---:|:---|:---|
| **PRIMARY BENCHMARK** | ViMD test | 2,026 utt (~10.87h) | **NO_EVIDENCE_FOUND IN EXAMINED MODEL DOCUMENTATION** | Genuine out-of-domain evaluation. Manifest frozen and untouched. |
| **SECONDARY REFERENCE** | VIVOS test | 760 utt (~0.75h) | **CONFIRMED CONTAMINATED REFERENCE** | Known training contamination for PhoWhisper. Measures in-domain ceiling only. |

*Wording Adherence*: The terms "clean benchmark", "proven clean", and "uncontaminated" are strictly prohibited and omitted.

---

## 6. Reproducibility Artifacts Generated

All preflight manifests, environment specifications, and configuration parameters are preserved in `artifacts/full_benchmark/`:

| Artifact | File Path | Verification Status |
|:---|:---|:---:|
| **Environment Specification** | `artifacts/full_benchmark/environment.json` | Recorded hardware, driver, OS, and PyTorch versions |
| **Manifest Checksums** | `artifacts/full_benchmark/manifest_hashes.json` | SHA-256 hashes of all 5 split manifests locked |
| **Transcript Normalization** | `artifacts/full_benchmark/transcript_normalization.json` | 4-step deterministic normalization policy locked |
| **Experiment Configuration** | `artifacts/full_benchmark/experiment_config.json` | Full 16-point experimental setup locked |
| **Benchmark Report** | `asr_experiment/14_full_asr_benchmark_report.md` | This document |
