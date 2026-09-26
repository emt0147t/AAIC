# Full E1 Readiness Reconciliation Audit Report

**Date:** 2026-09-26  
**Repository:** `Vietnamese_ASR_Week6`  
**Phase:** Phase 3.5 — Full E1 Readiness Reconciliation  
**Audit Status:** `COMPLETE`  
**Gate Status:** `FULL E1: BLOCKED`  

---

## EXECUTIVE SUMMARY & GATE VERDICT

This audit rigorously investigates and reconciles all outstanding Full E1 readiness blockers in strict compliance with the **Hard Safety Rule**.

```text
FULL E1: BLOCKED
```

### Blocker Breakdown:
1. **`CLOUD ACCESS` (Technical Preflight Blocker)**:
   - Operating System: Windows 11 AMD64
   - Host GPU: NVIDIA RTX A2000 Laptop GPU (4,096 MiB VRAM)
   - Python Version: 3.14.0 AMD64
   - PyTorch Distribution: `torch 2.10.0+cpu` (`torch.cuda.is_available() == False` because prebuilt PyTorch CUDA binary wheels are not yet released for Python 3.14 on Windows).
   - Critical Gate Criterion: The preflight gate strictly mandates recording actual peak VRAM via `torch.cuda.max_memory_allocated()`. On this host, PyTorch executes via CPU fallback (`torch.cuda.max_memory_allocated() = 0.0 MB`). Real GPU profiling and full training require launching on the target Google Colab Linux GPU environment via [`ASR_PHASE3_5_GPU_THROUGHPUT_PREFLIGHT.ipynb`](file:///D:/Vietnamese_ASR_Week6/ASR_PHASE3_5_GPU_THROUGHPUT_PREFLIGHT.ipynb).
2. **`PROVENANCE` (Compliance Blocker)**:
   - ViMD is released under Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (**CC BY-NC-ND 4.0**).
   - Under standard interpretations, distributing fine-tuned model checkpoint weights derived from ViMD violates the NoDerivatives ("ND") clause unless explicit institutional consent is secured.
   - All fine-tuned checkpoints must remain restricted to internal research evaluations.

---

## 1. BLOCKER A RECONCILIATION: FULL TRAIN MANIFEST DATASET COUNTS

### 1.1 The Reported Discrepancy
In the prior conversational summary, the training-ready corpus was reported as:
- *Erroneous Chat Summary:* VIVOS = 14,460, ViMD = 12,211 (Total = 26,671).
- *Previous Master Audit History:* VIVOS raw train = 11,660, ViMD raw train = 15,023 (Total = 26,683).

The user correctly flagged that $14,460 \text{ VIVOS} > 11,660 \text{ raw VIVOS}$, indicating an apparent impossibility.

### 1.2 Forensic Trace of Authoritative Primary Artifacts
Every authoritative registry, index, metadata, and data card artifact across the repository was inspected to trace the ground truth:

1. **VIVOS Corpus**:
   - **Canonical Origin**: AILab, VNU-HCM University of Science.
   - **Official Distribution**: Zenodo Record `7068130` (DOI: `10.5281/zenodo.7068130`).
   - **Pinned Repository Revision**: `thanhduycao/vivos_ng_only`, commit `b2fbc10431b721dc9b0409b716d56a759d1cf332` (registered in [`datasets/registry.py`](file:///D:/Vietnamese_ASR_Week6/vietnamese_asr_app/datasets/registry.py#L10-L28)).
   - **Official Splits**:
     - `train`: exactly **11,660 utterances** (13.94 hours, 46 speakers).
     - `test`: exactly **760 utterances** (0.75 hours, 19 speakers).
     - `validation`: **0** (VIVOS has no official validation split).
   - **Physical Reality**: There are only 11,660 train utterances in VIVOS in existence. It is physically impossible for VIVOS train to contain 14,460 utterances.

2. **ViMD Corpus**:
   - **Canonical Origin**: ViMD Multi-Dialect Speech Consortium.
   - **Official Distribution**: HuggingFace `nguyendv02/ViMD_Dataset`.
   - **Pinned Repository Revision**: commit `3a5b30157034e7eadd5c75fae1a820c6f9383398` (registered in [`datasets/registry.py`](file:///D:/Vietnamese_ASR_Week6/vietnamese_asr_app/datasets/registry.py#L29-L46)).
   - **Official Splits**:
     - `train`: exactly **15,023 utterances** (81.43 hours, 10,291 speakers across 63 provinces).
     - `valid`: exactly **1,900 utterances** (10.26 hours, 1,320 speakers).
     - `test`: exactly **2,026 utterances** (10.87 hours, 1,344 speakers).

3. **Duration Exclusion Audit Log**:
   - File: [`reports/duration_exclusions.json`](file:///D:/Vietnamese_ASR_Week6/vietnamese_asr_app/reports/duration_exclusions.json)
   - Hash (SHA-256): `5c707db5d9c72e9d298379058b8f2c0ce6aa123bce36c7a7eaecdb6cbef3e7e2`
   - Content:
     ```json
     {
       "cloud_full_corpus": {
         "total_candidates": 26683,
         "candidates_le_30s": 26671,
         "candidates_gt_30s": 12,
         "vivos_train_gt_30s": 0,
         "vimd_train_gt_30s": 12
       }
     }
     ```
   - All 12 exclusions (>30.0s) originate from **ViMD train** (durations 30.05s to 33.20s).
   - **VIVOS train exclusions (>30s):** Exactly **0** (all 11,660 samples are $\le 14.50\text{s}$).

4. **Authoritative Phase 3 Data Card**:
   - File: [`reports/phase3_full_corpus_data_card.md`](file:///D:/Vietnamese_ASR_Week6/vietnamese_asr_app/reports/phase3_full_corpus_data_card.md) (Lines 41–43):
     ```
     | Tier | Dataset | Raw Utterances | Excluded (>30s) | Training-Ready Utterances |
     | Cloud Full Scale | VIVOS train | 11,660 | 0  | 11,660 |
     | Cloud Full Scale | ViMD train  | 15,023 | 12 | 15,011 |
     | Cloud Full Scale Total | Combined | 26,683 | 12 | 26,671 |
     ```

### 1.3 Exact Mathematical Reconciliation
The side-by-side reconciliation confirms the ground truth:

| Dataset | Raw Train Utterances | Duration Excluded (>30s) | Authoritative Training-Ready | Erroneous Chat Transcription |
| :--- | :--- | :--- | :--- | :--- |
| **VIVOS train** | **11,660** | 0 | **11,660** | 14,460 *(Disproved)* |
| **ViMD train** | **15,023** | 12 | **15,011** | 12,211 *(Disproved)* |
| **Total** | **26,683** | **12** | **26,671** | 26,671 |

### Conclusion on Blocker A:
The numbers "14,460 VIVOS and 12,211 ViMD" resulted from an **accidental conversational transposition in the previous chat response prose** ($14,460 + 12,211 = 26,671$).  
The actual repository files, database registries, and duration exclusion logs have always recorded the true counts:
- **VIVOS training-ready:** exactly **11,660**
- **ViMD training-ready:** exactly **15,011**
- **Combined training-ready total:** exactly **26,671**

---

## 2. RECONCILIATION OF LOCAL DISK MANIFEST VS CLOUD STREAMING

### 2.1 Manifest on Disk: `manifests/full_train_manifest.csv`
- File: [`manifests/full_train_manifest.csv`](file:///D:/Vietnamese_ASR_Week6/vietnamese_asr_app/manifests/full_train_manifest.csv)
- Total Rows: **22 rows** (10 VIVOS clean audio samples + 12 ViMD clean audio samples)
- Duration: 156.58 seconds
- SHA-256: `f722a5397c56e7415d2ce9057ecb97a3a4bdc71eff21f0eb352af120aab62e09`
- Purpose: Represents the audited, locally instantiated physical slice used for local regression tests and offline development.

### 2.2 Cloud Full-Scale Dataset
- For the full 26,671-utterance (~95.26 hour) training run, samples are ingested via Hugging Face datasets streaming under pinned revisions (`thanhduycao/vivos_ng_only@b2fbc10` and `nguyendv02/ViMD_Dataset@3a5b301`) directly in the cloud GPU environment.

---

## 3. AUDIT OF VALIDATION & TEST SETS

### 3.1 Validation Manifests
1. **Pilot Validation Manifest**: `manifests/val_manifest.csv`
   - Samples: 5
   - SHA-256: `f4b22a21e91b7b061530cf3b330f47cda98ea01671a570c2b21814ffb543cc11`
   - Status: Untouched and preserved for local unit tests.
2. **Full Validation Manifest**: `manifests/full_val_manifest.csv`
   - Samples: 1,900 (ViMD valid)
   - Duration: 36,929.5 s (~10.26 hours)
   - Speaker Count: 1,320
   - Speaker Overlap with Train: **0**
   - Speaker Overlap with Test: **0**
   - SHA-256: `a15433fbdf94c35815af570020bee1fc8a23b8c6e8602b4ddc450fed38f86386`

### 3.2 Frozen Test Manifest
- File: `manifests/test_manifest.csv`
- Samples: 9 official ViMD gold test samples
- SHA-256: `efe54856d12613f109cebc9482a6f224ca4e2370d0ff3f8a2fe4ea0a013b60ca`
- Status: **Strictly frozen, verified, and unchanged**.

---

## 4. OPTIMIZER BUDGET RECONCILIATION

- **Effective Batch Size:** 32 ($micro\_batch=4 \times grad\_accum=8$)
- **Batches per Epoch:** $\lceil 26,671 / 32 \rceil = 834$ optimizer steps
- **Exposures per Epoch:** $834 \times 32 = 26,688$ exposures
- **Exposure Delta:** $+17$ exposures per epoch (deterministic padding under seed 42)
- **Full Schedule (3 Epochs):**
  - Total Optimizer Steps: $834 \times 3 = \mathbf{2,502\text{ steps}}$
  - Total Sample Exposures: $26,688 \times 3 = \mathbf{80,064\text{ exposures}}$
  - Warmup: 250 steps (10%) | Cosine Decay: 2,252 steps (90%)

---

## 5. FINAL READINESS CONCLUSION

All metadata, dataset counts, and split boundaries are fully reconciled and verified against physical evidence in the repository.

However, because:
1. `torch.cuda.max_memory_allocated()` cannot be invoked on real CUDA hardware on this local Windows machine (`torch.cuda.is_available() == False` under Python 3.14), and
2. ViMD CC BY-NC-ND 4.0 license derivative restrictions require external execution and compliance controls,

The preflight gate strictly adheres to Section 0 Hard Safety Rules:

```text
FULL E1: BLOCKED
```
