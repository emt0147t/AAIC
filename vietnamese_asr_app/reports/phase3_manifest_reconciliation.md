# Phase 3 Manifest & Corpus Identity Reconciliation Report

**Date:** 2026-09-26  
**Repository:** `Vietnamese_ASR_Week6`  
**Status:** `AUDITED & VERIFIED`  
**Benchmark Target:** Full E1 LoRA Fine-Tuning  

---

## 1. EXECUTIVE SUMMARY

This report authoritatively clarifies and formalizes the exact identity and boundaries of the training data assets within the repository, resolving any ambiguity between the local offline development slice and the full-scale cloud streaming training corpus.

---

## 2. LOCAL DEVELOPMENT MANIFEST VS. FULL-SCALE CORPUS

```
+--------------------------------------------------------------------------------------------------+
|                                    DATA CORPUS ARCHITECTURE                                      |
+--------------------------------------------------+-----------------------------------------------+
| A. LOCAL DEVELOPMENT MANIFEST                    | B. FULL-SCALE STREAMING CORPUS                |
| (Offline Development & Test Suite Validation)    | (Full E1 Fine-Tuning on Cloud GPU)            |
+--------------------------------------------------+-----------------------------------------------+
| Path: manifests/full_train_manifest.csv          | Access: HF streaming (pinned revisions)       |
| Rows: 22 physical WAV files on disk              | Total Unique Utterances: 26,671 (~95.26h)     |
| Duration: 156.58 seconds (~2.61 min)             | VIVOS train: 11,660 utterances (13.94h)       |
| Hash: f722a5397c56e7415d2ce9057ecb97a3a4bdc71... | ViMD train: 15,011 utterances (81.32h)        |
| Purpose: Smoke tests, CI regression suites       | Purpose: Full E1 LoRA 3-epoch optimization    |
+--------------------------------------------------+-----------------------------------------------+
```

### 2.1 Manifest A: Local Development Manifest
- **Path:** [`manifests/full_train_manifest.csv`](file:///D:/Vietnamese_ASR_Week6/vietnamese_asr_app/manifests/full_train_manifest.csv)
- **Row Count:** Exactly **22 rows**
  - VIVOS train slice: 10 rows (`vivos_001` .. `vivos_010`, 32.44s)
  - ViMD train clean slice: 12 rows (`vimd_train_01` .. `vimd_train_12`, 124.14s)
- **Total Duration:** 156.58 seconds (~2.61 minutes)
- **SHA-256 Hash:** `f722a5397c56e7415d2ce9057ecb97a3a4bdc71eff21f0eb352af120aab62e09`
- **Purpose:** Fast offline test suite execution (`pytest tests/`), collator integrity testing, and local verification without requiring multi-gigabyte audio downloads.
- **Explicit Boundary:** **This 22-row manifest is NOT the complete Full E1 training corpus.** It represents only an instantiated local testing partition.

### 2.2 Corpus B: Full-Scale Cloud Training Corpus
- **Access Architecture:** Pinned streaming via Hugging Face `datasets` directly on cloud GPU.
- **Component 1 — VIVOS Train**:
  - Source Repository: `thanhduycao/vivos_ng_only` (mirror of Zenodo record 7068130, AILab VNU-HCM)
  - Pinned Revision: `b2fbc10431b721dc9b0409b716d56a759d1cf332`
  - Source Split: `train`
  - Native Audio Format: 16 kHz WAV mono
  - Raw Candidate Count: **11,660 utterances** (13.94 hours, 46 speakers)
  - Duration Policy Filter ($0.5\text{s} \le \text{duration} \le 30.0\text{s}$): All samples $\le 14.50\text{s}$.
  - Exclusions (>30.0s): **0**
  - Training-Ready Count: **11,660 utterances**
- **Component 2 — ViMD Train**:
  - Source Repository: `nguyendv02/ViMD_Dataset` (ViMD Multi-Dialect Speech Consortium)
  - Pinned Revision: `3a5b30157034e7eadd5c75fae1a820c6f9383398`
  - Source Split: `train`
  - Native Audio Format: 44.1 kHz WAV in parquet (resampled to 16 kHz for Whisper)
  - Raw Candidate Count: **15,023 utterances** (81.43 hours, 10,291 speakers across 63 provinces)
  - Duration Policy Filter ($0.5\text{s} \le \text{duration} \le 30.0\text{s}$): Range 30.05s – 33.20s filtered out.
  - Exclusions (>30.0s): **12 utterances** (audited in `reports/duration_exclusions.json`, SHA-256: `5c707db5d9c72e9d298379058b8f2c0ce6aa123bce36c7a7eaecdb6cbef3e7e2`)
  - Training-Ready Count: **15,011 utterances** ($15,023 - 12$)
- **Combined Training-Ready Full-Scale Total:** Exactly **26,671 unique utterances** (~95.26 hours audio).

---

## 3. AUDIT OF DATASET COUNT RECONCILIATION

| Dimension | Raw Train Pool | Duration Exclusions (>30s) | Authoritative Training-Ready | Erroneous Chat Transcription | Audit Source File |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **VIVOS train** | 11,660 | 0 | **11,660** | 14,460 *(Refuted)* | [`reports/phase3_full_corpus_data_card.md#L41`](file:///D:/Vietnamese_ASR_Week6/vietnamese_asr_app/reports/phase3_full_corpus_data_card.md#L41) |
| **ViMD train** | 15,023 | 12 | **15,011** | 12,211 *(Refuted)* | [`reports/duration_exclusions.json#L15`](file:///D:/Vietnamese_ASR_Week6/vietnamese_asr_app/reports/duration_exclusions.json#L15) |
| **Combined Full Scale** | 26,683 | 12 | **26,671** | 26,671 | [`reports/phase3_full_corpus_data_card.md#L43`](file:///D:/Vietnamese_ASR_Week6/vietnamese_asr_app/reports/phase3_full_corpus_data_card.md#L43) |
| **Local Physical Disk** | 22 | 0 | **22** | — | [`manifests/full_train_manifest.csv`](file:///D:/Vietnamese_ASR_Week6/vietnamese_asr_app/manifests/full_train_manifest.csv) |

### Discrepancy Forensic Conclusion:
The values "14,460 VIVOS / 12,211 ViMD" represented an accidental conversational transposition in prose during chat synthesis ($14,460 + 12,211 = 26,671$). The underlying repositories, data cards, and exclusion logs never stored these numbers. The verified ground truth is **11,660 VIVOS** and **15,011 ViMD**.

---

## 4. ZERO CONTAMINATION & LEAKAGE VERIFICATION

1. **Validation Corpus (`manifests/full_val_manifest.csv`)**:
   - Authorized Source: ViMD `valid` split (1,900 utterances, 10.26 hours, 1,320 speakers).
   - SHA-256: `a15433fbdf94c35815af570020bee1fc8a23b8c6e8602b4ddc450fed38f86386`.
   - Speaker overlap with training corpus: **0**.
   - Speaker overlap with test set: **0**.
2. **Frozen Test Manifest (`manifests/test_manifest.csv`)**:
   - Authorized Source: ViMD official `test` gold slice (9 utterances).
   - SHA-256: `efe54856d12613f109cebc9482a6f224ca4e2370d0ff3f8a2fe4ea0a013b60ca`.
   - Status: **Strictly frozen and unchanged**.
   - Sample-ID overlap: **0**.
   - Audio-hash overlap: **0**.
   - Contamination violations: **0**.
