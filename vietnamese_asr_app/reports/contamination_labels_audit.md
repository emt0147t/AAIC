# Dataset Contamination Labels & Pretraining Provenance Audit Report

> **C STATUS: PASS**  
> **Investigation Target:** Transparent classification of evaluation benchmarks as `SAFE`, `CONTAMINATED`, or `UNKNOWN` based on PhoWhisper pretraining corpora provenance.

---

## 1. Executive Summary

In Section 15.2 of `application_technical_overview.md`, evaluating PhoWhisper on a 15-sample VIVOS pilot yielded 0.00% WER and 0.00% CER (100% exact match). An independent review flagged this as an indicator of in-domain data contamination.

To ensure transparent reporting and prevent misleading generalizability claims:
1. **VIVOS Corpus Audited:** Verified through the official PhoWhisper publication that VIVOS was included in pretraining. VIVOS benchmarks are officially re-labeled as **`CONTAMINATED`**.
2. **ViMD Corpus Audited:** Inspected PhoWhisper pretraining sources (VIVOS, VLSP 2020, Common Voice 11.0, and 500 hours of Vietnamese YouTube audio). ViMD is not listed by name, but because the pretraining crawl included YouTube audio, overlap cannot be mathematically ruled out without sample-level acoustic hashing. Adhering to the scientific mandate (*"Nếu không xác minh được, đánh dấu UNKNOWN, không suy luận 'không thấy nên chắc chắn sạch'"*), ViMD is officially labeled as **`UNKNOWN`**.
3. **Live Voice Demo Audited:** Verified as a novel, live acoustic recording, labeled as **`SAFE`**.

---

## 2. Pretraining Provenance & Contamination Taxonomy

| Evaluation Benchmark | Audio Source | PhoWhisper Pretraining Exposure | Contamination Status | Interpretation Guardrail |
| :--- | :--- | :--- | :---: | :--- |
| **Real Voice Demo** | Single live microphone recording | None (Created post-release) | **`SAFE`** | Valid for single-speaker pipeline smoke testing. |
| **VIVOS 15-Sample Pilot** | `thanhduycao/vivos_ng_only` | **Explicitly Included** (~15 hours VIVOS corpus used in pretraining) | **`CONTAMINATED`** | **In-Domain Overfitting Artifact**. 0.00% WER cannot be cited as generalization performance. |
| **ViMD 9-Sample Pilot** | `nguyendv02/ViMD_Dataset` | **Unverified Potential Overlap** (YouTube pretraining crawl) | **`UNKNOWN`** | Pipeline smoke test only; dialect generalization claims prohibited. |
| **ViMD 2,026 Full Test** | `nguyendv02/ViMD_Dataset` | **Unverified Potential Overlap** (YouTube pretraining crawl) | **`UNKNOWN`** | Primary empirical benchmark, interpreted with stated unknown pretraining overlap status. |

---

## 3. Evidence from PhoWhisper Model Documentation

From the official publication (*"PhoWhisper: A Vietnamese Speech Recognition Model"*, 2023):
- **Pretraining Corpora:**
  1. VIVOS (15 hours) — *Confirmed direct overlap with VIVOS pilot.*
  2. VLSP 2020 (84 hours)
  3. Mozilla Common Voice 11.0 Vietnamese (30 hours)
  4. Vietnamese YouTube Speech Crawl (~500 hours) — *Acoustic overlap with ViMD YouTube sources is unverified.*

### Policy Enforcement:
All summary tables (including Table C of `application_technical_overview.md`) are updated with the mandatory `Contamination Status` column.
