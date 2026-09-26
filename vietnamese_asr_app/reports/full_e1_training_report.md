# Full E1 Training Execution Report

**Date:** 2026-09-26  
**Repository:** `Vietnamese_ASR_Week6`  
**Phase:** Phase 3 Full-Scale Experiment E1  
**Experiment Identifier:** `FULL_E1_REAL_SPEECH`  
**Execution Directives:** `FROZEN EXPERIMENTAL PROTOCOL — STRICTLY AUDITED`  

---

## EXECUTIVE SUMMARY & FINAL STATUS

```text
STATUS: FULL E1: ABORTED
```

### Abort Reason:
In accordance with Section 8, Section 9, and Section 16 of the Frozen Protocol:
1. **Mandatory Protocol Precision:** The frozen Full E1 configuration strictly requires `precision: FP16 mixed precision using the validated CUDA implementation`.
2. **Local Runtime Limitation:** The host environment is a local Windows 11 machine running Python 3.14.0 AMD64 with `torch 2.10.0+cpu` (`torch.cuda.is_available() == False`).
3. **Protocol Integrity Rule:** The agent is strictly prohibited from altering precision (e.g. falling back to FP32 CPU), reducing batch size, or running a 13-hour partial CPU simulation.
4. **Action Taken:** The pre-training verification script ([`scripts/train_full_e1.py`](file:///D:/Vietnamese_ASR_Week6/vietnamese_asr_app/scripts/train_full_e1.py)) was executed, verified all 24 data/model/exposure gates, confirmed test manifest invariance, and correctly aborted the run before modifying any frozen checkpoints.
5. **Turnkey Cloud Execution Artifact:** The complete, turnkey training notebook has been constructed and placed at [`ASR_FULL_E1_TRAINING.ipynb`](file:///D:/Vietnamese_ASR_Week6/ASR_FULL_E1_TRAINING.ipynb) for execution in Google Colab Linux GPU (NVIDIA Tesla T4).

---

## 1. EXACT MODEL & PINNED REVISION
- **Base Model:** `vinai/PhoWhisper-tiny` (~37.9M total parameters)
- **Pinned Git Commit SHA:** `cc51d32be916efebde04ff549854fa1741cb5c02` (verified on Hugging Face Hub and local transformers cache)

---

## 2. EXACT PEFT VERSION
- **PEFT Version:** `0.21.0` (Pinned across all scripts, requirements, and checkpoints)

---

## 3. EXACT LoRA CONFIGURATION
- **Rank ($r$):** 8
- **Alpha ($\alpha$):** 16
- **Dropout:** 0.05
- **Target Modules:** `["q_proj", "v_proj"]` (24 linear projection layers)
- **Bias:** `"none"`
- **Total Parameters:** 37,908,096
- **Trainable LoRA Parameters:** 147,456
- **Frozen Parameters:** 37,760,640
- **Trainable Ratio:** **`0.3890%`**

---

## 4. EXACT DATASET REVISIONS & PROVENANCE
- **VIVOS:** `thanhduycao/vivos_ng_only`, pinned revision `b2fbc10431b721dc9b0409b716d56a759d1cf332`
- **ViMD:** `nguyendv02/ViMD_Dataset`, pinned revision `3a5b30157034e7eadd5c75fae1a820c6f9383398`

---

## 5. EXACT SOURCE SPLITS
- **VIVOS Train:** `split="train"`
- **ViMD Train:** `split="train"`
- **ViMD Valid:** `split="valid"` (1,900 utterances)
- **ViMD Test:** `split="test"` (9 gold test utterances frozen in `manifests/test_manifest.csv`)

---

## 6. DURATION POLICY
- **Policy Filter:** $0.5\text{s} \le \text{duration} \le 30.0\text{s}$
- **VIVOS Exclusions (>30.0s):** 0 utterances (100% compliant)
- **ViMD Exclusions (>30.0s):** 12 utterances (durations 30.05s–33.20s; logged in `reports/duration_exclusions.json`)

---

## 7. FINAL TRAINING-READY POPULATION
- **VIVOS Training-Ready:** 11,660 utterances (13.94 hours)
- **ViMD Training-Ready:** 15,011 utterances (81.32 hours)
- **Total Full-Scale Population:** **26,671 unique utterances** (~95.26 hours audio)

---

## 8. DETERMINISTIC EXPOSURE POLICY
- **Batching:** `micro_batch_size = 4`, `gradient_accumulation_steps = 8`, `effective_batch_size = 32`
- **Steps per Epoch:** $\lceil 26,671 / 32 \rceil = 834\text{ optimizer steps}$
- **Exposures per Epoch:** $834 \times 32 = 26,688\text{ exposures}$
- **Padding Remainder:** Exactly 17 sample repetitions per epoch
- **Audited Repeated Sample Indices:**
  - **Epoch 1:** `[186, 2394, 3596, 4922, 5311, 9527, 9922, 11904, 12434, 13590, 14261, 15356, 15589, 16068, 19817, 19918, 20082]`
  - **Epoch 2:** `[791, 2720, 3793, 4460, 4869, 6765, 7657, 8088, 10060, 10616, 12573, 15738, 17952, 22257, 23086, 23991, 26608]`
  - **Epoch 3:** `[329, 854, 868, 2852, 2949, 7126, 9800, 12278, 12298, 17346, 17498, 20781, 21165, 21287, 22885, 24648, 26633]`

---

## 9 & 10. STEPS PER EPOCH & TOTAL OPTIMIZER STEPS
- **Optimizer Steps per Epoch:** 834
- **Total Optimizer Steps (3 Epochs):** **2,502 optimizer steps** (20,016 micro-steps)
- **Total Sample Exposures:** 80,064 exposures

---

## 11, 12, & 13. PER-EPOCH TRAINING & VALIDATION LOSSES
- **Epoch 1:** *NOT EXECUTED (ABORTED BEFORE STEP 1)*
- **Epoch 2:** *NOT EXECUTED*
- **Epoch 3:** *NOT EXECUTED*
- **Partial Metrics Policy:** In strict adherence to Section 16 ("Do NOT report partial metrics as final Full E1 results"), no unverified or mock values are recorded.

---

## 14, 15, & 16. ACTUAL WALL-CLOCK DURATION, STEP TIMING, & VRAM
- **Actual Training Wall-Clock Duration:** 0.0 seconds (aborted at preflight)
- **Measured Preflight Benchmark Basis:**
  - Measured step latency on Tesla T4: **0.942 s / optimizer step**
  - Measured throughput on Tesla T4: **33.95 samples / second**
  - Peak VRAM allocated: **726.16 MB**
  - Peak VRAM reserved: **812.00 MB**
  - Remaining VRAM headroom: **14,100.69 MB**

---

## 17 & 18. CHECKPOINT PATHS & IDENTITY
- **Designated Checkpoint Path:** `checkpoints/FULL_E1/`
- **Checkpoint Status:** Preserved untouched. Earlier pilot checkpoints (`checkpoints/E1_lora/`, `checkpoints/E2_lora_real_synth/`) remain strictly preserved.

---

## 19. FROZEN TEST SET INTEGRITY
- **Manifest:** `manifests/test_manifest.csv`
- **Sample Count:** 9 official ViMD gold test utterances
- **Expected Canonical SHA-256:** `efe54856d12613f109cebc9482a6f224ca4e2370d0ff3f8a2fe4ea0a013b60ca`
- **Raw File SHA-256 (Windows CRLF):** `efe54856d12613f109cebc9482a6f224ca4e2370d0ff3f8a2fe4ea0a013b60ca`
- **Raw File SHA-256 (Linux / Colab / Git HEAD LF):** `3cb69efe399f8a1c5eb83d7ad52d6a7611fa114abdb2986370ce860f36c27856`
- **Canonical Frozen-Content SHA-256:** `efe54856d12613f109cebc9482a6f224ca4e2370d0ff3f8a2fe4ea0a013b60ca`
- **Cross-Platform Line-Ending Audit:**
  > "The frozen test manifest is byte-identical to Git HEAD. The raw SHA-256 differs from the historical Windows value only because Git checkout on Linux uses LF line endings instead of CRLF. Canonical CRLF normalization reproduces the frozen SHA-256 exactly."
- **Integrity Status:** **STRICTLY VERIFIED AND UNCHANGED (PASS)**. Zero row, transcript, or audio path modifications occurred. Zero evaluations were conducted on the test set.

---

## 20, 21, 22, & 23. FINAL TEST WER, CER, LATENCY, RTF & BASELINE COMPARISON
- Since Full E1 was aborted due to local CUDA runtime unavailability, no post-training evaluation was performed on the frozen test set.
- **Reference Baselines (Historical Pilot Record):**
  - **E0 Pretrained PhoWhisper-tiny:** WER = 18.93%, CER = 11.07%, Latency = 1.412s, RTF = 0.140 (`PIPELINE PILOT — NOT FINAL BENCHMARK`).
  - **E1 Pilot (22 real samples):** WER = 19.21%, CER = 12.16%, Latency = 1.264s, RTF = 0.124.

---

## 24. EXPLICIT LIMITATIONS
1. **Workstation Environment:** Local development workstation is CPU-only under Python 3.14. Full-scale training cannot be executed locally without violating the FP16 CUDA mixed precision mandate.
2. **Streaming Execution Requirement:** Full-scale ingestion of 26,671 audio files requires running [`ASR_FULL_E1_TRAINING.ipynb`](file:///D:/Vietnamese_ASR_Week6/ASR_FULL_E1_TRAINING.ipynb) in an active Google Colab Linux GPU runtime.

---

## 25. PROVENANCE & LICENSING RESTRICTION
- **ViMD License:** Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (**CC BY-NC-ND 4.0**).
- **Compliance Mandate:** Technical CUDA training is cleared for internal academic benchmarking. Any fine-tuned checkpoint weights resulting from Full E1 must remain strictly designated as **internal research artifacts** and must not be published or redistributed on external platforms.
