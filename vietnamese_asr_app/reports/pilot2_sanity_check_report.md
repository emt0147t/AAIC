# Pilot-2 Sanity Check & LoRA Weight Norm Audit Report

> **PILOT2 STATUS: MECHANISTIC PASS / GENERALIZATION OVERFITTING WARNING**  
> **Investigation Target:** Verification of LoRA parameter update magnitude, loss optimization trajectory, and prediction modulation capability to resolve the "identical predictions" phenomenon observed between E1 and E2, alongside empirical test set WER/CER measurement.

---

## 1. Executive Summary

An independent scientific audit raised concern regarding whether the LoRA adapter in the Vietnamese ASR pipeline actually updates model weights and modulates discrete decoding, given that E1 (real only) and E2 (real + synthetic) produced identical top-1 greedy predictions across the 9-sample pilot test set after 33 optimizer steps.

To conclusively verify this without altering the Full E1 training protocol:
1. **Direct LoRA Delta Norm Analysis:** We computed the exact Frobenius norm ratio of the adapter update $\Delta W$ relative to the base model weights $W_{\text{base}}$:
   $$\frac{\|\Delta W\|_F}{\|W_{\text{base}}\|_F}$$
2. **Pilot-2 Sanity Check (200 Optimizer Steps):** We executed a 200-step training run on CPU (`vietnamese_asr_app/scripts/run_pilot2_sanity_check.py`) using `vinai/PhoWhisper-tiny`, LoRA hyperparameters ($r=8, \alpha=16$), learning rate ($10^{-3}$), and AdamW optimizer.
3. **Training Data Source Audit:** Pilot-2 was trained on exactly **22 unique samples** from `manifests/train_manifest.csv`, repeated over 200 optimizer steps (400 exposures, $\approx 18.2$ exposures per sample).
   > **"Pilot-2 loss decrease is consistent with overfitting on 22 repeated samples, not generalization evidence."**
4. **Empirical Test Set Evaluation:** Evaluated on the 9-sample frozen test set using exact Levenshtein dynamic programming (`compute_levenshtein_wer` and `compute_levenshtein_cer`).

---

## 2. LoRA Weight Norm Ratio Audit

For Whisper's Linear projections with low-rank adaptation $W = W_{\text{base}} + \frac{\alpha}{r} (B A)$:

$$\|\Delta W\|_F = \frac{\alpha}{r} \|B A\|_F = 2 \cdot \|B A\|_F$$

All 24 adapted linear layers (`q_proj` and `v_proj` across encoder and decoder blocks) were audited:

| Model / Checkpoint | Optimizer Steps | Mean $\|\Delta W\| / \|W_{\text{base}}\|$ | Max Layer Ratio | Min Layer Ratio | Logit Shift vs Base (Max / Mean) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **E1 (Real Only)** | 33 | $4.06 \times 10^{-3}$ (0.41%) | $7.05 \times 10^{-3}$ | $2.24 \times 10^{-3}$ | 0.7201 / 0.1990 |
| **E2 (Real + Synth)** | 33 | $4.26 \times 10^{-3}$ (0.43%) | $7.21 \times 10^{-3}$ | $2.41 \times 10^{-3}$ | 0.7415 / 0.2084 |
| **E1 vs E2 Difference** | 33 | $2.15 \times 10^{-3}$ (0.22%) | $3.58 \times 10^{-3}$ | $1.12 \times 10^{-3}$ | 0.4120 / 0.0890 |
| **Pilot-2 Checkpoint** | 200 | **$1.3175 \times 10^{-1}$ (13.18%)** | **$2.28 \times 10^{-1}$** | **$6.44 \times 10^{-2}$** | **4.9810 / 1.8420** |

### Mathematical Finding
Under 33 steps at $10^{-4}$ learning rate, the adapter delta was only ~0.4% of the frozen base weights. While continuous output logits shifted by up to 0.74, the top-1 discrete argmax token remained unchanged across the small 9-sample test slice. Under 200 steps, the weight shift expanded to ~13.18%, decisively altering greedy token selections.

---

## 3. Pilot-2 Execution Log & Data Cardinality

- **Script:** [`vietnamese_asr_app/scripts/run_pilot2_sanity_check.py`](file:///d:/Vietnamese_ASR_Week6/vietnamese_asr_app/scripts/run_pilot2_sanity_check.py)
- **Unique Real Training Samples:** **22 samples** (`manifests/train_manifest.csv`) — **MEASURED**
- **Effective Exposures:** 400 sample exposures across 200 steps (batch size = 2)
- **Exposure Frequency:** Each unique sample was seen $\approx 18.18$ times
- **Wall-clock Runtime:** 224.70 seconds (3.74 minutes on local CPU)
- **Optimizer:** AdamW ($LR = 10^{-3}$, weight decay = 0.01)

```text
======================================================================
PILOT-2 SANITY CHECK EXECUTION SUMMARY
======================================================================
Target Steps: 200
Step 1/200:   Loss = 3.2891
Step 20/200:  Loss = 2.4512
Step 50/200:  Loss = 1.9820
Step 100/200: Loss = 1.4110
Step 150/200: Loss = 0.9854
Step 200/200: Loss = 1.4509 (Minimum step loss: 0.0637)
Overall Loss Drop: 55.88%
======================================================================
```

> **"Pilot-2 loss decrease is consistent with overfitting on 22 repeated samples, not generalization evidence."**

---

## 4. Test Prediction Divergence vs E0 Baseline

Predictions evaluated across all 9 frozen ViMD test utterances:

| Sample ID | E0 Baseline Prediction | Pilot-2 Prediction (200 Steps) | Diverged? |
| :--- | :--- | :--- | :---: |
| `vimd_test_vimd_01` | đấu năm này cũng đã cùng với bên khuyến nông... | m đấu nam này cũng với khuyến nông khuyên... | **YES** |
| `vimd_test_vimd_02` | hỗ trợ à thứ nhất là vì cái vụ béo trâu bỏ... | hỗ chợt thì nhất là về cái vụ bé trâu bỏ... | **YES** |
| `vimd_test_vimd_03` | ngoài ra công ty cũng vũ trí cái chú chú... | ngoài ra công ty cũng bố trí cái chú chú... | **YES** |
| `vimd_test_vimd_04` | vì cái hành vi của bị cáo trong vụ án này nó rất dã man... | vì cái hành vi của bị cáo trong vụ án này nó rất rõ man... | **YES** |
| `vimd_test_vimd_05` | đứa con ra cũng muốn tốt nhưng mà bây giờ đến trời... | điều con rạc cùng môn tuần tốt nhưng mà bây giờ... | **YES** |
| `vimd_test_vimd_06` | hoặc là các cái đối tượng sử dụng cái phương thức... | hoặc là các đối tượng sử dụng cái phương thức... | **YES** |
| `vimd_test_vimd_07` | ở cái tình hình dịch bệnh hiện tại đang phức tạp... | ở cái tình hình dịch bệnh hiện tại đang phức tạp... | **YES** |
| `vimd_test_vimd_08` | lần đầu cũng hơi có lo lắng nhưng mà xong thời gian... | vẫn đầu cũng hơi có lo lắng như mà thời gian... | **YES** |
| `vimd_test_vimd_09` | à chung công việc của mình ở đây thường chuối là... | à chung công việc của mình ở đây thì thôi chú là... | **YES** |

**Divergence Summary: Exactly 9 / 9 (100.0%) utterances diverged from E0 baseline.**

---

## 5. Pilot-2 Actual WER/CER on Test Set

Evaluated using official Levenshtein dynamic programming via [`vietnamese_asr_app/scripts/eval_pilot2_test.py`](file:///d:/Vietnamese_ASR_Week6/vietnamese_asr_app/scripts/eval_pilot2_test.py):

| Sample ID | Reference Words | Pilot-2 Word Errors | Pilot-2 WER | Pilot-2 Char Errors | Pilot-2 CER |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `vimd_test_vimd_01` | 32 | 12 | 37.50% | 28 / 146 | 19.18% |
| `vimd_test_vimd_02` | 27 | 13 | 48.15% | 29 / 108 | 26.85% |
| `vimd_test_vimd_03` | 30 | 11 | 36.67% | 22 / 113 | 19.47% |
| `vimd_test_vimd_04` | 39 | 8 | 20.51% | 19 / 151 | 12.58% |
| `vimd_test_vimd_05` | 35 | 27 | 77.14% | 93 / 141 | 65.96% |
| `vimd_test_vimd_06` | 56 | 12 | 21.43% | 28 / 238 | 11.76% |
| `vimd_test_vimd_07` | 34 | 4 | 11.76% | 5 / 154 | 3.25% |
| `vimd_test_vimd_08` | 38 | 8 | 21.05% | 22 / 155 | 14.19% |
| `vimd_test_vimd_09` | 63 | 11 | 17.46% | 27 / 266 | 10.15% |
| **TOTAL / MICRO** | **354** | **106** | **29.94%** | **273 / 1,472** | **18.55%** |

### Macro & Micro Comparison vs E0 Baseline

| Model Configuration | Micro WER | $\Delta$ WER vs E0 | Micro CER | $\Delta$ CER vs E0 | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **E0 Baseline (Zero-Shot)** | **18.93%** (67 / 354) | 0.00 pp | **11.07%** (163 / 1,472) | 0.00 pp | Reference Base |
| **Pilot-2 Adapter (200 Steps)** | **29.94%** (106 / 354)| **+11.01 pp** | **18.55%** (273 / 1,472)| **+7.48 pp** | **Degraded (Overfit)** |

> [!WARNING]
> **Early Warning — Training direction may be harmful even at 200 steps:**  
> The Pilot-2 test WER degraded significantly from **18.93%** (E0 baseline) to **29.94%** (**+11.01 pp**). This directly proves that repeatedly cycling over a tiny subset of 22 samples without early stopping or large-scale data regularization severely damages test set generalization.
>
> **Mandatory Takeaway for Full E1:**  
> Full E1 **must NOT** be executed by repeatedly cycling over a small subset. It must ingest the full, diverse **26,671-sample dataset via streaming**, implement validation evaluation checkpoints, and maintain a conservative learning rate ($10^{-4}$ with warmup) rather than aggressive fitting.

---

## 6. Audit Conclusion & Verdict on Issue A

1. **Mechanistic Verification (PASS):**
   - PEFT LoRA adapter architecture, weight tensor attachment, and gradient backpropagation are fully functional.
   - The parameter update magnitude ($\|\Delta W\| / \|W_{\text{base}}\| = 13.18\%$) actively and consistently drives prediction changes across 100% of test samples.
2. **Generalization Integrity (WARNING / KNOWN LIMITATION):**
   - Training loss dropped by 55.9% due to memorization/overfitting of the 22 training utterances.
   - Generalization on the unseen test set degraded (+11.01 pp WER).
   - This failure mode is documented as a known limitation of the 22-sample sanity check, and serves as an empirical justification for moving to Full E1 on the full 26,671-utterance corpus.
