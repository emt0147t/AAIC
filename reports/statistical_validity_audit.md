# Statistical Validity & Test Partition Audit Report

> **B STATUS: PARTIAL**  
> **Investigation Target:** Resolution of statistical power limitations of the 9-sample test set ($n=9$) via 1,000-sample bootstrap confidence intervals, official test set expansion to the 2,026-sample ViMD test set, and transparent reporting of local vs cloud execution status.

---

## 1. Executive Summary

An independent scientific audit correctly noted that an $n=9$ test set lacks sufficient statistical power for scientific claims:
- A single word error difference out of 354 reference words represents a shift of $\pm 0.28$ percentage points ($18.93\% \leftrightarrow 19.21\%$).
- Without empirical confidence intervals, differences between E0, E1, and E2 cannot be interpreted with statistical significance.

To resolve this issue:
1. **Bootstrap Confidence Intervals:** We calculated exact 95% confidence intervals using 1,000 empirical bootstrap resamples across the 9 test samples.
2. **Re-classification of the 9-Sample Set:** Formally designated the 9-sample set as a **Smoke Test / Pipeline Integrity Gate** only, explicitly barring it from serving as a final benchmark.
3. **Official Test Partition Expansion:** Expanded the authoritative test benchmark to the full 2,026-sample test split of `nguyendv02/ViMD_Dataset`.
4. **Local Execution Status Transparency:** Per scientific reporting principles, local execution on CPU is recorded as **`NOT_EXECUTED (LOCAL CPU WORKSTATION)`** due to the ~9.3-hour CPU inference time. A turnkey evaluation script (`vietnamese_asr_app/scripts/evaluate_vimd_full_test.py`) has been constructed for immediate execution on Cloud GPU.

---

## 2. 1,000-Bootstrap 95% Confidence Interval Analysis (9-Sample Pilot)

From 1,000 bootstrap resamples on the 9 test utterances (354 words, 1,472 characters):

| Experiment | Metric | Point Estimate | 95% Confidence Interval (Bootstrap 1,000) | CI Width |
| :--- | :--- | :---: | :---: | :---: |
| **E0 Baseline** | WER | 18.93% | **[12.01%, 28.18%]** | 16.17 pp |
| **E1 LoRA Real** | WER | 19.21% | **[12.47%, 28.27%]** | 15.80 pp |
| **E2 LoRA Real+Synth** | WER | 19.21% | **[12.47%, 28.27%]** | 15.80 pp |
| **E0 Baseline** | CER | 11.07% | **[7.28%, 15.93%]** | 8.65 pp |
| **E1 LoRA Real** | CER | 12.16% | **[7.63%, 18.14%]** | 10.51 pp |
| **E2 LoRA Real+Synth** | CER | 12.16% | **[7.63%, 18.14%]** | 10.51 pp |

### Key Scientific Insight:
The observed delta between E0 and E1 (+0.28 pp WER, +1.09 pp CER) is completely enveloped by the ~16 percentage-point 95% confidence interval. This rigorously confirms that the 9-sample set cannot distinguish between these model states.

---

## 3. Official Test Partition Restructuring

To establish statistical validity, the evaluation framework is formally partitioned into two distinct tiers:

```
+-------------------------------------------------------------------------+
|                         EVALUATION FRAMEWORK                            |
+-------------------------------------------------------------------------+
                                    |
          +-------------------------+-------------------------+
          |                                                   |
          v                                                   v
+-----------------------------+           +-----------------------------+
|    TIER 1: SMOKE TEST       |           |   TIER 2: OFFICIAL BENCHMARK|
|  (9-Sample Frozen ViMD)     |           |    (2,026-Sample ViMD Test) |
+-----------------------------+           +-----------------------------+
| * Size: 9 samples, 93.7s    |           | * Size: 2,026 samples, ~11h |
| * Role: Pipeline preflight, |           | * Role: Final benchmark,    |
|   I/O validation, code CI   |           |   statistical comparisons   |
| * Statistical Claims: NONE  |           | * Statistical Claims: VALID |
+-----------------------------+           +-----------------------------+
```

---

## 4. Execution Status & Cloud GPU Protocol

- **Local Workstation Status:** `NOT_EXECUTED (LOCAL CPU WORKSTATION)`  
  *Rationale:* Sequential CPU decoding of 2,026 utterances requires approximately $2,026 \times 16.5\,\text{s} \approx 9.3\,\text{hours}$, plus downloading ~15 GB of multi-shard parquet files to local disk.
- **Turnkey Cloud GPU Script:** [`vietnamese_asr_app/scripts/evaluate_vimd_full_test.py`](file:///d:/Vietnamese_ASR_Week6/vietnamese_asr_app/scripts/evaluate_vimd_full_test.py)  
  *Protocol:*
  ```bash
  # Execute on Google Colab / Tesla T4 (~15-20 minutes total runtime)
  python vietnamese_asr_app/scripts/evaluate_vimd_full_test.py --adapter checkpoints/full_e1_checkpoint
  ```
