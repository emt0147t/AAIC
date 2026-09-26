# E2 LoRA Controlled Pilot Experiment Report

> [!NOTE]
> **Status**: `E2 PILOT: PASSED`  
> **Benchmark Label**: `PIPELINE PILOT — NOT FINAL BENCHMARK`  
> Results reflect an isolated proof-of-concept pipeline over a controlled pilot subset. Do not interpret as final benchmark or claim generalization.

---

## 1. Research Question
> *"Under an identical LoRA optimization budget, does changing the training-data composition to include QC-verified synthetic Vietnamese speech change ASR performance?"*

**Core Protocol Clarification**:
This experiment uses a fixed optimization budget. E2 changes the composition of the training exposures by introducing synthetic speech.
Do NOT state or assume that "22 real + 22 synthetic samples were all trained each epoch."

## 2. Exact Resolved E2 Configuration
```json
{
  "experiment_id": "E2_lora_real_synth_tiny",
  "base_model_id": "vinai/PhoWhisper-tiny",
  "mode": "train_and_eval",
  "seed": 42,
  "output_dir": "checkpoints/E2_lora_real_synth",
  "lora": {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": [
      "q_proj",
      "v_proj"
    ],
    "bias": "none"
  },
  "training": {
    "batch_size": 2,
    "gradient_accumulation_steps": 1,
    "learning_rate": 0.0001,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "epochs": 3,
    "lr_scheduler_type": "cosine",
    "max_grad_norm": 1.0,
    "mixed_precision": "fp16"
  },
  "data": {
    "train_manifest": "manifests/train_manifest.csv",
    "val_manifest": "manifests/val_manifest.csv",
    "test_manifest": "manifests/test_manifest.csv",
    "max_train_samples": 22,
    "max_val_samples": 5,
    "max_test_samples": 9,
    "quarantine_manifests": [
      "manifests/test_manifest.csv"
    ],
    "synthetic_manifest": "manifests/synthetic_qc_passed.csv",
    "synthetic_ratio": 0.3182
  },
  "training_budget": {
    "total_train_samples": 22,
    "real_sample_count": 15,
    "synthetic_sample_count": 7,
    "real_ratio": 0.6818,
    "synthetic_ratio": 0.3182,
    "micro_batch_size": 2,
    "gradient_accumulation_steps": 1,
    "effective_batch_size": 2,
    "fixed_optimizer_steps_per_epoch": 11,
    "total_optimizer_steps": 33,
    "sample_exposures_per_epoch": 22,
    "total_sample_exposures": 66
  }
}
```

## 3. Data Composition
- **Total Real Training Pool**: 22 clean real speech samples (10 VIVOS + 12 ViMD clean train)
- **Total Synthetic Pool**: 22 QC-passed synthetic utterances (Gwen-TTS 0.6B)
- **Per-Epoch Exposures**: 22 exposures (15 real + 7 synthetic)
- **Composition Ratio**: `68.18% real` / `31.82% synthetic` (~30% synthetic)
- **Sampling Policy**: Sampling without replacement per epoch; deterministic under `seed = 42`.

## 4. Per-Epoch Real / Synthetic Exposure Breakdown
- **Epoch 1**: 15 real exposures + 7 synthetic exposures = 22 total exposures
- **Epoch 2**: 15 real exposures + 7 synthetic exposures = 22 total exposures
- **Epoch 3**: 15 real exposures + 7 synthetic exposures = 22 total exposures

## 5. Total Optimizer Steps & Optimization Budget
- **Epochs**: 3
- **Micro Batch Size**: 2
- **Gradient Accumulation Steps**: 1
- **Effective Batch Size**: 2
- **Steps per Epoch**: 11
- **Total Optimizer Steps**: 33 (Target: 33, exactly matching E1)
- **Total Sample Exposures**: 66 (Target: 66, exactly matching E1)

## 6. LoRA Parameter Counts & Trainable Ratio
- **Base Model**: `vinai/PhoWhisper-tiny`
- **Total Parameters**: 37,908,096
- **Trainable LoRA Parameters**: 147,456
- **Frozen Parameters**: 37,760,640
- **Trainable Ratio**: `0.3890%`
- **Target Modules**: `q_proj`, `v_proj` (24 linear projection layers)

## 7. Training Loss Trajectory
| Epoch | Step | Global Step | Train Loss |
| :--- | :--- | :--- | :--- |
| 1 | 1 | 1 | 7.5897 |
| 1 | 2 | 2 | 5.7653 |
| 1 | 3 | 3 | 11.2162 |
| 1 | 4 | 4 | 4.4924 |
| 1 | 5 | 5 | 5.9987 |
| 1 | 6 | 6 | 3.8511 |
| 1 | 7 | 7 | 4.4920 |
| 1 | 8 | 8 | 1.0621 |
| 1 | 9 | 9 | 4.9647 |
| 1 | 10 | 10 | 5.6482 |
| 1 | 11 | 11 | 5.5610 |
| 2 | 1 | 12 | 2.4475 |
| 2 | 2 | 13 | 10.2351 |
| 2 | 3 | 14 | 4.7795 |
| 2 | 4 | 15 | 6.9089 |
| 2 | 5 | 16 | 5.6145 |
| 2 | 6 | 17 | 9.9677 |
| 2 | 7 | 18 | 10.8336 |
| 2 | 8 | 19 | 2.0144 |
| 2 | 9 | 20 | 2.8893 |
| 2 | 10 | 21 | 1.3979 |
| 2 | 11 | 22 | 1.3831 |
| 3 | 1 | 23 | 10.2492 |
| 3 | 2 | 24 | 5.7040 |
| 3 | 3 | 25 | 5.2140 |
| 3 | 4 | 26 | 6.7758 |
| 3 | 5 | 27 | 10.5794 |
| 3 | 6 | 28 | 2.1418 |
| 3 | 7 | 29 | 2.8801 |
| 3 | 8 | 30 | 4.5130 |
| 3 | 9 | 31 | 1.6987 |
| 3 | 10 | 32 | 4.4679 |
| 3 | 11 | 33 | 4.4436 |

### Epoch Average Train Losses:
- Epoch 1: `5.5128`
- Epoch 2: `5.3156`
- Epoch 3: `5.3334`

## 8. Validation Loss Trajectory
- Epoch 1: `11.1825`
- Epoch 2: `10.6740`
- Epoch 3: `10.5687`

## 9. Total Training Duration
- **Duration**: `48.86 seconds` (`0.81 minutes`)

## 10. Measured Peak CUDA Memory
- **Peak CUDA Allocated**: `0.0 MB` (`0 bytes`)
- **Peak CUDA Reserved**: `0.0 MB` (`0 bytes`)
- **GPU Name**: `CPU`
- **CUDA Version**: `N/A (CPU execution)`

## 11. E2 Test Word Error Rate (WER)
- **Micro WER**: `19.21%` (9 samples, 354 words)
- Label: `PIPELINE PILOT — NOT FINAL BENCHMARK`

## 12. E2 Test Character Error Rate (CER)
- **Micro CER**: `12.16%` (1472 chars)
- Label: `PIPELINE PILOT — NOT FINAL BENCHMARK`

## 13. E2 Mean Latency
- **Mean Inference Latency**: `1.389 seconds`

## 14. E2 Mean Real-Time Factor (RTF)
- **Mean RTF**: `0.137`

## 15. E0 vs E1 vs E2 Controlled Comparison
| Metric | E0 Baseline (Pretrained) | E1 LoRA (Real Speech Only) | E2 LoRA (Real + Synthetic) | Delta (E2 - E1) | Delta (E2 - E0) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Micro WER** | `18.93%` | `19.21%` | `19.21%` | `+0.00%` | `+0.28%` |
| **Micro CER** | `11.07%` | `12.16%` | `12.16%` | `+0.00%` | `+1.09%` |
| **Mean Latency** | `1.412 s` | `1.264 s` | `1.389 s` | `+0.125 s` | `-0.023 s` |
| **Mean RTF** | `0.140` | `0.124` | `0.137` | `+0.013` | `-0.003` |

> [!NOTE]
> Label: `PIPELINE PILOT — NOT FINAL BENCHMARK`  
> Descriptive comparison only. No claims of statistical significance or generalization from this 9-sample pilot.

### Per-Utterance Breakdown:
| Sample ID | Ref Words | WER (%) | CER (%) | Latency (s) | RTF |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `vimd_test_vimd_01` | 32 | 28.12% | 26.03% | 1.872 | 0.203 |
| `vimd_test_vimd_02` | 27 | 44.44% | 25.93% | 1.132 | 0.154 |
| `vimd_test_vimd_03` | 30 | 36.67% | 17.70% | 1.024 | 0.081 |
| `vimd_test_vimd_04` | 39 | 7.69% | 5.96% | 1.167 | 0.137 |
| `vimd_test_vimd_05` | 35 | 34.29% | 19.86% | 1.127 | 0.137 |
| `vimd_test_vimd_06` | 56 | 8.93% | 6.30% | 1.961 | 0.153 |
| `vimd_test_vimd_07` | 34 | 2.94% | 0.65% | 1.213 | 0.095 |
| `vimd_test_vimd_08` | 38 | 18.42% | 13.55% | 1.224 | 0.149 |
| `vimd_test_vimd_09` | 63 | 12.70% | 7.14% | 1.776 | 0.127 |

## 16. Checkpoint Reload Verification
- **Fresh Base Model Reload**: SUCCESS (`vinai/PhoWhisper-tiny`)
- **Adapter Mount**: SUCCESS (`checkpoints/E2_lora_real_synth`)
- **Inference Probe**: Sample `vimd_test_vimd_01`
- **Probe Hypothesis**: `đấu năm này cũng đã cùng với bên khuyến nông khuyến nông và kết hợp bên ủy ban đã chỉ đạo để bên khuyến nông khuyến nông khuyến nông thường trường tuyên t`
- **Runtime Exceptions**: None (0)
- **Reload Verification Status**: `PASSED`

## 17. Manifest Checksums & Cryptographic Hashes
- `manifests/train_manifest.csv`: `47a5aa1630713ca183f55a49d9e271d9839731fd3634069e0fe55c0b47540edb`
- `manifests/synthetic_qc_passed.csv`: `21ce3ad0b72ab70734e357a50d35eb1d37d527601a648904f1517afdc2883254`
- `manifests/val_manifest.csv`: `f4b22a21e91b7b061530cf3b330f47cda98ea01671a570c2b21814ffb543cc11`
- `manifests/test_manifest.csv`: `efe54856d12613f109cebc9482a6f224ca4e2370d0ff3f8a2fe4ea0a013b60ca`
- Git Commit: `a2dced84868f5988b3c8e783301f814064d393ca`

## 18. Provenance and Leakage Audit
- **Synthetic $\cap$ Test Transcript Overlap**: 0 (0%)
- **Synthetic $\cap$ Val Transcript Overlap**: 0 (0%)
- **Synthetic Speaker IDs $\cap$ Real Speaker IDs**: 0 (0%)
- **Lineage**: All 7 synthetic exposures per epoch originate strictly from QC-passed synthetic utterances generated from `train_manifest.csv`.

## 19. Limitations & Controlled Scope
1. **Sample Size**: Fixed to a 22-exposure budget per epoch over 3 epochs (33 steps) to ensure strictly identical compute between E1 and E2.
2. **Evaluation Domain**: 9 official ViMD gold test samples. Results demonstrate proof-of-concept pipeline feasibility and do not represent generalized benchmark claims.
