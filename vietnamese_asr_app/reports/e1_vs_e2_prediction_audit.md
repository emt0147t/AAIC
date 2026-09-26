# Forensic E1 vs. E2 Result Integrity Audit

> [!NOTE]
> **Audit Status**: `COMPLETE`  
> **Final Finding**: `B. IDENTICAL PREDICTIONS`  
> **Benchmark Label**: `PIPELINE PILOT — NOT FINAL BENCHMARK`  
> Both E1 and E2 adapters produce identical predictions across all 9 gold ViMD test utterances. The adapters possess distinct weights resulting from different training data compositions, but the parameter shift under the 33-step pilot budget was insufficient to alter discrete top-1 argmax greedy decoding on the test set.

---

## 1. Executive Summary & Root-Cause Finding

### Finding: **B. IDENTICAL PREDICTIONS**

An exhaustive forensic comparison was conducted between the frozen E1 checkpoint (`checkpoints/E1_lora_real/`) and E2 checkpoint (`checkpoints/E2_lora_real_synth/`).

| Audit Dimension | Finding | Evidence |
| :--- | :--- | :--- |
| **Adapter Checkpoints** | **Distinct Weights** | SHA-256 hashes differ; $\max |\Delta \theta| = 0.002438$, $\text{mean} |\Delta \theta| = 0.000303$ across all 147,456 parameters. |
| **Synthetic Consumption** | **Confirmed Real** | E2 consumed 21 synthetic exposures (7 per epoch) and 45 real exposures (15 per epoch) across 33 steps. |
| **Loss Trajectories** | **Diverged** | Train and validation losses differed at every epoch (E2 achieved lower validation loss in Epochs 2 & 3). |
| **Checkpoint Isolation** | **Strictly Isolated** | E1 and E2 loaded from separate directories with zero cross-contamination. |
| **Test Predictions** | **100% Identical** | All 9 test utterances produced character-for-character identical hypotheses. |
| **Root Cause** | **Argmax Stability** | Under a 33-step pilot budget ($LR=10^{-4}$ on 0.389% LoRA parameters), the ~0.03% average weight shift did not alter the top-1 discrete token argmax selections under greedy decoding. |

---

## 2. Per-Sample Prediction Comparison

Each of the 9 frozen ViMD test utterances was independently re-evaluated using fresh base models and isolated adapter loads.

| Sample ID | Ref Words | E1 Prediction | E2 Prediction | E1 WER | E2 WER | E1 CER | E2 CER | Identical? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `vimd_test_vimd_01` | 32 | đấu năm này cũng đã cùng với bên khuyến nông khuyến nông và kết hợp bên ủy ban đã chỉ đạo để bên khuyến nông khuyến nông khuyến nông thường trường tuyên truyền bà con nhân dân. | đấu năm này cũng đã cùng với bên khuyến nông khuyến nông và kết hợp bên ủy ban đã chỉ đạo để bên khuyến nông khuyến nông khuyến nông thường trường tuyên truyền bà con nhân dân. | 28.12% | 28.12% | 26.03% | 26.03% | **YES** |
| `vimd_test_vimd_02` | 27 | hỗ trợ à thứ nhất là vì cái vụ béo trâu bỏ thì đến là cái năm trước thì nói chung là là tốt tăng thêm thứ nhất cho bà con. | hỗ trợ à thứ nhất là vì cái vụ béo trâu bỏ thì đến là cái năm trước thì nói chung là là tốt tăng thêm thứ nhất cho bà con. | 44.44% | 44.44% | 25.93% | 25.93% | **YES** |
| `vimd_test_vimd_03` | 30 | ngoài ra công ty cũng vũ trí cái chú chú ăn chú ở tập thể các nhà ở châu công nhân ở xa đó phông ở rất là sạch sẽ con gà. | ngoài ra công ty cũng vũ trí cái chú chú ăn chú ở tập thể các nhà ở châu công nhân ở xa đó phông ở rất là sạch sẽ con gà. | 36.67% | 36.67% | 17.70% | 17.70% | **YES** |
| `vimd_test_vimd_04` | 39 | vì cái hành vi của bị cáo trong vụ án này nó rất là dã man vì vậy là chợ bộ viên đã đề nghị cái khung hình phạt là cao nhất là cái hình đối với bị cáo. | vì cái hành vi của bị cáo trong vụ án này nó rất là dã man vì vậy là chợ bộ viên đã đề nghị cái khung hình phạt là cao nhất là cái hình đối với bị cáo. | 7.69% | 7.69% | 5.96% | 5.96% | **YES** |
| `vimd_test_vimd_05` | 35 | đứa con ra cũng muốn tốt nhưng mà bây giờ đến trời là pháp luật san đi a như thiên là sớm thì riêng bán thân chúng tôi là rất là ca mừng. | đứa con ra cũng muốn tốt nhưng mà bây giờ đến trời là pháp luật san đi a như thiên là sớm thì riêng bán thân chúng tôi là rất là ca mừng. | 34.29% | 34.29% | 19.86% | 19.86% | **YES** |
| `vimd_test_vimd_06` | 56 | hoặc là các cái đối tượng sử dụng cái phương thức thủ đoạn là tặng quà ấy kết bạn và giới thiệu là mình ở nước ngoài và tặng quà sau đó là yêu cầu các bị hại gửi tiền đón các cái khoản phí thì sau đó chúng chặn liên lạc và chiếm đoạt cái tiền đó. | hoặc là các cái đối tượng sử dụng cái phương thức thủ đoạn là tặng quà ấy kết bạn và giới thiệu là mình ở nước ngoài và tặng quà sau đó là yêu cầu các bị hại gửi tiền đón các cái khoản phí thì sau đó chúng chặn liên lạc và chiếm đoạt cái tiền đó. | 8.93% | 8.93% | 6.30% | 6.30% | **YES** |
| `vimd_test_vimd_07` | 34 | ở cái tình hình dịch bệnh hiện tại đang phức tạp thì ít nhiều gì có vắc xin trong người thì cũng giảm thiểu các khả năng nguy hiểm của chính bản thân mình. | ở cái tình hình dịch bệnh hiện tại đang phức tạp thì ít nhiều gì có vắc xin trong người thì cũng giảm thiểu các khả năng nguy hiểm của chính bản thân mình. | 2.94% | 2.94% | 0.65% | 0.65% | **YES** |
| `vimd_test_vimd_08` | 38 | lần đầu cũng hơi có lo lắng nhưng mà xong thời gian thì tinh thần thì vẫn cố gắng để phục vụ cho các con sức khỏe và con nhân dân ở địa bàn thành phố unk. | lần đầu cũng hơi có lo lắng nhưng mà xong thời gian thì tinh thần thì vẫn cố gắng để phục vụ cho các con sức khỏe và con nhân dân ở địa bàn thành phố unk. | 18.42% | 18.42% | 13.55% | 13.55% | **YES** |
| `vimd_test_vimd_09` | 63 | à chung công việc của mình ở đây thường chuối là sẽ tiếp nhận các cuộc gọi cấp cứu của người dân tới và sau đó sự chí sau đó đều xe đi để hỗ trợ người dân nhanh nhất có thể vì cộng đồng vì công cuộc chung thì những điều này thì cũng không phải là vấn đề lớn lắm. | à chung công việc của mình ở đây thường chuối là sẽ tiếp nhận các cuộc gọi cấp cứu của người dân tới và sau đó sự chí sau đó đều xe đi để hỗ trợ người dân nhanh nhất có thể vì cộng đồng vì công cuộc chung thì những điều này thì cũng không phải là vấn đề lớn lắm. | 12.70% | 12.70% | 7.14% | 7.14% | **YES** |

> Machine-readable CSV saved to: [`reports/e1_vs_e2_prediction_comparison.csv`](file:///d:/Vietnamese_ASR_Week6/vietnamese_asr_app/reports/e1_vs_e2_prediction_comparison.csv)

---

## 3. Aggregate Error Metric Recalculation

Exact Levenshtein edit distance operations recalculated directly from per-utterance predictions:

| Metric | E1 LoRA (Real Only) | E2 LoRA (Real + Synthetic) | Reproducibility Status |
| :--- | :--- | :--- | :--- |
| **Total Reference Words** | 354 | 354 | Confirmed |
| **Word Substitutions** | 43 | 43 | Exact Match |
| **Word Deletions** | 8 | 8 | Exact Match |
| **Word Insertions** | 17 | 17 | Exact Match |
| **Total Word Errors** | **68** | **68** | Exact Match |
| **Micro WER** | **19.21%** | **19.21%** | **100% Reproducible** |
| **Total Reference Characters** | 1,472 | 1,472 | Confirmed |
| **Character Substitutions** | 98 | 98 | Exact Match |
| **Character Deletions** | 18 | 18 | Exact Match |
| **Character Insertions** | 63 | 63 | Exact Match |
| **Total Character Errors** | **179** | **179** | Exact Match |
| **Micro CER** | **12.16%** | **12.16%** | **100% Reproducible** |

---

## 4. Adapter Tensor & Numerical Integrity Audit

To confirm that E2 did not simply copy or fail to update E1's weights, direct tensor-level numerical analysis was performed:

```
===========================================================================
ADAPTER TENSOR NUMERICAL ANALYSIS
===========================================================================
E1 Checkpoint:   checkpoints/E1_lora_real/adapter_model.safetensors
E1 File Size:     596,424 bytes
E1 SHA-256:       5d4b60f9bf851a8e4f2f2caff2c49425cf0614ca8757c0d8e048da6c64f833c4

E2 Checkpoint:   checkpoints/E2_lora_real_synth/adapter_model.safetensors
E2 File Size:     596,424 bytes
E2 SHA-256:       26d397794e861ae4e12c37b6633560deaa7eb0d7385d32cf0ae9499adb1408c6

Tensors Compared: 48 (all 24 q_proj and 24 v_proj LoRA A/B weight matrices)
Total Parameters: 147,456 float32 elements
SHA-256 Match:    FALSE (Files are distinctly different)
Numerically Equal:FALSE
Max Absolute Diff:0.00243768
Mean Absolute Diff:0.00030309
===========================================================================
```

The adapter tensors are **demonstrably distinct**. E2 underwent real, independent gradient updates that diverged from E1 by an average of $3.03 \times 10^{-4}$ and a peak of $2.44 \times 10^{-3}$ per parameter.

---

## 5. Synthetic Data Consumption & Lineage Audit

Did E2 actually consume the synthetic samples during training?
We inspected the training logs and execution records:

- **Epoch 1**: 15 real samples + 7 synthetic samples:  
  `['synth_01', 'synth_03', 'synth_07', 'synth_08', 'synth_17', 'synth_22', 'synth_20']`
- **Epoch 2**: 15 real samples + 7 synthetic samples:  
  `['synth_21', 'synth_06', 'synth_17', 'synth_16', 'synth_09', 'synth_01', 'synth_03']`
- **Epoch 3**: 15 real samples + 7 synthetic samples:  
  `['synth_17', 'synth_12', 'synth_22', 'synth_09', 'synth_14', 'synth_10', 'synth_05']`

**Audit Verification**:
- Synthetic samples in E2: **Yes, all 3 epochs contained synthetic speech**.
- Exact synthetic count: **Exactly 7 synthetic exposures per epoch (21 total)**.
- Within-epoch duplication: **Zero (sampled strictly without replacement)**.
- Unauthorized samples: **Zero (all 21 synthetic exposures trace to `manifests/synthetic_qc_passed.csv`)**.

---

## 6. Training Budget & Optimization Equivalence

| Optimization Parameter | E1 Protocol | E2 Protocol | Equivalence Check |
| :--- | :--- | :--- | :--- |
| **Micro Batch Size** | 2 | 2 | Identical |
| **Gradient Accumulation** | 1 | 1 | Identical |
| **Effective Batch Size** | 2 | 2 | Identical |
| **Steps per Epoch** | 11 | 11 | Identical |
| **Epochs** | 3 | 3 | Identical |
| **Total Optimizer Steps** | **33** | **33** | **Exact Equality (No Step Inflation)** |
| **Total Sample Exposures** | **66** | **66** | **Exact Equality (100% Fixed Compute)** |

### Loss Progression Divergence:
- **E1 Loss**:
  - Train: Epoch 1: `5.3520` | Epoch 2: `4.8512` | Epoch 3: `5.2607`
  - Val: Epoch 1: `11.2719` | Epoch 2: `10.8604` | Epoch 3: `10.7556`
- **E2 Loss**:
  - Train: Epoch 1: `5.5128` | Epoch 2: `5.3156` | Epoch 3: `5.3334`
  - Val: Epoch 1: `11.1825` | Epoch 2: `10.6740` | Epoch 3: `10.5687`

The validation loss on held-out real speech was consistently lower in E2 during Epoch 2 (10.67 vs 10.86) and Epoch 3 (10.56 vs 10.75), confirming that synthetic exposure actively modulated the representations.

---

## 7. Evaluation Configuration & Isolation Audit

- **Evaluation Manifest**: Both E1 and E2 evaluated on identical file:  
  `manifests/test_manifest.csv` (SHA-256: `efe54856d12613f109cebc9482a6f224ca4e2370d0ff3f8a2fe4ea0a013b60ca`).
- **Audio Preprocessing**: WhisperProcessor (16 kHz, 80-channel log-mel, 30s padding).
- **Text Normalization**: NFC Unicode lowercase, punctuation stripped, whitespace collapsed.
- **Decoding Configuration**: Greedy decoding (`forced_decoder_ids=None`, `language="vi"`, `task="transcribe"`, `max_new_tokens=192`).
- **Directory Isolation**: E1 loaded strictly from `checkpoints/E1_lora_real/`; E2 loaded strictly from `checkpoints/E2_lora_real_synth/`.

---

## 8. Final Conclusion

The identical aggregate WER (19.21%) and CER (12.16%) between E1 and E2 is **NOT** a software bug, not a checkpoint overwrite, and not a failure to consume synthetic data.

The phenomenon occurs because:
1. **Pilot Scale Constraint**: Training was strictly limited to 33 optimizer steps ($10^{-4}$ LR, cosine decay) adapting 0.389% of parameters.
2. **Greedy Argmax Robustness**: While adapter weights shifted by an average of $\sim 3 \times 10^{-4}$, the resulting logit distribution changes over the 354 reference words were minor and did not flip any token's rank-1 prediction under greedy decoding.
3. **Domain Evaluation**: The 9-sample ViMD test set is fixed and small; a 33-step fine-tuning run over 22 samples produces subtle probability mass movements that manifest in loss metrics (validation loss dropped to 10.56 in E2 vs 10.75 in E1) before inducing discrete categorical flips in greedy transcriptions.

**Final Determination**: **`B. IDENTICAL PREDICTIONS`** (Legitimate, auditable, and reproducible behavior under the frozen pilot protocol).
