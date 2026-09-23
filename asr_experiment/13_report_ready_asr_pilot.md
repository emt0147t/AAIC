# 13 — Report-Ready ASR Engineering Pilot Summary

> **CLASSIFICATION**: **ENGINEERING PILOT — NOT FINAL BENCHMARK**  
> **Evaluation Constraint**: Pilot WER and CER metrics measure code execution integrity and basic forward/backward gradient pathways. They **must not** be interpreted as baseline model accuracy, model ranking, or benchmark performance claims.

---

## 1. Pilot Objectives & Operational Scope

The Gate 6 pilot served as an end-to-end engineering verification test before committing GPU hours or performing large-scale training:
1. Validate dataset stream collation and batching across two distinct data sources (VIVOS and ViMD).
2. Validate software resampling from 44.1 kHz to 16.0 kHz on the fly.
3. Validate Whisper tokenizer padding and label masking (`-100`).
4. Validate gradient backpropagation, parameter updates, and optimizer stepping on CPU.
5. Verify model checkpoint serialization and reloading.
6. Verify greedy autoregressive decoding on held-out validation speech.
7. **Strictly protect test manifests**: Zero test samples were used for training, hyperparameter tuning, or validation sanity checks.

---

## 2. Experimental Setup & Training Parameters

| Parameter | Value | Notes |
|:---|:---|:---|
| **Model** | `vinai/PhoWhisper-tiny` | Whisper encoder-decoder architecture |
| **Total Parameters** | 37,760,640 | 37,184,640 trainable |
| **Hardware** | Intel x86_64 CPU | Zero GPU used; PyTorch 2.10.0+cpu |
| **Training Epochs** | 1 epoch | Engineering verification only |
| **Training Utterances** | 100 total | 50 VIVOS train + 50 ViMD train (1,351.6 s audio) |
| **Validation Utterances** | 20 total | ViMD valid split (346.0 s audio) |
| **Random Seed** | 42 | Fixed for shuffling and weight init |
| **Batch Size** | 2 | Micro-batch size per forward step |
| **Gradient Accumulation** | 2 steps | Effective batch size = 4 utterances |
| **Total Optimizer Steps** | 25 steps | 50 forward batches |
| **Optimizer** | AdamW | Learning rate = $1 \times 10^{-4}$, weight decay = 0.01 |
| **Gradient Clipping** | Max norm 1.0 | Stable backprop |
| **Training Runtime** | **68.26 seconds** | CPU execution |
| **Peak Working Memory** | ~4.69 GB RAM | No OOM encountered |
| **Initial Loss** | **1.1522** | Batch 1 |
| **Final Loss** | **0.3679** | Batch 25 (strictly finite throughout) |
| **Checkpoint Status** | PASS | Saved to `artifacts/pilot/checkpoint/` and reloaded |

---

## 3. Five-Sample Qualitative Validation Sanity Check

> **Data Source**: Strictly from the **ViMD validation split**. Primary test (ViMD test) and secondary reference test (VIVOS test) were completely untouched.

All metrics are labeled: **ENGINEERING PILOT — NOT FINAL BENCHMARK**.

| Sample ID | Split | Ref Words | Hyp Words | Pilot WER | Pilot CER | Ground Truth Reference (Normalized) | Hypothesis Prediction (Normalized) |
|:---|:---:|:---:|:---:|:---:|:---:|:---|:---|
| `vimd_val_000` | ViMD valid | 44 | 51 | 0.5000 | 0.3448 | tôi đọc những cái tư liệu đầu tiên về chiến lịch sử của thập vạn đại sơn thì cũng qua những các tạp chí lịch sử và nhân chứng ngày xưa khi tôi mới làm báo thì tôi bắt đầu đọc những bài kia | tôi đọc và đọc và những tư liệu đầu tiên để chiến lịch sử tập vật đại sôn thì cũng qua những tư tri kỳ tạp tri í lịch sử và nhân chứng về xưa thì tôi bị làm báo thì tôi bị bắt đầu đọc tình đi bãi mạng kia |
| `vimd_val_001` | ViMD valid | 26 | 32 | 0.8462 | 0.6500 | đã nghe các cụ ví dụ là ông vũ xuân toàn hoặc là bác lâm ngọc thủ kể lại những chuyện về thập vạn đại sơn | đã góp nhe các cụ giữ nổi thế này nồng vương xuân toàn và mắc lâm ngọc thụ kể lại những thứ tư tư truyền về thuận vãn tài thơm tế thơm |
| `vimd_val_002` | ViMD valid | 47 | 67 | 0.5319 | 0.4133 | sau này thì tôi mới có dịp tìm hiểu thêm các tài liệu khác ví dụ là những tài liệu về các tướng lĩnh quân đội và nhà xuất bản quân đội nhân dân cũng đã xuất bản là ai là danh nhân quân sự việt nam | sau này từ tôi mới có dịp tìm hiểu thêm tất cả thủ tục tại tài liệu khác ví dụ là là lợi tại lợi từ tại tại lượt khác ví dụ là là là những tài liệu về cao tướng lĩnh quân đội và nhà thất bản quân đội nhân dân cũng đã xuất bản là những ý ai là doanh nhân quân sự việt nam |
| `vimd_val_003` | ViMD valid | 85 | 77 | 0.6706 | 0.5421 | trong những năm qua và những năm tới đây thì bộ đội biên phòng cũng rất là nhiều công việc ngoài công việc bảo vệ an ninh chính trị trật tự an toàn xã hội cũng như công tác tuần tra bảo vệ biên giới thì đội biên phòng còn có nhiệm vụ cũng quan trọng nữa đó là đồng hành cùng với chính quyền địa phương và bà con nhân dân để phát triển kinh tế nhất là trong dịch covid những năm qua | và chào nữ năm qua và những năm tới đây tìm nguồn viên phòng thì cũng rất là nhiều vụ việc ngoại việt bảo vệ của mình từ từ tri ây tri ây tri ây tri ây còn có tưởng tức quan trọng nữa đó là đồng hành cùng với trường đại phương và bảo vệ tổng nhận dân việt phát triển kinh tế trên thời trang việt việt ba lê dịch của việt nguyên năm qua |
| `vimd_val_004` | ViMD valid | 66 | 70 | 0.6515 | 0.5354 | đối với đội biên phòngbộ đội biên phòng nói chung và đồn biên phòng đức long nói riêng thì cũng rất là nhiều cũng có rất là nhiều đồn có những mô hình mới để bà con phát triển kinh tế nhưng đối riêng với đồn đức long thì chủ yếu là giúp bà con về giống giúp đỡ bà con về phát triển chăn nuôi | và đối với nội viên phòng của thủy phòng võ trung và nội viên thì cũng rất là nhiều cũng có rất là nhiều nổi cong như mô hình mới để bà con phát triển kinh tế đối với nội viên đổ đêm long và chủ yếu là rối mạc con về giống nếu là những nơi nhớ những nơi nhớ những nơi nhớ những nỗi bảo con trăn nuôi |

### Summary Statistics of Pilot Check
- **Mean Pilot WER**: `0.6400`
- **Mean Pilot CER**: `0.4971`
- **Output Validity**: 5 / 5 outputs non-empty (100%).
- **Language Validity**: 5 / 5 outputs contain authentic Vietnamese vocabulary, syllables, and tone diacritics (100%).
- **Degeneracy / Crashes**: Zero NaNs, zero infinite values, zero CUDA/CPU runtime crashes.

---

## 4. Key Engineering Takeaways

1. **Pipeline Feasibility Confirmed**:
   - The end-to-end Python pipeline functions cleanly on constrained CPU compute.
   - Resampling from 44.1 kHz to 16.0 kHz runs reliably in-memory without corrupting utterance length.
2. **Deterministic Preprocessing**:
   - Applying Unicode NFC normalization and removing punctuation before Levenshtein computation eliminates spurious errors caused by divergent comma/period annotations between read and broadcast transcripts.
3. **Readiness for Full Training**:
   - The training script, data collators, loss logging, and checkpointing logic are verified and ready to scale to GPU training once formal approval and resources are allocated.
