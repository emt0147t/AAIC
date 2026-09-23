# 09 — Tiny ASR Engineering Pilot Report (Gate 6)

> **CLASSIFICATION**: **ENGINEERING PILOT — NOT FOR FINAL BENCHMARK**  
> **Status**: Completed successfully on CPU. Test manifests kept strictly untouched.

---

## 1. Executive Summary

The objective of Gate 6 was to execute a **tiny ASR engineering pilot** to verify the end-to-end training and inference execution path for the frozen Vietnamese ASR protocol before undertaking any full-scale experiment.

- **Model**: `vinai/PhoWhisper-tiny` (37,760,640 total parameters, 37,184,640 trainable).
- **Execution Hardware**: Intel CPU (PyTorch `2.10.0+cpu`, zero GPU used, zero OOM).
- **Pilot Data**: Exactly **100 training utterances** (50 VIVOS train + 50 ViMD train) and **20 validation utterances** (ViMD valid split). Test sets were **strictly untouched**.
- **1-Epoch Training**: Completed in **68.26 seconds** (25 optimizer updates, batch size 2, gradient accumulation 2, AdamW, $\text{lr}=10^{-4}$).
- **Loss Convergence**: Initial loss $1.1522 \rightarrow$ final loss $0.3679$. Loss was strictly finite (no NaN, no Inf) at all steps.
- **Checkpoint**: Full model weights and processor configuration successfully saved to disk and reloaded.
- **Inference Verification**: 5 validation samples qualitatively inspected. All generated outputs are non-empty, grammatically plausible Vietnamese text with proper tone marks and diacritics.

---

## 2. Environment & Hardware Specifications

| Component | Specification |
|:---|:---|
| **Operating System** | Windows 11 |
| **Python Version** | 3.14.0 |
| **PyTorch Version** | 2.10.0+cpu |
| **Hugging Face Transformers** | 4.57.6 |
| **Hardware** | Multi-core x86_64 CPU |
| **CUDA Available** | `False` (CPU execution only) |
| **Peak Working Set Memory** | ~4.69 GB RAM |
| **GPU Used** | `False` |

---

## 3. Model Architecture & Parameters

| Attribute | Value |
|:---|:---|
| **Model Name** | `vinai/PhoWhisper-tiny` |
| **Base Architecture** | OpenAI Whisper (encoder-decoder sequence-to-sequence) |
| **Pretrained Specialization** | Fine-tuned on Vietnamese speech by VinAI Research |
| **Total Parameters** | 37,760,640 |
| **Trainable Parameters** | 37,184,640 |
| **Non-Trainable Parameters** | 576,000 (positional embeddings) |
| **Vocabulary Size** | 51,865 |
| **Target Language & Task** | Language: Vietnamese (`vi`), Task: Transcribe (`transcribe`) |

---

## 4. Dataset Composition & Sampling

In accordance with the frozen ASR protocol:

| Split | Dataset Source | Utterance Count | Total Audio Duration | Role |
|:---|:---|:---:|:---:|:---|
| **Train (Subset 1)** | VIVOS train (`thanhduycao/vivos_ng_only`) | 50 | 315.1 s (~5.25 min) | Read academic baseline speech |
| **Train (Subset 2)** | ViMD train (`nguyendv02/ViMD_Dataset`) | 50 | 1,036.5 s (~17.28 min) | Multi-dialect broadcast speech |
| **Total Pilot Train** | **VIVOS + ViMD (50/50 balance)** | **100** | **1,351.6 s (~22.53 min)** | **1-epoch engineering training** |
| **Validation** | ViMD valid (`nguyendv02/ViMD_Dataset`) | 20 | 346.0 s (~5.77 min) | Validation tuning/inspection |
| **Primary Test** | ViMD test | — | — | **UNTOUCHED (frozen benchmark)** |
| **Secondary Reference** | VIVOS test | — | — | **UNTOUCHED (frozen reference)** |

- **Random Seed**: Fixed at `42` for all data shuffling and weight initialization.
- **Data Integrity**: Audio was decoded from raw byte streams; original repositories and files were not modified in-place.

---

## 5. Audio Pipeline Verification

1. **Decoding**: Raw PCM byte streams decoded into NumPy arrays using `scipy.io.wavfile`.
2. **Channel Format**: Mono channel enforced (if multi-channel, averaged across channel axis).
3. **Resampling**:
   - **VIVOS**: Native sample rate is **16,000 Hz**. No resampling needed.
   - **ViMD**: Native sample rate is **44,100 Hz**. Resampled to **16,000 Hz** using `librosa.resample`.
   - **Verification**: Original duration ($24.24\text{ s}$) and resampled duration ($24.24\text{ s}$) match exactly.
4. **Featurization**:
   - 80-channel log-mel spectrogram calculated by `WhisperProcessor.feature_extractor` with standard Whisper FFT parameters (hop size 160, FFT window 400).
   - **Input Feature Tensor Shape**: `[80 mel bins, 3000 time frames]` per utterance (zero-padded/truncated to 30.0 seconds).

---

## 6. Transcript Pipeline & Pre-Training Inspection

### Deterministic Normalization Policy
The raw transcript (`sentence_raw`) is preserved unaltered. Deterministic normalization (`sentence_norm`) applies:
1. Unicode NFC normalization (`unicodedata.normalize('NFC', text)`).
2. Lowercasing.
3. Punctuation removal via regex (`[^\w\s]`), preserving all Vietnamese accented characters and tone marks.
4. Collapsing repeated whitespace to a single space.

### Pre-Training Inspection of 20 Transcript Examples (10 VIVOS + 10 ViMD)

| ID | Dataset | Casing | Punctuation | Word Count | Char Count | Normalized Preview |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | VIVOS | UPPERCASE | No | 25 | 107 | có trường hợp như gia đình bà nguyễn... |
| 2 | VIVOS | UPPERCASE | No | 24 | 97 | ông nguyễn văn châu giám đốc sở y t... |
| 3 | VIVOS | UPPERCASE | No | 20 | 92 | nhiệm vụ cụ thể của ông nguyễn quốc... |
| 4 | VIVOS | UPPERCASE | No | 14 | 69 | hiện nguyễn dương cũng đang làm đạo... |
| 5 | VIVOS | UPPERCASE | No | 15 | 66 | như trần hưng đạo lịch sử tôn thất ... |
| 6 | VIVOS | UPPERCASE | No | 15 | 70 | nghe bạn gọi nguyễn văn huy nhảy xu... |
| 7 | VIVOS | UPPERCASE | No | 11 | 48 | vỉa hè đường nguyễn hữu cầu phải đà... |
| 8 | VIVOS | UPPERCASE | No | 13 | 62 | giới thiệu triển lãm tranh tĩnh vật... |
| 9 | VIVOS | UPPERCASE | No | 27 | 123 | những lúc ấy bạn nào cũng buồn nhưn... |
| 10 | VIVOS | UPPERCASE | No | 22 | 101 | nguyễn ngọc trung bên tượng bán thâ... |
| 11 | ViMD | Mixed | Yes | 62 | 292 | nghiên cứu học tậpcác ứng dụng các ... |
| 12 | ViMD | Mixed | Yes | 83 | 363 | chuyển đổi số hiện nay đang được là... |
| 13 | ViMD | Mixed | Yes | 83 | 368 | trong quá trình mình làm việc để qu... |
| 14 | ViMD | Mixed | Yes | 64 | 299 | ý tưởng đầu tiên tôi phải nhắc đến ... |
| 15 | ViMD | Mixed | Yes | 67 | 297 | giúp giải quyết đa dạng các nội dun... |
| 16 | ViMD | Mixed | Yes | 35 | 147 | em rất là tự hào vì năm đầu tiên em... |
| 17 | ViMD | Mixed | Yes | 67 | 269 | em cảm thấy rất là tự hào và qua cá... |
| 18 | ViMD | Mixed | No | 83 | 370 | năm nay thì cao trương tôi thì cũng... |
| 19 | ViMD | Mixed | Yes | 76 | 321 | em thưa chị là em cảm thấy rất là v... |
| 20 | ViMD | Mixed | Yes | 77 | 320 | qua thực tế thì cán bộ công nhân vi... |

#### Inspection Analysis
- **Empty Transcripts**: 0 / 20.
- **Malformed Characters / Encoding Errors**: None detected. Unicode NFC normalization validated across all strings.
- **Casing Difference**: VIVOS is natively 100% UPPERCASE read prompts; ViMD is mixed-case broadcast speech transcripts with natural punctuation.
- **Length Distribution**: VIVOS utterances are concise (mean 18.6 words, range 11–27 words); ViMD broadcast segments are substantially longer (mean 69.7 words, range 35–83 words).

---

## 7. Training Run Configuration & Loss Progression

- **Epochs**: 1
- **Batch Size**: 2
- **Gradient Accumulation Steps**: 2 (Effective Batch Size: 4)
- **Optimizer**: AdamW (`lr = 1e-4`, `weight_decay = 0.01`)
- **Gradient Clipping**: Maximum norm 1.0
- **Label Masking**: Padding tokens masked to `-100` so they do not contribute to cross-entropy loss.
- **Total Optimizer Steps**: 25
- **Total Runtime**: 68.26 seconds

### Step-by-Step Training Log

| Optimizer Step | Forward Batch | Average Loss | Is Finite? | Learning Rate | Elapsed Time (s) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 2 | 1.1522 | True | 0.0001 | 3.53 |
| 2 | 4 | 1.2848 | True | 0.0001 | 6.15 |
| 3 | 6 | 0.6947 | True | 0.0001 | 8.54 |
| 4 | 8 | 0.4290 | True | 0.0001 | 11.14 |
| 5 | 10 | 0.4607 | True | 0.0001 | 13.67 |
| 6 | 12 | 0.3160 | True | 0.0001 | 16.32 |
| 7 | 14 | 0.4860 | True | 0.0001 | 19.15 |
| 8 | 16 | 0.2769 | True | 0.0001 | 21.73 |
| 9 | 18 | 0.3539 | True | 0.0001 | 24.40 |
| 10 | 20 | 0.1265 | True | 0.0001 | 27.05 |
| 11 | 22 | 0.2211 | True | 0.0001 | 29.45 |
| 12 | 24 | 0.3559 | True | 0.0001 | 32.17 |
| 13 | 26 | 1.0316 | True | 0.0001 | 34.81 |
| 14 | 28 | 0.2811 | True | 0.0001 | 37.27 |
| 15 | 30 | 0.4336 | True | 0.0001 | 39.91 |
| 16 | 32 | 0.4392 | True | 0.0001 | 42.72 |
| 17 | 34 | 0.5275 | True | 0.0001 | 45.56 |
| 18 | 36 | 0.4676 | True | 0.0001 | 48.28 |
| 19 | 38 | 0.3781 | True | 0.0001 | 51.09 |
| 20 | 40 | 0.3883 | True | 0.0001 | 54.19 |
| 21 | 42 | 0.2844 | True | 0.0001 | 56.71 |
| 22 | 44 | 0.1207 | True | 0.0001 | 59.34 |
| 23 | 46 | 0.4762 | True | 0.0001 | 62.45 |
| 24 | 48 | 0.5433 | True | 0.0001 | 65.54 |
| 25 | 50 | 0.3679 | True | 0.0001 | 68.26 |

---

## 8. Checkpoint Verification

- **Export Path**: `artifacts/pilot/checkpoint/`
- **Files Saved**: `model.safetensors`, `config.json`, `generation_config.json`, `tokenizer.json`, `preprocessor_config.json`, `vocab.json`, `merges.txt`.
- **Reload Verification**: Successfully loaded back into memory using `WhisperForConditionalGeneration.from_pretrained()`. Verified exact parameter count of 37,760,640.

---

## 9. Qualitative Inference Sanity Check (5 Validation Samples)

> **Important**: Evaluated strictly on the **ViMD validation split** (`vimd_val_000` to `vimd_val_004`). Neither the primary test set (ViMD test) nor the secondary reference test set (VIVOS test) was used.

| Sample ID | Ref Words | Hyp Words | Pilot WER | Pilot CER | Qualitative Observations |
|:---|:---:|:---:|:---:|:---:|:---|
| `vimd_val_000` | 44 | 51 | 0.5000 | 0.3448 | Captures major phrases ("tôi đọc", "tư liệu đầu tiên", "chiến lịch sử", "nhân chứng ngày xưa", "làm báo") with minor regional dialect substitution. |
| `vimd_val_001` | 26 | 32 | 0.8462 | 0.6500 | Proper names ("vũ xuân toàn", "lâm ngọc thủ") partially transcribed phonetically. Non-empty, fully Vietnamese. |
| `vimd_val_002` | 47 | 67 | 0.5319 | 0.4133 | Accurately reconstructs complex phrase "sau này thì tôi mới có dịp tìm hiểu thêm các tài liệu khác ví dụ là... quân đội và nhà xuất bản quân đội nhân dân". |
| `vimd_val_003` | 85 | 77 | 0.6706 | 0.5421 | Long spontaneous broadcast sentence. Successfully decodes temporal anchor "những năm qua và những năm tới đây" and "đồng hành cùng với... phát triển kinh tế". |
| `vimd_val_004` | 66 | 70 | 0.6515 | 0.5354 | Broadcast dialect speech. Captures "bà con phát triển kinh tế", "giúp đỡ bà con", "chăn nuôi". Repetitions observed in tail frames. |

### Aggregate Pilot Metrics
- **Mean Pilot WER**: `0.6400`
- **Mean Pilot CER**: `0.4971`
- **Non-Empty Output Rate**: 5 / 5 (100%)
- **Vietnamese Language Adherence**: 5 / 5 (100% Vietnamese vocabulary with diacritics)
- **Status**: `ENGINEERING PILOT — NOT FOR FINAL BENCHMARK`

---

## 10. Failures Handled & Engineering Lessons

1. **Large Monolithic Parquet Timeout**:
   - Initial streaming from community mirrors containing 350–700 MB monolithic files timed out over high-latency CDN requests.
   - **Resolution**: Used verified lightweight parquet mirror `thanhduycao/vivos_ng_only` (10.8 MB total) for VIVOS and canonical `nguyendv02/ViMD_Dataset` train/valid splits.
2. **ViMD Sample Rate Incompatibility**:
   - Verified that ViMD native audio is 44,100 Hz. Feeding 44.1 kHz directly into Whisper's 16 kHz feature extractor would corrupt temporal durations by $2.75\times$.
   - **Resolution**: Resampling pipeline (`librosa.resample`) cleanly converts 44.1 kHz $\rightarrow$ 16.0 kHz, preserving exact utterance duration.
3. **CPU Execution Stability**:
   - Training Whisper-tiny on CPU with batch size 2 and gradient accumulation 2 achieved stable ~1.36s forward+backward steps with zero out-of-memory errors and finite loss throughout.

---

## 11. Pilot Limitations

1. **Small Data Sample**: 100 utterances (~22.5 minutes) is designed solely to verify gradient updates, data loaders, resampling, tokenization, loss tracking, checkpointing, and beam/greedy decoding. It is **not** sufficient for production accuracy.
2. **Single Epoch**: A single epoch does not allow the model to adapt to multi-dialect acoustic variations.
3. **No Benchmark Claim**: The pilot WER (~64%) reflects zero-shot / single-epoch behavior on challenging spontaneous provincial Vietnamese speech; it must not be cited as the model's benchmark capability.
