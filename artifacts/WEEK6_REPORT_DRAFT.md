# Weekly Research Report Draft: Vietnamese ASR Experimental Foundation

**Author**: AI Student (Year 2)  
**Project**: Vietnamese Automatic Speech Recognition (ASR)  
**Period**: Week 6  
**Document Status**: Final Consolidated Report Draft  
**Primary Artifact**: `WEEK6_REPORT_DRAFT.md`

---

## 1. Objectives

The primary objective of this week's work was to establish a rigorous, clean, and reproducible experimental foundation for Vietnamese Speech Recognition under constrained compute and storage resources, avoiding unverified legacy dependencies.

### In Scope
1. **Audit Vietnamese Speech Datasets**: Systematically survey candidate corpora cited in prior reports, inspecting original repository metadata, licenses, split structures, and access pathways.
2. **Select and Combine Datasets**: Design reproducible data mixtures, separating human-verified (Gold), pseudo-labeled (Silver), and untranscribed data.
3. **Design and Freeze ASR Protocol**: Define an end-to-end specification covering acoustic models, feature extraction, deterministic transcript normalization, disjoint evaluation splits, and contamination policies.
4. **Reconstruct and Visualize Audio Augmentations**: Reconstruct the audio augmentation primitives described in the weekly report (MixMatch / ReMixMatch context), clearly separating time-domain signal manipulation from feature-space spectrogram transforms.
5. **Execute Tiny Engineering Pilot**: Implement a 1-epoch CPU pilot pipeline to verify data streaming, resampling, gradient updates, checkpoint serialization, and decoding integrity.

### Out of Scope
- Full-scale multi-epoch ASR model training.
- Running long unattended GPU compute jobs.
- Evaluating on frozen test sets (test manifests remained completely untouched).
- Resolving the legacy Language Identification (LID) experiment.

---

## 2. Dataset Survey

Seven candidate Vietnamese speech datasets mentioned in historical weekly reports and literature were audited:

| Dataset | Claimed Hours | Intended Domain | Speech Style | Transcript Provenance | Original Source / Access |
|:---|:---:|:---|:---|:---|:---|
| **VIVOS** | 15.67h | Academic / General | Scripted read speech | Human (scripted prompts) | AILAB VNUHCM / Zenodo 7068130 |
| **FOSD** | ~30.0h | General conversational | Scripted read speech | Human (FPT staff recordings) | FPT Corporation / Mendeley Data |
| **ViMD** | 102.56h | Multi-dialect news | Spontaneous broadcast | Human (EMNLP 2024 paper) | VinAI Research / Hugging Face |
| **VietMed** | ~16h (labeled) | Healthcare | Doctor-patient conversations | Human (medical transcripts) | Duckhai Le et al. / Hugging Face |
| **VietSuperSpeech** | 103h–300h+ | News, vlogs, podcasts | Spontaneous conversation | Silver (Zipformer pseudo-labels) | thanhnew2001 / Hugging Face |
| **VLSP 2021 (Transcribed)** | ~280h | Broad domain | Varied / competition | Human (competition grade) | VLSP Consortium / vlsp.org.vn |
| **VLSP 2021 (Untranscribed)** | ~400h | Broadcast / general | Varied speech | None (unlabeled audio) | VLSP Consortium / vlsp.org.vn |

---

## 3. Dataset Verification

Authoritative sources, dataset cards, and repository metadata were directly inspected to establish verifiable facts:

1. **VIVOS**:
   - Authoritative archival record: Zenodo record `7068130` (1.47 GB `vivos.tar.gz`).
   - Hugging Face repository `AILAB-VNUHCM/vivos` contains legacy dataset loading scripts that fail under modern `datasets` versions; community parquet mirrors (`thanhduycao/vivos_ng_only`) provide identical data in lightweight streaming format.
2. **FOSD**:
   - Hosted on Mendeley Data (`doi:10.17632/k9sxg2twv4.4`), authored by FPT Corporation.
   - Contains 25,921 recordings (~30.0 hours) stored as **MP3 files** (not uncompressed WAV), accompanied by UTF-8 plain-text transcript files.
3. **ViMD**:
   - Canonical repository is `nguyendv02/ViMD_Dataset` (Hugging Face), associated with the EMNLP 2024 paper *"ViMD: A Vietnamese Multi-Dialect Speech Dataset"*.
   - Stored in 130 parquet files (~74 GB total with embedded audio). Publicly accessible without gated registration.
4. **VietMed**:
   - Canonical repository is `leduckhai/VietMed` (Hugging Face) and associated institutional Google Drive.
   - Audio is encoded in **OGG format**, covering doctor-patient interactions with ICD-10 medical diagnostic codes.
5. **VietSuperSpeech**:
   - Hosted at `thanhnew2001/VietSuperSpeech` (Hugging Face). Manifests use Icefall JSON format.
   - Audio originates from 4 YouTube channels: `nguoivietdailynews`, `nguyenkhangofficial`, `trinhlieu`, and `vietcetera`.
6. **VLSP 2021**:
   - Hosted on `vlsp.org.vn`. Access requires formal institutional registration and execution of the VLSP data sharing agreement.

---

## 4. Data Quality, Provenance, and Licensing

### Master Data Audit Table

| Dataset | Quality Tier | Verified Hours | Utterance Count | Speaker Metadata | Official Splits | Verified License | Storage Format |
|:---|:---:|:---:|:---:|:---|:---|:---|:---|
| **VIVOS** | **GOLD** | 15.67h | 12,420 | 65 speakers (46 train / 19 test) | Train (11,660) / Test (760) | CC BY-NC-SA 4.0 | 16 kHz WAV mono |
| **ViMD** | **GOLD** | 102.56h | 18,949 | 12,955 speakers across 63 provinces | Train (15,023) / Valid (1,900) / Test (2,026) | CC BY-NC-ND 4.0 | 44.1 kHz WAV in parquet |
| **FOSD** | **GOLD** | ~30.0h | 25,921 | Inconsistent / limited | None (single pool) | FPT Public License | Compressed MP3 |
| **VietMed** | **GOLD** | ~16.0h | ~13,000 | Limited (role/gender) | Train / Valid / Test | MIT | OGG |
| **VietSuperSpeech** | **SILVER** | ~103h–300h+ | 32k–67k | None | Train / Dev (no independent test) | MIT (README) / Not Set | 16 kHz WAV mono |
| **VLSP 2021** | **GOLD/UNLAB** | 280h / 400h | Unknown | Unknown | Train / Dev / Eval | VLSP Agreement | 16 kHz WAV (expected) |

### Critical Licensing & Provenance Discoveries
- **FOSD License Correction [Verified Fact]**: FOSD is **not** licensed under CC BY 4.0. The official Mendeley record specifies the **FPT Public License**—a custom permissive license granting free, worldwide, irrevocable rights including commercial use, but explicitly excluding patent and trademark licenses and requiring copyright attribution.
- **VIVOS License [Verified Fact]**: The authoritative Zenodo record explicitly designates **CC BY-NC-SA 4.0** (NonCommercial, ShareAlike).
- **ViMD License & Redistribution Constraint [Unresolved Issue]**: ViMD is licensed under **CC BY-NC-ND 4.0** (NonCommercial, NoDerivatives). While internal academic training and evaluation are permitted, public distribution of fine-tuned model checkpoints may legally constitute a derivative work. Institutional clarification is required before open-sourcing fine-tuned weights.
- **VietSuperSpeech Version Discrepancy [Unresolved Issue]**: Three contradictory versions exist:
  - Hugging Face README: 32,267 samples / 103.18 hours / 3 sources.
  - arXiv Paper: 52,023 samples / 267.39 hours / 4 sources.
  - Current HF Commit `cbf624a`: 67,405 manifest samples / ~300+ hours.
  - *Decision*: Classified as `VERSION_UNRESOLVED`. Excluded from frozen training/evaluation sets.

---

## 5. Dataset Selection and Combination

Three data mixture configurations were analyzed:

- **CONFIG A (Read Speech Baseline — Feasible)**: VIVOS train (~14h) + FOSD subset (~15–30h). All Gold read speech (~29–44h). Limited dialect diversity; lacks official FOSD splits.
- **CONFIG B (General + Multi-Dialect — Selected Provisional)**: VIVOS train (~14h) + ViMD train (81.43h). Total **~95.4 hours Gold speech** (26,683 utterances). Combines read speech with spontaneous speech across 63 provinces. Fully disjoint speaker splits.
- **CONFIG C (Conversational Silver Mixture — Not Ready)**: VIVOS train (~14h Gold) + VietSuperSpeech (~300h Silver). Excluded because VietSuperSpeech has unresolved versions and circular pseudo-label contamination.

### Data Tier Partitioning
- **Primary Training Data (Gold)**: VIVOS train (11,660 utt) + ViMD train (15,023 utt) = 26,683 utterances (~95.4 hours).
- **Validation Data (Gold)**: ViMD valid (1,900 utt, 10.26 hours, 1,320 disjoint speakers).
- **Primary Benchmark Test Set (Gold)**: ViMD test (2,026 utt, 10.87 hours, 1,344 disjoint speakers).
- **Secondary Reference Test Set (Gold)**: VIVOS test (760 utt, 0.75 hours, 19 disjoint speakers).
- **Standby Unlabeled Pools (SSL)**: VLSP 2021 untranscribed (~400h) and VietMed unlabeled (~2,200h).

---

## 6. ASR Model Selection

To satisfy low-compute constraints while maximizing Vietnamese speech representation, models were evaluated based on parameter footprint, architectural stability, and pre-existing Vietnamese fine-tuning:

| Candidate Model | Architecture | Total Params | Trainable Params | License | Role |
|:---|:---|:---:|:---:|:---|:---|
| **PhoWhisper-tiny** (`vinai/PhoWhisper-tiny`) | Whisper Encoder-Decoder | 37.8M | 37.2M | Apache 2.0 | **Primary Candidate** |
| **PhoWhisper-base** (`vinai/PhoWhisper-base`) | Whisper Encoder-Decoder | 74.0M | 73.1M | Apache 2.0 | **Fallback Candidate** |
| **Whisper-small** (`openai/whisper-small`) | Whisper Encoder-Decoder | 241.7M | 241.7M | MIT | General Multilingual Baseline |
| **MMS-1B-all** (`facebook/mms-1b-all`) | wav2vec 2.0 / Conformer | ~1.0B | ~1.0B | CC BY-NC 4.0 | High-Compute Baseline (Deferred) |

*Decision*: **`vinai/PhoWhisper-tiny`** was selected as the primary model. It provides specialized Vietnamese BPE tokenization and competitive acoustic representations with minimal memory footprint (~150 MB weights), enabling stable CPU and single-GPU training.

---

## 7. Contamination Analysis

A systematic audit was conducted cross-referencing published model training corpora against all candidate evaluation datasets:

### Master Contamination Matrix

| Model \ Dataset | VIVOS | ViMD | FOSD | VLSP 2020 | VLSP 2021 | VietMed | VietSuperSpeech |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **PhoWhisper (all sizes)** | **CONFIRMED** | NEF | NEF | **CONFIRMED** | NEF | NEF | NEF |
| **OpenAI Whisper (base)** | NEF | NEF | NEF | NEF | NEF | NEF | NEF |
| **Meta MMS** | NEF | NEF | NEF | NEF | NEF | NEF | NEF |
| **Zipformer-30M-6000h** | **CONFIRMED** | NEF | **CONFIRMED** | **CONFIRMED** | **CONFIRMED** | **CONFIRMED** | **CIRCULAR\*** |
| **wav2vec2-vi-160h** | **CONFIRMED** | NEF | **CONFIRMED** | **CONFIRMED** | NEF | NEF | NEF |

*\*CONFIRMED = Confirmed in documented training pool. NEF = NO_EVIDENCE_FOUND in examined model documentation. CIRCULAR = Zipformer was the teacher model that generated VietSuperSpeech pseudo-labels.*

### Evaluation Implications
1. **ViMD Test Set Evaluation Status**:
   - ViMD was published at EMNLP 2024, postdating the training runs of PhoWhisper (arXiv:2406.02555), Zipformer, MMS, and Whisper.
   - Status: **NO_EVIDENCE_FOUND IN EXAMINED MODEL DOCUMENTATION** across all models.
   - It is the only candidate offering a genuine, independent out-of-domain evaluation benchmark.
2. **VIVOS Test Set Evaluation Status**:
   - Table 1 of the PhoWhisper paper explicitly includes VIVOS test (0.75h) in the training data mixture.
   - Status: **CONFIRMED_CONTAMINATED** for PhoWhisper-tiny, PhoWhisper-base, and Zipformer.
   - Rule: VIVOS test is strictly designated as **SECONDARY REFERENCE ONLY** to evaluate in-domain acoustic ceilings, never reported as an out-of-domain generalization metric.

---

## 8. Frozen / Provisional ASR Protocol

The formal ASR protocol is structured as follows (designated **PROVISIONAL** pending ViMD institutional licensing review):

```
================================================================================
                    FINAL ASR EXPERIMENTAL PROTOCOL SPECIFICATION
================================================================================
TASK                    : Vietnamese Speech-to-Text (ASR)
MODEL                   : vinai/PhoWhisper-tiny (Fallback: vinai/PhoWhisper-base)
TOKENIZER               : WhisperProcessor (language="vi", task="transcribe")
TRAINING DATA (GOLD)    : VIVOS train (~14h) + ViMD train (81.43h) = ~95.4h (26,683 utt)
VALIDATION DATA (GOLD)  : ViMD valid (10.26h, 1,900 utt, 1,320 disjoint speakers)
PRIMARY TEST (GOLD)     : ViMD test (10.87h, 2,026 utt, 1,344 disjoint speakers)
                          [NO_EVIDENCE_FOUND IN EXAMINED MODEL DOCUMENTATION]
SECONDARY TEST (GOLD)   : VIVOS test (0.75h, 760 utt) [CONFIRMED_CONTAMINATED]
AUDIO SPECIFICATION     : Single-channel mono, 16,000 Hz, float32 [-1.0, 1.0]
RESAMPLING PIPELINE     : VIVOS native (16 kHz) -> direct
                          ViMD native (44.1 kHz) -> resampled 44.1 kHz -> 16.0 kHz
FEATURE REPRESENTATION  : 80-channel log-mel spectrogram, input shape [80, 3000]
TRANSCRIPT POLICY       : Raw text preserved; deterministic normalization:
                          NFC Unicode, lowercasing, punctuation stripped, whitespace collapsed
METRICS & NORMALIZATION : WER and CER calculated via deterministic Levenshtein distance
                          after applying identical normalization to reference and hypothesis
OPTIMIZATION (FULL RUN) : AdamW, lr=1e-4 (linear decay), weight_decay=0.01, clip_norm=1.0
                          Effective batch size = 32 (micro-batch 8-16 + grad accum)
BENCHMARK INTEGRITY     : Test manifests FROZEN and UNTOUCHED during development
================================================================================
```

---

## 9. Augmentation Reconstruction

### Methodology Attribution
**The Gate 5 experiment was an augmentation reconstruction based on the weekly report methodology, not a reproduction of the historical MixMatch/ReMixMatch implementation.** The workspace contained no historical implementation scripts.

### Domain Separation Framework
Audio augmentations were categorized into their proper mathematical representations:

```
Discrete Audio Waveform (Time Domain)
   ├── Amplitude Perturbation (Gain scaling in [-6dB, +6dB])       --> amplitude.wav
   ├── Time Shifting (Circular rotation up to +/-100ms)           --> time_shift.wav
   └── Noise Injection (Additive Gaussian noise at 20dB SNR)      --> noise.wav

Log-Mel Spectrogram (Time-Frequency Representation)
   ├── Time Masking (SpecAugment: T=20 frames, n=2 masks)         --> time_mask.png (NO WAV)
   └── Frequency Masking (SpecAugment: F=15 mel bins, n=2 masks)  --> freq_mask.png (NO WAV)

Latent Space / Optimization (SSL Training Strategy)
   ├── MixUp (Convex linear interpolation of inputs and labels)   --> Loss-level formulation
   └── Augmentation Anchoring (Weak-view pseudo-label anchor)     --> Training loop logic
```

### Demonstration & Quality Audit
Transformations were evaluated on 3 Vietnamese speech samples from ViMD (native 44.1 kHz):

| Sample ID | Augmentation | Category | Parameter | Duration | Decodable? | Accidental Clipping? | Silent? |
|:---|:---|:---|:---|:---:|:---:|:---:|:---:|
| `sample_000` | Amplitude | Waveform | Gain = -1.506 dB | 9.239s | Yes | No | No |
| `sample_000` | Time Shift | Waveform | Shift = +64.85 ms | 9.239s | Yes | No | No |
| `sample_000` | Noise | Waveform | SNR = 20.0 dB | 9.239s | Yes | No | No |
| `sample_000` | Time Mask | Feature-Space | $T=20, n=2$ | 9.239s | N/A (Image) | N/A | N/A |
| `sample_000` | Freq Mask | Feature-Space | $F=15, n=2$ | 9.239s | N/A (Image) | N/A | N/A |
| `sample_001` | Amplitude | Waveform | Gain = -1.506 dB | 7.358s | Yes | No | No |
| `sample_001` | Time Shift | Waveform | Shift = +64.85 ms | 7.358s | Yes | No | No |
| `sample_001` | Noise | Waveform | SNR = 20.0 dB | 7.358s | Yes | No | No |
| `sample_001` | Time Mask | Feature-Space | $T=20, n=2$ | 7.358s | N/A (Image) | N/A | N/A |
| `sample_001` | Freq Mask | Feature-Space | $F=15, n=2$ | 7.358s | N/A (Image) | N/A | N/A |
| `sample_002` | Amplitude | Waveform | Gain = -1.506 dB | 3.651s | Yes | No | No |
| `sample_002` | Time Shift | Waveform | Shift = +64.85 ms | 3.651s | Yes | No | No |
| `sample_002` | Noise | Waveform | SNR = 20.0 dB | 3.651s | Yes | No | No |
| `sample_002` | Time Mask | Feature-Space | $T=20, n=2$ | 3.651s | N/A (Image) | N/A | N/A |
| `sample_002` | Freq Mask | Feature-Space | $F=15, n=2$ | 3.651s | N/A (Image) | N/A | N/A |

- **Quality Audit Pass**: 100% of the 9 generated WAV files decode cleanly, preserve exact duration ($\Delta t = 0.000\text{ s}$), contain zero NaNs/Infs, show no clipping (`clipped = False`), and maintain healthy signal levels (`silent = False`).
- **Feature-Space Integrity**: Time and frequency masking were exported strictly as spectrogram visualizations (`.png`). **Zero fake WAV files were fabricated**.

---

## 10. Tiny ASR Engineering Pilot

### Pilot Setup & Objectives
To verify the complete training, audio resampling, tokenization, gradient backpropagation, checkpointing, and inference pipeline, a tiny engineering pilot was executed in Gate 6.
- **Model**: `vinai/PhoWhisper-tiny` (37.8M params).
- **Compute**: CPU-only (`PyTorch 2.10.0+cpu`, Intel x86_64, zero GPU).
- **Data Scope**: Exactly **100 train utterances** (50 VIVOS + 50 ViMD; 1,351.6s audio) and **20 validation utterances** (ViMD valid; 346.0s audio).
- **Test Integrity**: Test manifests were strictly untouched.

### Pre-Training Transcript Inspection (20 Examples)
- Inspected 10 VIVOS samples (100% UPPERCASE prompts, mean word count 18.6) and 10 ViMD samples (100% mixed-case punctuated broadcast speech, mean word count 69.7).
- 0/20 empty transcripts. Unicode NFC normalization validated across all text.

### Training Progression
- **Configuration**: 1 epoch, batch size 2, gradient accumulation 2 (effective batch size 4), 25 optimizer updates, AdamW ($\text{lr} = 10^{-4}$).
- **Runtime**: **68.26 seconds** on CPU.
- **Loss Progression**: Loss dropped smoothly from **1.1522** (step 1) to **0.3679** (step 25). Loss remained strictly finite throughout.
- **Checkpoint**: Serialized to `artifacts/pilot/checkpoint/` and reloaded with exact parameter match (37,760,640).

### Five-Sample Qualitative Validation Check

> **CRITICAL LABEL**: **ENGINEERING PILOT — NOT FOR FINAL BENCHMARK**  
> Both reference and hypothesis strings underwent identical deterministic normalization before metric computation.

| Sample ID | Split | Ref Words | Hyp Words | Pilot WER | Pilot CER | Ground Truth Reference (Normalized) | Hypothesis Prediction (Normalized) |
|:---|:---:|:---:|:---:|:---:|:---:|:---|:---|
| `vimd_val_000` | ViMD val | 44 | 51 | 0.5000 | 0.3448 | tôi đọc những cái tư liệu đầu tiên về chiến lịch sử của thập vạn đại sơn thì cũng qua những các tạp chí lịch sử và nhân chứng ngày xưa khi tôi mới làm báo thì tôi bắt đầu đọc những bài kia | tôi đọc và đọc và những tư liệu đầu tiên để chiến lịch sử tập vật đại sôn thì cũng qua những tư tri kỳ tạp tri í lịch sử và nhân chứng về xưa thì tôi bị làm báo thì tôi bị bắt đầu đọc tình đi bãi mạng kia |
| `vimd_val_001` | ViMD val | 26 | 32 | 0.8462 | 0.6500 | đã nghe các cụ ví dụ là ông vũ xuân toàn hoặc là bác lâm ngọc thủ kể lại những chuyện về thập vạn đại sơn | đã góp nhe các cụ giữ nổi thế này nồng vương xuân toàn và mắc lâm ngọc thụ kể lại những thứ tư tư truyền về thuận vãn tài thơm tế thơm |
| `vimd_val_002` | ViMD val | 47 | 67 | 0.5319 | 0.4133 | sau này thì tôi mới có dịp tìm hiểu thêm các tài liệu khác ví dụ là những tài liệu về các tướng lĩnh quân đội và nhà xuất bản quân đội nhân dân cũng đã xuất bản là ai là danh nhân quân sự việt nam | sau này từ tôi mới có dịp tìm hiểu thêm tất cả thủ tục tại tài liệu khác ví dụ là là lợi tại lợi từ tại tại lượt khác ví dụ là là là những tài liệu về cao tướng lĩnh quân đội và nhà thất bản quân đội nhân dân cũng đã xuất bản là những ý ai là doanh nhân quân sự việt nam |
| `vimd_val_003` | ViMD val | 85 | 77 | 0.6706 | 0.5421 | trong những năm qua và những năm tới đây thì bộ đội biên phòng cũng rất là nhiều công việc ngoài công việc bảo vệ an ninh chính trị trật tự an toàn xã hội cũng như công tác tuần tra bảo vệ biên giới thì đội biên phòng còn có nhiệm vụ cũng quan trọng nữa đó là đồng hành cùng với chính quyền địa phương và bà con nhân dân để phát triển kinh tế nhất là trong dịch covid những năm qua | và chào nữ năm qua và những năm tới đây tìm nguồn viên phòng thì cũng rất là nhiều vụ việc ngoại việt bảo vệ của mình từ từ tri ây tri ây tri ây tri ây còn có tưởng tức quan trọng nữa đó là đồng hành cùng với trường đại phương và bảo vệ tổng nhận dân việt phát triển kinh tế trên thời trang việt việt ba lê dịch của việt nguyên năm qua |
| `vimd_val_004` | ViMD val | 66 | 70 | 0.6515 | 0.5354 | đối với đội biên phòngbộ đội biên phòng nói chung và đồn biên phòng đức long nói riêng thì cũng rất là nhiều cũng có rất là nhiều đồn có những mô hình mới để bà con phát triển kinh tế nhưng đối riêng với đồn đức long thì chủ yếu là giúp bà con về giống giúp đỡ bà con về phát triển chăn nuôi | và đối với nội viên phòng của thủy phòng võ trung và nội viên thì cũng rất là nhiều cũng có rất là nhiều nổi cong như mô hình mới để bà con phát triển kinh tế đối với nội viên đổ đêm long và chủ yếu là rối mạc con về giống nếu là những nơi nhớ những nơi nhớ những nơi nhớ những nỗi bảo con trăn nuôi |

- **Summary Stats**: Mean Pilot WER = `0.6400`, Mean Pilot CER = `0.4971`.
- **Engineering Observations**: 100% of outputs were non-empty and generated phonetically legitimate Vietnamese syllables with appropriate tone diacritics. Word error patterns stem from regional dialect vocabulary and unadapted acoustic weights after just 1 epoch.

---

## 11. Limitations

1. **Pilot Scope**: The 100-sample, 1-epoch pilot confirms software execution and backpropagation stability; it does not represent acoustic convergence or model benchmark performance.
2. **Compute Environment**: Work was performed strictly on CPU. GPU compute allocation is required for full ~95-hour training.
3. **Licensing**: ViMD's `CC BY-NC-ND 4.0` license precludes distributing modified datasets and requires legal clarification regarding public release of fine-tuned weights.
4. **Excluded Corpora**: VietSuperSpeech was excluded due to version ambiguity and circular teacher-model pseudo-labeling; FOSD was excluded due to compressed MP3 format and absence of official speaker-disjoint splits.

---

## 12. What Was NOT Done

To maintain rigorous experimental boundaries and control compute/token usage, the following were **deliberately not performed**:
- **No full ASR training** was initiated on the 95-hour training corpus.
- **No evaluation on primary or secondary test sets** was conducted (test manifests remain 100% untouched).
- **No hyperparameter sweep** (learning rate, weight decay, batch size) was attempted.
- **No large-scale dataset download** was executed (used small streaming slices only).
- **No historical MixMatch/ReMixMatch code** was re-run (labeled strictly as reconstruction).
- **No fake WAV files** were generated for spectrogram-space masking transforms.

---

## 13. Next Experiment

With the data audit, protocol design, augmentation reconstruction, and engineering pilot successfully validated, the roadmap for the next experimental phase is:

1. **Institutional License Sign-Off**: Confirm institutional interpretation of ViMD `CC BY-NC-ND 4.0` for internal academic reporting vs model checkpoint publishing.
2. **GPU Environment Setup**: Transition to a dedicated GPU instance (e.g., NVIDIA RTX 3090 / A100 with PyTorch CUDA).
3. **Full-Scale Baseline Fine-Tuning**:
   - Ingest the full provisional mixture (Config B: VIVOS train 11,660 utt + ViMD train 15,023 utt = ~95.4 hours).
   - Execute 5–10 epochs of fine-tuning on `vinai/PhoWhisper-tiny` with linear learning rate warmup and decay.
   - Implement early stopping monitored on the speaker-disjoint `ViMD valid` split.
4. **Frozen Benchmark Evaluation**:
   - Run a single evaluation pass on `ViMD test` (`NO_EVIDENCE_FOUND IN EXAMINED MODEL DOCUMENTATION`) for generalization WER/CER.
   - Run a reference evaluation pass on `VIVOS test` (`CONFIRMED_CONTAMINATED`) to measure in-domain ceiling performance.
