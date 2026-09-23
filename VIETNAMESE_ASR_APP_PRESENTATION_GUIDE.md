# TÀI LIỆU TỔNG HỢP TOÀN DIỆN & HƯỚNG DẪN BÁO CÁO DỰ ÁN
# VIETNAMESE SPEECH-TO-TEXT (ASR) & DATA AUGMENTATION STUDIO
**Thư mục làm việc:** `vietnamese_asr_app/`  
**Mục đích sử dụng:** Học tập, ôn luyện phản biện và thuyết trình trực tiếp trước Hội đồng / Giảng viên.  
**Nguyên tắc cốt lõi:** 100% Trung thực với mã nguồn thực tế (Code-as-Truth), không phóng đại, không ngụy tạo kết quả.

---

# MỤC LỤC
1. [PHẦN 1 — EXECUTIVE OVERVIEW (TỔNG QUAN ĐIỀU HÀNH)](#phần-1--executive-overview)
2. [PHẦN 2 — APP GIẢI QUYẾT BÀI TOÁN GÌ? (GÓC NHÌN KỸ THUẬT HỆ THỐNG)](#phần-2--app-giải-quyết-bài-toán-gì)
3. [PHẦN 3 — KIẾN TRÚC TOÀN BỘ HỆ THỐNG (SYSTEM ARCHITECTURE)](#phần-3--kiến-trúc-toàn-bộ-hệ-thống)
4. [PHẦN 4 — BẢN CHẤT MODEL ASR: NGUỒN GỐC, HUẤN LUYỆN & TÍCH HỢP](#phần-4--bản-chất-model-asr)
5. [PHẦN 5 — AUDIO PREPROCESSING PIPELINE (XỬ LÝ TÍN HIỆU ÂM THANH)](#phần-5--audio-preprocessing-pipeline)
6. [PHẦN 6 — INFERENCE & DECODING PIPELINE (CƠ CHẾ NHẬN DẠNG)](#phần-6--inference--decoding-pipeline)
7. [PHẦN 7 — AUDIO VALIDATION GATE (CỔNG KIỂM TRA 7 ĐIỂM ĐẦU VÀO)](#phần-7--audio-validation-gate)
8. [PHẦN 8 — WAVEFORM AUGMENTATION STUDIO (BIẾN ĐỔI SÓNG ÂM 1D)](#phần-8--waveform-augmentation-studio)
9. [PHẦN 9 — AUDIO QUALITY CONTROL (KIỂM SOÁT CHẤT LƯỢNG 10 TIÊU CHÍ)](#phần-9--audio-quality-control)
10. [PHẦN 10 — ĐÁNH GIÁ CHẤT LƯỢNG & ĐỘ ỔN ĐỊNH: WER, CER VÀ DRIFT](#phần-10--đánh-giá-chất-lượng--độ-ổn-định)
11. [PHẦN 11 — QUẢN TRỊ DỮ LIỆU & BẢO ĐẢM KHÔNG NHIỄM BẨN (DATASET GOVERNANCE)](#phần-11--quản-trị-dữ-liệu)
12. [PHẦN 12 — USER WORKFLOW & DEMO SCRIPT 3 PHÚT](#phần-12--user-workflow--demo-script)
13. [PHẦN 13 — XỬ LÝ HÀNG LOẠT (BATCH PROCESSING & ISOLATION)](#phần-13--xử-lý-hàng-loạt)
14. [PHẦN 14 — ĐÓNG GÓI XUẤT BẢN KẾT QUẢ (SESSION EXPORT)](#phần-14--đóng-gói-xuất-bản-kết-quả)
15. [PHẦN 15 — THIẾT KẾ GIAO DIỆN UI/UX (MINIMAL, TEXT-FIRST DESIGN)](#phần-15--thiết-kế-giao-diện-uiux)
16. [PHẦN 16 — TESTING & VERIFICATION (KIỂM THỬ PHẦN MỀM)](#phần-16--testing--verification)
17. [PHẦN 17 — HIỆU NĂNG THỰC TẾ (BENCHMARK ĐÃ ĐO ĐƯỢC)](#phần-17--hiệu-năng-thực-tế)
18. [PHẦN 18 — PHÂN BIỆT RÕ RÀNG: APP NÀY VÀ THỰC NGHIỆM BENCHMARK CLOUD](#phần-18--phân-biệt-app-và-benchmark)
19. [PHẦN 19 — 25 CÂU HỎI VÀ TRẢ LỜI PHẢN BIỆN TRỰC TIẾP](#phần-19--25-câu-hỏi-phản-biện)
20. [PHẦN 20 — CHIẾN LƯỢC TRẢ LỜI CÁC CÂU HỎI XOÁY CỦA GIẢNG VIÊN](#phần-20--chiến-lược-trả-lời-câu-hỏi-xoáy)
21. [PHẦN 21 — KỊCH BẢN THUYẾT TRÌNH 5 PHÚT](#phần-21--kịch-bản-thuyết-trình-5-phút)
22. [PHẦN 22 — KỊCH BẢN THUYẾT TRÌNH 10 PHÚT CHUYÊN SÂU](#phần-22--kịch-bản-thuyết-trình-10-phút)
23. [PHẦN 23 — ONE-PAGE CHEAT SHEET (BẢNG TÓM TẮT TRƯỚC GIỜ BÁO CÁO)](#phần-23--one-page-cheat-sheet)
24. [PHẦN 24 — FACT SHEET (BẢNG THÔNG SỐ VÀ TRẠNG THÁI THỰC TẾ)](#phần-24--fact-sheet)
25. [PHẦN 25 — LỜI DẶN DÒ: NHỮNG ĐIỀU NÊN NÓI VÀ TUYỆT ĐỐI TRÁNH NÓI](#phần-25--lời-dặn-dò)

---

# PHẦN 1 — EXECUTIVE OVERVIEW

* **Tên dự án:** Vietnamese Speech-to-Text (ASR) & Data Augmentation Studio (`vietnamese_asr_app`).
* **Bài toán:** Nhận dạng tiếng nói tiếng Việt (Vietnamese Automatic Speech Recognition - ASR) trong môi trường thực tế, kết hợp kiểm soát chất lượng tín hiệu âm thanh (Audio Quality Control), tăng cường dữ liệu sóng âm thời gian thực (Waveform Data Augmentation) và quản trị tập dữ liệu đa nguồn (Multi-Corpus Dataset Governance).
* **Mục tiêu:** Xây dựng một ứng dụng hoàn chỉnh cấp độ kỹ thuật sản xuất (production-oriented application) giải quyết bài toán suy luận (inference), phòng vệ lỗi decoder Whisper, kiểm soát chất lượng âm thanh nghiêm ngặt và tạo dữ liệu huấn luyện tăng cường đạt chuẩn, thay vì chỉ là một script gọi model mẫu.
* **Người dùng mục tiêu:** 
  1. Kỹ sư AI / Xử lý tiếng nói cần công cụ thẩm định chất lượng model và tăng cường dữ liệu âm thanh.
  2. Người dùng phổ thông / Doanh nghiệp cần công cụ gỡ băng tiếng Việt (transcription) chính xác, hỗ trợ cả ghi âm micro trực tiếp và xử lý file hàng loạt (batch).
* **Chức năng chính:**
  1. **ASR Transcription:** Nhận diện giọng nói tiếng Việt thời gian thực với cơ chế chống tràn bộ nhớ decoder (`safe max_new_tokens clamping`).
  2. **7-Point Audio Validation Gate:** Cổng tiền kiểm soát tín hiệu (Sample rate, mono, duration, non-silent RMS, peak amplitude, finite values).
  3. **Waveform Augmentation Studio:** Tạo $K$ biến thể sóng âm 1D (Gain, Calibrated Noise SNR, Time Shift, Time Stretch, Pitch Shift, Synthetic Reverb, Bandpass Filter).
  4. **10-Check Audio Quality Control (QC):** Hàng rào kiểm thử âm thanh vật lý và chữ ký mã hóa SHA-256.
  5. **Robustness & Consistency Diagnostics:** Đánh giá độ trôi văn bản (Drift WER/CER) của model trước các biến dạng âm thanh.
  6. **Dataset Governance & Simulator:** Quản trị bản quyền/siêu dữ liệu 6 bộ dữ liệu tiếng Việt (VIVOS, ViMD, BUD500, VietMed, VietSuperSpeech, VLSP) với cơ chế bảo đảm chia tách loa (Speaker-disjoint) và cách ly tập kiểm thử tuyệt đối.
  7. **Export Packaging:** Xuất dữ liệu phiên làm việc thành chuẩn WAV PCM 16-bit, bảng kê JSON/CSV và file nén ZIP độc lập.
* **Model cốt lõi:** Dòng mô hình `vinai/PhoWhisper` (mặc định cấu hình `PhoWhisper-small`, hỗ trợ cả `tiny`, `base`, `medium`), kiến trúc Transformer Encoder-Decoder được phát triển bởi VinAI Research.
* **Luồng xử lý tổng thể:**  
  `Tín hiệu âm thanh (Mic/File)` $\to$ `Chuẩn hóa 16kHz mono float32` $\to$ `Cổng kiểm định 7 tiêu chí` $\to$ `PhoWhisper Inference (Safe Decoded)` $\to$ `Chuẩn hóa văn bản NFC` $\to$ `(Tùy chọn) K-Augmentation + 10-Check QC` $\to$ `Đo lường WER/CER & Drift` $\to$ `Đóng gói xuất bản ZIP`.
* **Giá trị cốt lõi của ứng dụng:** Chuyển đổi một mô hình nền tảng ASR lý thuyết thành một hệ thống phần mềm có độ tin cậy cao, có khả năng phòng thủ trước các lỗi tràn vị trí decoder (`max_target_positions`), tự phát hiện âm thanh hỏng/rỗng, và cung cấp quy trình sinh dữ liệu âm thanh nhân tạo có kiểm chuẩn toán học và vật lý.

---

### “Nếu chỉ có 30 giây để giới thiệu project, tôi nên nói gì?”

> *"Dự án của em là một hệ thống Studio hoàn chỉnh cho Nhận dạng tiếng nói tiếng Việt và Quản trị Tăng cường dữ liệu âm thanh. Thay vì chỉ dựng một web demo gọi API đơn giản, dự án tập trung vào kỹ thuật hệ thống (system engineering): em tích hợp mô hình PhoWhisper với cơ chế tự động giới hạn token động nhằm ngăn chặn lỗi tràn kiến trúc 448 vị trí của Whisper, xây dựng cổng kiểm định tín hiệu đầu vào 7 tiêu chí, bộ công cụ tạo $K$ biến thể sóng âm 1D kèm hàng rào kiểm soát chất lượng âm thanh 10 tiêu chí và chữ ký SHA-256, cùng mô-đun mô phỏng phân chia dữ liệu đảm bảo không rò rỉ người nói. Toàn bộ logic phần mềm được kiểm chứng qua bộ 31 bài kiểm thử tự động đạt độ tin cậy tuyệt đối."*

---

# PHẦN 2 — APP GIẢI QUYẾT BÀI TOÁN GÌ?

### 2.1 Bản chất kỹ thuật của ASR
ASR (*Automatic Speech Recognition* - Nhận dạng tiếng nói tự động) là bài toán chuyển đổi chuỗi tín hiệu sóng âm liên tục theo thời gian $x(t)$ thành chuỗi ký tự/từ ngôn ngữ có nghĩa $W = (w_1, w_2, \dots, w_n)$:
$$W^* = \arg\max_W P(W \mid X)$$
Trong ứng dụng này:
* **Input:** File âm thanh số (WAV, MP3, FLAC, OGG) hoặc luồng ghi âm trực tiếp từ microphone.
* **Output:** Chuỗi văn bản tiếng Việt chuẩn Unicode dựng sẵn (NFC), kèm phiên bản chuẩn hóa bỏ dấu câu cho chấm điểm, thời gian xử lý (latency) và hệ số thời gian thực (Real-Time Factor - RTF).

### 2.2 Vì sao nhận dạng tiếng nói tiếng Việt (Vietnamese ASR) lại khó?
1. **Ngôn ngữ thanh điệu (Tonal Language):** Tiếng Việt có 6 thanh điệu (ngang, huyền, sắc, hỏi, ngã, nặng). Độ cao của cao độ cơ bản ($F_0$) đóng vai trò quyết định ngữ nghĩa của từ. Ví dụ: *ma, má, mà, mả, mã, mạ* là 6 từ vựng hoàn toàn khác nhau.
2. **Đa dạng phương ngữ (Dialectal Diversity):** Tiếng Việt có sự khác biệt phát âm rất lớn giữa ba miền Bắc, Trung, Nam và 63 tỉnh thành. Nhiều phụ âm đầu (`tr/ch`, `s/x`, `d/gi/r`) hoặc vần cuối (`n/ng`, `t/c`) bị biến âm mạnh mẽ theo vùng miền.
3. **Từ ghép đơn âm tiết & hiện tượng đồng âm:** Tiếng Việt là ngôn ngữ đơn lập, ranh giới từ vựng phụ thuộc chặt chẽ vào ngữ cảnh của câu.
4. **Nhiễu môi trường và thiết bị thu âm:** Trong thực tế, âm thanh thu từ điện thoại, micro rẻ tiền thường bị méo biên độ, vang vọng phòng hoặc có tỷ số tín hiệu trên nhiễu (SNR) thấp.

### 2.3 Tại sao một ứng dụng ASR thực tế cần nhiều hơn một Model Inference?
Trong các bài báo khoa học, mô hình thường được đánh giá trên các tập dữ liệu sạch (gold standard datasets) đã được tiền xử lý hoàn hảo. Nhưng trong ứng dụng sản xuất:
* **Âm thanh thực tế là "bẩn":** Người dùng có thể vô tình gửi file rỗng (toàn silence), file bị rè vỡ tiếng (severe clipping), file quá ngắn (dưới 0.2s - chỉ có tiếng click chuột), hoặc file quá dài (vượt cửa sổ ngữ cảnh 30 giây của Whisper). Nếu không có tiền xử lý và kiểm định, model sẽ sinh ra ảo giác (*hallucination*), lặp từ vô tận, hoặc crash chương trình.
* **Nguy cơ lỗi kiến trúc decoder:** Whisper có giới hạn cứng $448$ vị trí giải mã. Nếu câu lệnh giải mã yêu cầu quá số token này cộng với các token mồi (prompt tokens), hệ thống sẽ sập với lỗi `ValueError`. Ứng dụng phải có cơ chế kẹp an toàn (*safe clamping*).
* **Nhu cầu thẩm định độ bền vững (Robustness Testing):** Model nhận dạng tốt trên giọng chuẩn trong phòng thu liệu có nhận dạng được khi có tiếng ồn đường phố, tiếng vang phòng họp, hay giọng nói bị biến đổi âm sắc không? Tính năng Waveform Augmentation và Drift Measurement sinh ra để trả lời câu hỏi này.
* **Quản trị bản quyền dữ liệu:** Huấn luyện hoặc tinh chỉnh model đòi hỏi trộn nhiều tập dữ liệu. Nếu vô tình để rò rỉ loa (speaker leak) hoặc trộn tập test vào tập train, kết quả nghiên cứu sẽ bị vô hiệu hóa.

---

# PHẦN 3 — KIẾN TRÚC TOÀN BỘ HỆ THỐNG

### 3.1 Sơ đồ kiến trúc thực tế (Concrete Architecture Diagram)

```text
====================================================================================================
                                      NGƯỜI DÙNG / TRÌNH DUYỆT
                                                 │
                                                 ▼
                             Gradio Text-First UI (app.py : build_app)
       ┌───────────────────────┬─────────────────────────┬───────────────────────┐
       ▼                       ▼                         ▼                       ▼
  Tab 1: STUDIO        Tab 2: BATCH QUEUE       Tab 3: PROVENANCE        Tab 4: SETTINGS
 (Workflow 6 bước)    (Streaming isolation)   (Governance & Sim)     (Model Cache & HW)
       │                       │                         │                       │
       ▼                       ▼                         │                       ▼
┌──────────────────────────────────────────────┐         │             ┌──────────────────┐
│ asr/audio.py : load_audio                    │         │             │ asr/model.py     │
│ - Decode bytes/filepath/mic tuple            │         │             │ - detect_hardware│
│ - Resample to 16000 Hz                       │         │             │ - ASRModelManager│
│ - Average multi-channel to 1D Mono           │         │             │   (Singleton     │
│ - Normalize float32 range [-1.0, 1.0]        │         │             │    Model Cache)  │
└──────────────────────┬───────────────────────┘         │             └────────┬─────────┘
                       ▼                                 │                      │
┌──────────────────────────────────────────────┐         │                      │
│ asr/audio.py : validate_audio                │         │                      │
│ [7-Point Audio Validation Gate]              │         │                      │
│ (1.Signal, 2.Finite, 3.Mono, 4.Min Dur,      │         │                      │
│  5.Max Dur, 6.RMS silence, 7.Peak clipping)  │         │                      │
└──────────────────────┬───────────────────────┘         │                      │
                       │ [PASS]                          │                      │
                       ▼                                 │                      │
┌──────────────────────────────────────────────┐         │                      │
│ asr/inference.py : ASRInferenceEngine        │◄────────┼──────────────────────┘
│ - Inspect model.config.max_target_positions  │         │
│ - Get decoder prompt tokens ('vi', transcribe)│        │
│ - Dynamic Token Clamping:                    │         │
│   effective = min(req, 448 - prompt_len - 1) │         │
│ - 30s Sequential Chunking (Long audio)       │         │
│ - Feature Extractor (80-band Log-Mel)        │         │
│ - WhisperForConditionalGeneration.generate() │         │
│ - batch_decode(skip_special_tokens=True)     │         │
└──────────────────────┬───────────────────────┘         │
                       ▼                                 │
┌──────────────────────────────────────────────┐         │
│ asr/normalization.py                         │         │
│ - Unicode NFC Canonical Normalization        │         │
│ - Lowercase conversion                       │         │
│ - Punctuation stripping & whitespace collaps │         │
└──────────────────────┬───────────────────────┘         │
                       ▼                                 │
           [Hiển thị Transcript & RTF]                   │
                       │                                 │
                       ├──────────────────────────┐      │
                       ▼ (Tùy chọn)               │      │
┌──────────────────────────────────────────────┐  │      │
│ augmentation/pipeline.py : AugEngine         │  │      │
│ - Mode: Random K (1<=K<=20) or Exact Plan    │  │      │
│ - Seeded deterministic PRNG                  │  │      │
│ - augmentation/transforms.py (1D Waveform):  │  │      │
│   Gain, Noise SNR, Shift, Stretch, Pitch,    │  │      │
│   Synthetic Reverb, Bandpass Filter          │  │      │
└──────────────────────┬───────────────────────┘  │      │
                       ▼                          │      │
┌──────────────────────────────────────────────┐  │      │
│ augmentation/qc.py : run_10_point_qc         │  │      │
│ [10-Check Audio Quality Control Pipeline]    │  │      │
│ - Finite, NaN, Inf, RMS, Clipping ratio,     │  │      │
│   Duration bounds, 16kHz mono, SHA-256       │  │      │
└──────────────────────┬───────────────────────┘  │      │
                       ▼                          │      │
┌──────────────────────────────────────────────┐  │      │
│ evaluation/metrics.py                        │  │      │
│ - Re-transcribe augmented waveforms          │  │      │
│ - Compute Drift WER & Drift CER              │  │      │
│ - Compute WER/CER vs Reference text (jiwer)  │  │      │
└──────────────────────┬───────────────────────┘  │      │
                       ▼                          ▼      │
┌─────────────────────────────────────────────────────┐  │
│ export/exporter.py : AugmentationSessionExporter    │  │
│ - Save original.wav & aug_XX.wav (16-bit PCM WAV)   │  │
│ - Save transcripts (.txt) & metadata (.json)        │  │
│ - Export manifest.csv & results.json                │  │
│ - Build Standalone Downloadable ZIP Archive         │  │
└─────────────────────────────────────────────────────┘  │
                                                         ▼
                               ┌──────────────────────────────────────────────┐
                               │ datasets/mixture.py & registry.py            │
                               │ - 6 Candidate Corpora Metadata & Licences    │
                               │ - Speaker-Disjoint Partitioner Simulation    │
                               │ - Zero Test Contamination Quarantine Check   │
                               └──────────────────────────────────────────────┘
====================================================================================================
```

### 3.2 Bảng phân nhiệm chi tiết từng Mô-đun (Module Breakdown)

| Tên File / Module | Đầu vào (Input) | Đầu ra (Output) | Vai trò kỹ thuật |
| :--- | :--- | :--- | :--- |
| `app.py` | Tương tác người dùng từ Web UI | Các component Gradio cập nhật trạng thái | Điều phối toàn bộ sự kiện giao diện, kết nối các luồng dữ liệu (orchestrator). |
| `asr/audio.py` | Đường dẫn file, byte stream, hoặc tuple micro `(sr, ndarray)` | Mảng NumPy 1D `float32` (16kHz), kết quả cổng 7 điểm | Tải âm thanh đa định dạng, resample chuẩn, ép mono và thẩm định vật lý. |
| `asr/model.py` | Model ID (`vinai/PhoWhisper-small`), yêu cầu thiết bị | Tuple `(model, processor, hw_diagnostics)` | Quản lý vòng đời mô hình theo mẫu Singleton Cache, phát hiện phần cứng (CPU/CUDA). |
| `asr/inference.py` | Sóng âm 16kHz, chiến lược giải mã, số beam | `TranscriptionResult` (Text, RTF, Token audit) | Trích xuất đặc trưng log-Mel spectrogram, giải mã văn bản an toàn không lỗi tràn token. |
| `asr/normalization.py`| Chuỗi văn bản thô | Chuỗi chuẩn hóa NFC tiếng Việt | Chuẩn hóa Unicode tiếng Việt, xóa dấu câu, hạ chữ thường, triệt tiêu khoảng trắng thừa. |
| `augmentation/transforms.py`| Sóng âm 1D `float32`, tham số biến đổi | Sóng âm 1D `float32` đã biến đổi | Áp dụng các phép biến đổi âm thanh trực tiếp trong miền thời gian (không qua spectrogram). |
| `augmentation/pipeline.py` | Sóng âm gốc, số lượng $K$, hạt giống seed | Danh sách các đối tượng `AugmentedAudioSample` | Sinh ngẫu nhiên hoặc theo kịch bản $K$ biến thể âm thanh kèm lịch sử biến đổi. |
| `augmentation/qc.py` | Sóng âm biến thể, sample rate | `QualityCheckResult` (PASS/FAIL, SHA-256) | Chạy 10 bài kiểm tra vật lý âm thanh và tính toán mã băm mật mã toàn vẹn. |
| `evaluation/metrics.py`| Cặp chuỗi văn bản (Ref/Hyp, Orig/Aug) | `MetricBreakdown`, `ASRComparisonEvaluation` | Tính toán khoảng cách Levenshtein (WER, CER, S, D, I, H) và độ lệch văn bản (Drift). |
| `datasets/registry.py` | Mã dataset (`vivos`, `vimd`,...) | `DatasetProvenance` metadata object | Lưu trữ siêu dữ liệu có thẩm quyền, phân hạng chất lượng (Gold/Silver), bản quyền. |
| `datasets/mixture.py` | Danh sách mẫu dữ liệu thô đa nguồn | Tập `Train`, tập `Val`, `audit_summary` | Phân chia tập dữ liệu đảm bảo người nói độc lập hoàn toàn và cách ly tập test. |
| `export/exporter.py` | Âm thanh, transcript, metadata phiên làm việc | Đường dẫn thư mục xuất bản và file ZIP | Đóng gói toàn bộ kết quả thành gói dữ liệu độc lập chuẩn WAV PCM 16-bit. |

---

# PHẦN 4 — BẢN CHẤT MODEL ASR: NGUỒN GỐC, HUẤN LUYỆN & TÍCH HỢP

> [!CAUTION]
> **ĐÂY LÀ PHẦN QUAN TRỌNG NHẤT TRONG BUỔI BÁO CÁO.**  
> Giảng viên sẽ chất vấn sâu về nguồn gốc model. Bạn phải trả lời chính xác từng chữ theo nội dung dưới đây. Tuyệt đối không được nói *"nhóm em tự huấn luyện ra model PhoWhisper"*.

### 4.1 Model đang sử dụng là model nào?
* **Tên mô hình:** `PhoWhisper` (mặc định trong app là `vinai/PhoWhisper-small`, ngoài ra hỗ trợ chuyển đổi qua `vinai/PhoWhisper-tiny`, `vinai/PhoWhisper-base`, `vinai/PhoWhisper-medium` trong tab Settings).
* **Kho lưu trữ (Hugging Face Repository):** [`vinai/PhoWhisper-small`](https://huggingface.co/vinai/PhoWhisper-small)
* **Đơn vị phát triển mô hình gốc:** **VinAI Research** (tác giả chính: Linh The Nguyen, Dat Quoc Nguyen).
* **Kiến trúc mô hình (Architecture):** Sequence-to-Sequence Transformer dạng Encoder-Decoder theo chuẩn OpenAI Whisper:
  * **Audio Encoder:** Nhận ma trận 80 kênh log-Mel spectrogram (được tính bằng cửa sổ FFT 25ms, bước nhảy hop 10ms trên âm thanh 16kHz). Encoder gồm 2 lớp tích chập 1D (Conv1D với stride=2) để giảm kích thước thời gian xuống 4 lần, tiếp theo là chồng các khối Transformer Encoder Blocks (12 blocks đối với bản Small) có cơ chế self-attention và sinusoidal positional encoding.
  * **Text Decoder:** Mô hình sinh tự hồi quy (Autoregressive Transformer Decoder - 12 blocks đối với bản Small) sử dụng masked self-attention, cross-attention hướng tới output của encoder, và learned positional embeddings với giới hạn độ dài vị trí tối đa `max_target_positions = 448`.
* **Kích thước mô hình (Parameters):**
  * `PhoWhisper-tiny`: ~39 triệu tham số.
  * `PhoWhisper-base`: ~74 triệu tham số.
  * `PhoWhisper-small`: ~244 triệu tham số (mặc định của app).
  * `PhoWhisper-medium`: ~769 triệu tham số.
* **Bộ xử lý & Tokenizer:** `WhisperProcessor` kết hợp `WhisperFeatureExtractor` (tạo Mel spectrogram) và `WhisperTokenizerFast` (Byte-level BPE với từ vựng mở rộng hỗ trợ ngôn ngữ tiếng Việt).

### 4.2 Model là pretrained hay tự train?
* **Khẳng định dứt khoát:** Trong phạm vi ứng dụng `vietnamese_asr_app/`, đây là **MÔ HÌNH PRETRAINED ĐÃ ĐƯỢC HUẤN LUYỆN SẴN BỞI VINAI RESEARCH** và được tải trực tiếp từ Hugging Face Hub.
* **Nhóm có tự train lại model trong app không?** **KHÔNG.** Nhóm không huấn luyện lại các trọng số này từ đầu (from scratch) và cũng không thực hiện fine-tune full model bên trong thư mục ứng dụng `vietnamese_asr_app/`.
* **Phân định ranh giới đóng góp kỹ thuật:**
  * **Phần của VinAI:** Nghiên cứu kiến trúc, chuẩn bị dữ liệu tiền huấn luyện trên hàng nghìn giờ tiếng Việt, huấn luyện trọng số mô hình PhoWhisper.
  * **Phần đóng góp của nhóm em (Application Engineering & Robust Systems):** 
    1. Thiết kế và triển khai toàn bộ tầng tiền xử lý âm thanh, chuẩn hóa đầu vào và cổng thẩm định 7 tiêu chí (`asr/audio.py`).
    2. Phát hiện và xử lý dứt điểm lỗi tràn kiến trúc giải mã của Whisper (`max_target_positions=448`) bằng cơ chế tính toán ngân sách token động (`safe max_new_tokens clamping` trong `asr/inference.py`).
    3. Triển khai cơ chế phân đoạn tuần hoàn (sequential chunking) cho các file âm thanh dài hơn 30 giây để tránh việc bị cắt cụt văn bản ngầm.
    4. Xây dựng toàn bộ hệ thống tăng cường sóng âm vật lý 1D (Waveform Augmentation Studio) và hàng rào 10 bài kiểm tra chất lượng (Audio QC Pipeline).
    5. Xây dựng hệ thống chẩn đoán độ trôi văn bản (Drift WER/CER) nhằm kiểm tra độ ổn định của PhoWhisper khi gặp biến dạng âm thanh.
    6. Thiết kế mô hình quản trị bản quyền 6 tập dữ liệu tiếng Việt và thuật toán phân chia tập train/val độc lập người nói (Speaker-disjoint) kèm bộ bảo vệ chống nhiễm bẩn tập test (Zero test contamination).

### 4.3 “Vậy model của nhóm được tích hợp vào app như thế nào?”
Mô hình được tích hợp qua một quy trình kỹ thuật chặt chẽ trong mã nguồn:
```text
Tải Pretrained Model & Processor từ HF Hub (asr/model.py : ASRModelManager)
                              ↓
    Singleton Memory Cache (Tránh cấp phát lại bộ nhớ mỗi request)
                              ↓
    Âm thanh đầu vào (Microphone / File) qua load_audio (16kHz Mono)
                              ↓
        Cổng thẩm định tín hiệu 7-Point Audio Validation Gate
                              ↓
    Kiểm toán cấu hình giải mã:
    - prompt_len = len([decoder_start, 'vi', 'transcribe', 'notimestamps']) = 4
    - effective_tokens = min(requested, 448 - 4 - 1) = 443
                              ↓
    Feature Extractor: Chuyển sóng âm 16kHz -> 80-channel log-Mel Spectrogram
                              ↓
    Forward Pass & Beam Search / Greedy Decoding (torch.inference_mode())
                              ↓
    Batch Decode thành chuỗi ký tự tiếng Việt thô
                              ↓
    Chuẩn hóa văn bản Unicode NFC (asr/normalization.py)
                              ↓
    Đo đạc độ trễ (Latency) & Hệ số thời gian thực (RTF)
```

### 4.4 Lịch sử huấn luyện (Training History) trong mã nguồn
Trong mã nguồn `vietnamese_asr_app/scripts/train.py`, ta thấy có code training. Cần phân định 3 mức độ thực tế:
1. **Đã thực sự huấn luyện (Full Training):** `CHƯA THỰC HIỆN` trong thư mục app này.
2. **Đã chạy thử nghiệm khởi tạo và lưu vết (Dry-run & Resume Smoke Test):** `ĐÃ THỰC HIỆN & ĐÃ KIỂM CHỨNG (VERIFIED)`. File `vietnamese_asr_app/checkpoints/test_resume_smoke.pt` (kích thước 20.6 KB) và `run_metadata.json` là minh chứng cho việc chạy lệnh `python scripts/train.py --dry_run`. Nó chứng minh rằng cơ chế lưu trữ đầy đủ trạng thái model, optimizer AdamW, scheduler CosineAnnealing, và khôi phục 100% hạt giống ngẫu nhiên (Python, NumPy, PyTorch, CUDA RNG) hoạt động hoàn hảo.
3. **Mã nguồn huấn luyện hoàn chỉnh:** Code được thiết kế sẵn sàng cho việc huấn luyện đa tập dữ liệu (multi-dataset mixture) khi đưa lên cụm GPU đám mây.

### 4.5 Cơ chế nạp và tái sử dụng mô hình (Model Caching)
* Triển khai tại class `ASRModelManager` (`asr/model.py`).
* Sử dụng biến từ điển `_instances: Dict[str, Tuple[Model, Processor]]` lưu trên bộ nhớ RAM.
* Khi người dùng thực hiện nhận dạng liên tiếp, hệ thống kiểm tra `cache_key = f"{model_id}_{device}"`. Nếu đã nạp, nó tái sử dụng ngay lập tức đối tượng trong bộ nhớ, **tuyệt đối không load lại từ ổ cứng hay download lại từ internet**, giúp giảm độ trễ từ vài giây xuống mili-giây.
* Hỗ trợ nút `Clear memory cache` trong tab Settings để giải phóng VRAM/RAM (gọi `gc.collect()` và `torch.cuda.empty_cache()`).

---

# PHẦN 5 — AUDIO PREPROCESSING PIPELINE

Quy trình xử lý âm thanh trong `asr/audio.py` tuân theo chuẩn kỹ thuật số khắt khe:

```text
File WAV/MP3/FLAC hoặc Tuple Micro (sr, data)
                      │
                      ▼
            [1. Giải mã định dạng]
   - Nếu từ micro: Ép kiểu nguyên (int16/int32) về float32 [-1.0, 1.0].
   - Nếu từ file: Sử dụng librosa / soundfile đọc ma trận âm thanh.
                      │
                      ▼
            [2. Chuẩn hóa kênh (Mono)]
   - Nếu âm thanh đa kênh (Stereo / 2D array): Lấy trung bình cộng các kênh:
     y = np.mean(y, axis=1) để loại bỏ hiện tượng trôi pha.
                      │
                      ▼
       [3. Lấy mẫu lại (Resampling) về 16,000 Hz]
   - Nếu sr != 16000: Dùng bộ lọc nội suy librosa.resample() chuẩn Polyphase.
   - 16 kHz là tần số lấy mẫu chuẩn mực của Whisper và đa số hệ thống ASR,
     đảm bảo giữ lại băng thông tần số tiếng nói con người (lên tới 8 kHz theo định lý Nyquist).
                      │
                      ▼
        [4. Đánh giá Cổng thẩm định 7 điểm]
                      │
                      ▼
       [5. Trích xuất đặc trưng (Feature Extraction)]
   - Thực hiện bởi WhisperFeatureExtractor:
     + Chia khung (framing): Cửa sổ 25ms (400 samples tại 16kHz).
     + Bước dịch (hop size): 10ms (160 samples tại 16kHz).
     + Biến đổi Fourier nhanh (STFT) với cửa sổ Hanning.
     + Chiếu năng lượng phổ qua ma trận tam giác 80 dải lọc Mel (Mel filterbank)
       từ 0 Hz đến 8000 Hz.
     + Lấy logarithm cơ số 10 của năng lượng: log10(Mel + 1e-5), sau đó chuẩn hóa biên độ.
```

---

# PHẦN 6 — INFERENCE & DECODING PIPELINE

### 6.1 Cơ chế giải mã và Giải quyết dứt điểm Bug tràn Decoder Tokens
* **Nguyên nhân gốc rễ của lỗi tràn Token:**  
  Trong kiến trúc OpenAI Whisper, lớp nhúng vị trí của Decoder (`decoder.embed_positions`) có kích thước cố định $448$. Khi bắt đầu sinh văn bản tiếng Việt, WhisperProcessor tạo ra các token mồi ban đầu:
  1. `<|startoftranscript|>` (50258)
  2. `<|vi|>` (Mã ngôn ngữ tiếng Việt - 50327)
  3. `<|transcribe|>` (Tác vụ nhận dạng - 50359)
  4. `<|notimestamps|>` (Không sinh nhãn thời gian - 50363)  
  Tổng độ dài prompt ban đầu là $\text{prompt\_len} = 4$.  
  Nếu code gọi `generate(max_new_tokens=448)`, decoder sẽ cố gắng sinh tối đa 448 token mới. Độ dài chuỗi decoder tối đa có thể đạt tới $4 + 448 = 452$. Con số này vượt quá $448$, khiến PyTorch ném lỗi `ValueError: The length of decoder_input_ids is 4, and max_new_tokens is 448. The combined length is 452. This exceeds max_target_positions=448`.

* **Giải pháp kỹ thuật của nhóm em (`asr/inference.py`):**
  1. Truy vấn động tham số kiến trúc thực tế của model:  
     `max_target_positions = getattr(self._model.config, "max_target_positions", 448)`
  2. Đo đạc độ dài prompt thực tế:  
     `prompt_len = int(decoder_input_ids.shape[-1])` (thường là 4, nhưng có thể dài hơn nếu có tiền tố ngữ cảnh).
  3. Áp dụng công thức kẹp an toàn với biên an toàn $\text{margin} = 1$:  
     $$\text{safe\_max\_new\_tokens} = \max(1, \min(\text{requested}, \text{max\_target\_positions} - \text{prompt\_len} - 1))$$
     Với cấu hình mặc định: $\min(448, 448 - 4 - 1) = 443$.
  4. Xác lập `forced_decoder_ids = None` để tránh xung đột hoặc cảnh báo trùng lặp cấu hình giải mã giữa Transformers cũ và mới.
  5. Đặt assertion chặn trước khi chạy:  
     `assert prompt_len + effective_max_new_tokens <= max_target_positions`

### 6.2 Xử lý âm thanh dài hơn 30 giây (Sequential Chunking)
Kiến trúc Whisper chỉ nhận spectrogram có độ dài cố định 30 giây (tương đương 3000 frames phổ, hoặc 480,000 mẫu tại 16kHz). Nếu âm thanh dài hơn 30s:
* Thay vì cắt cụt ngầm gây mất mát dữ liệu của người dùng, ứng dụng chia file âm thanh thành các đoạn tuần tự $30.0$ giây (`audio_16k[i * 480000 : (i + 1) * 480000]`).
* Thực hiện suy luận độc lập trên từng đoạn dưới khối lệnh `with torch.inference_mode():`.
* Ghép các câu kết quả lại bằng khoảng trắng và chuyển qua chuẩn hóa NFC.

---

# PHẦN 7 — AUDIO VALIDATION GATE

Ứng dụng cài đặt một cổng kiểm soát 7 tiêu chí (`asr/audio.py : validate_audio`) bắt buộc mọi âm thanh phải vượt qua trước khi tiêu tốn tài nguyên GPU/CPU:

| Thứ tự | Tên tiêu chí (Check Name) | Điều kiện kỹ thuật (Threshold) | Ý nghĩa vật lý & Thuật toán | Hành động khi thất bại (Fail Action) |
| :---: | :--- | :--- | :--- | :--- |
| **1** | Signal Existence | `audio is not None and len(audio) > 0` | Đảm bảo mảng dữ liệu có tồn tại, tránh lỗi `NullPointer` hoặc mảng 0 byte. | Chặn ngay, báo `"Audio signal is empty or None"`. |
| **2** | Finite Values | `np.all(np.isfinite(audio))` | Kiểm tra mảng không chứa giá trị `NaN` hoặc `+Inf/-Inf` do lỗi giải mã byte. | Chặn, báo `"Audio contains NaN or Infinite values"`. |
| **3** | Mono Channel | `audio.ndim == 1` | Đảm bảo tín hiệu là 1 chiều (1D mono) đúng chuẩn ma trận đầu vào ASR. | Chặn, báo định dạng sai số kênh. |
| **4** | Min Duration | $\text{duration} \ge 0.20\text{ s}$ ($3,200$ mẫu tại 16kHz) | Loại bỏ các tạp âm click chuột, nhiễu xung quá ngắn không thể chứa âm vị tiếng nói. | Chặn, trả về độ dài thực tế và ngưỡng tối thiểu. |
| **5** | Max Duration | $\text{duration} \le 30.00\text{ s}$ (hoặc `allow_chunking=True`) | Bảo vệ bộ nhớ của mô hình giải mã, kích hoạt cờ phân đoạn nếu file dài. | Nếu không cho phép chunking: chặn. Nếu trong App Studio: cho phép phân đoạn. |
| **6** | RMS Silence Check | $\text{RMS} = \sqrt{\frac{1}{N}\sum x_i^2} \ge 1\times 10^{-5}$ | Phát hiện file âm thanh rỗng (im lặng hoàn toàn), tránh model sinh hallucination lặp từ. | Chặn, báo tín hiệu im lặng `"Audio appears completely silent"`. |
| **7** | Dynamic Range & Peak | $\text{Peak} = \max(\|x_i\|) \le 1.20$ | Ngăn chặn tín hiệu bị khuếch đại vượt ngưỡng cho phép gây rè vỡ kỹ thuật số (harsh distortion). | Chặn, thông báo biên độ đỉnh quá lớn vượt chuẩn. |

---

# PHẦN 8 — WAVEFORM AUGMENTATION STUDIO

### 8.1 Triết lý thiết kế: Tăng cường trên sóng âm 1D (Waveform Domain)
Nhiều nghiên cứu áp dụng tăng cường trên spectrogram (như SpecAugment: che thời gian/tần số). Tuy nhiên, SpecAugment chỉ phù hợp khi đang huấn luyện mạng nơ-ron (gradient descent). Đối với một ứng dụng Studio kiểm tra độ bền vững ngoại tuyến:
* Mọi biến đổi trong ứng dụng này đều diễn ra trực tiếp trên **sóng âm 1D trong miền thời gian** (`augmentation/transforms.py`).
* **Ưu điểm:** Âm thanh sau khi biến đổi là âm thanh vật lý thực sự, có thể phát lại qua loa/tai nghe cho con người thẩm định, có thể lưu thành file WAV chuẩn và tương thích với bất kỳ hệ thống ASR nào mà không gặp hiện tượng lỗi pha khi tổng hợp ngược.

### 8.2 Bảy phép biến đổi sóng âm chính xác (7 Waveform Transforms)

1. **Thay đổi âm lượng (Linear Gain):**
   * *Công thức:* $y(t) = x(t) \cdot 10^{\frac{\text{gain\_db}}{20}}$.
   * *Ý nghĩa:* Mô phỏng người nói ở xa hoặc gần microphone. Có cơ chế tự động nén nếu đỉnh vượt quá $0.99$ để tránh clipping.
2. **Cộng nhiễu Gauss theo tỷ số SNR định chuẩn (Calibrated Additive Noise):**
   * *Thuật toán:* Đo công suất tín hiệu $P_{\text{signal}} = \frac{1}{N}\sum x^2$. Tính công suất nhiễu cần thiết dựa trên SNR mục tiêu (ví dụ 15dB - 25dB): $P_{\text{noise}} = \frac{P_{\text{signal}}}{10^{\text{SNR}/10}}$. Sinh nhiễu trắng $\mathcal{N}(0, \sigma^2)$ và cộng trực tiếp.
   * *Ý nghĩa:* Mô phỏng môi trường quán cà phê, tiếng gió, tiếng ồn thiết bị điện tử.
3. **Dịch chuyển thời gian (Time Shift):**
   * *Thuật toán:* Dịch toàn bộ mảng sóng âm sang trái hoặc phải $k$ mẫu, chèn giá trị 0 vào vùng khuyết (zero-padding).
   * *Ý nghĩa:* Mô phỏng việc người nói bắt đầu câu nói sớm hay muộn trong khung thu âm.
4. **Co giãn thời gian không đổi cao độ (Time Stretch):**
   * *Thuật toán:* Sử dụng Phase Vocoder của Librosa trên biến đổi STFT để thay đổi tốc độ (từ 0.85x đến 1.15x) mà không làm biến đổi tần số cơ bản của giọng nói.
   * *Ý nghĩa:* Mô phỏng tốc độ nói nhanh (nói vội) hoặc nói chậm (người già).
5. **Dịch chuyển cao độ (Pitch Shift):**
   * *Thuật toán:* Dịch chuyển cao độ âm thanh lên hoặc xuống theo số bán âm (semitones: -2 đến +2) bằng thuật toán Librosa.
   * *Ý nghĩa:* Mô phỏng các tông giọng khác nhau (giọng nam trầm, giọng nữ cao, giọng trẻ em).
6. **Tạo tiếng vang phòng tổng hợp (Synthetic Room Reverb):**
   * *Thuật toán:* Tạo phản ứng xung phòng nhân tạo (Room Impulse Response - RIR) phân rã theo hàm mũ $h(t) = e^{-\frac{3t}{\tau}} \cdot \mathcal{N}(0, 1)$, sau đó tích chập trong miền tần số (FFT Convolution) với sóng âm gốc và hòa trộn theo tỷ lệ wet/dry.
   * *Ý nghĩa:* Mô phỏng âm thanh phát ra trong phòng họp lớn, hành lang hoặc lớp học có hiện tượng dội âm.
7. **Lọc dải tần viễn thông (Bandpass Filtering):**
   * *Thuật toán:* Bộ lọc Butterworth bậc 4 dải thông từ 300 Hz đến 3400 Hz (chuẩn đường truyền băng hẹp điện thoại cố định/di động PSTN).
   * *Ý nghĩa:* Thẩm định độ chính xác của ASR khi ứng dụng vào hệ thống tổng đài chăm sóc khách hàng (Call Center).

### 8.3 Cơ chế $K$-Augmentation và tính lặp lại (Reproducibility)
* Người dùng có thể chọn chế độ sinh ngẫu nhiên $K$ biến thể ($1 \le K \le 20$).
* Toàn bộ quá trình được điều khiển bằng `master_seed` qua bộ sinh số ngẫu nhiên cô lập `np.random.default_rng(seed + i)`.
* **Đảm bảo tính khoa học:** Cùng một file âm thanh và cùng một seed sẽ luôn sinh ra $K$ biến thể sóng âm đồng nhất $100\%$ đến từng bit mã hóa.

---

# PHẦN 9 — AUDIO QUALITY CONTROL (QC)

Trước khi một file âm thanh tăng cường được phép đưa vào nhận dạng hoặc lưu trữ, nó phải vượt qua **Hàng rào 10 bài kiểm tra chất lượng âm thanh nghiêm ngặt** (`augmentation/qc.py : run_10_point_qc`):

| Mã QC | Tên bài kiểm tra | Điều kiện kỹ thuật chấp thuận | Mục đích khoa học |
| :---: | :--- | :--- | :--- |
| **QC-01** | Decodable Array | `isinstance(audio, np.ndarray) and size > 0` | Đảm bảo biến đổi không trả về giá trị `None` hoặc mảng rỗng. |
| **QC-02** | Finite Values | `np.all(np.isfinite(audio))` | Xác thực 100% mẫu âm thanh là số thực hữu hạn. |
| **QC-03** | No NaNs | `np.isnan(audio).sum() == 0` | Ngăn chặn hiện tượng lan truyền giá trị bất định vào mạng nơ-ron. |
| **QC-04** | No Infs | `np.isinf(audio).sum() == 0` | Ngăn chặn hiện tượng tràn số học (+/- Vô cùng). |
| **QC-05** | Range Boundaries | $\max(\|x_i\|) \le 1.05$ | Đảm bảo biên độ âm thanh không vượt trần biên độ số học. |
| **QC-06** | Clipping Ratio | $\frac{\sum [\|x_i\| \ge 0.999]}{N} \le 0.001$ ($0.1\%$) | Không cho phép âm thanh bị cắt ngọn sóng quá 0.1% tổng số mẫu, tránh tiếng méo rè phá hủy tín hiệu. |
| **QC-07** | RMS Signal Level | $\text{RMS} \ge 1\times 10^{-5}$ | Đảm bảo âm thanh sau biến đổi không bị triệt tiêu thành khoảng lặng. |
| **QC-08** | Duration Bounds | $0.2\text{ s} \le \text{Duration} \le 30.0\text{ s}$ | Đảm bảo thời lượng âm thanh nằm trong khung xử lý an toàn của hệ thống. |
| **QC-09** | Format Conformity | $\text{Channels} == 1 \land \text{SampleRate} == 16000$ | Kiểm định âm thanh giữ đúng chuẩn 1 kênh mono và 16 kHz. |
| **QC-10** | SHA-256 Digest | Tính mã băm SHA-256 trên chuỗi byte `float32` | Định danh mật mã duy nhất cho từng file âm thanh, phục vụ truy xuất nguồn gốc và chống trùng lặp. |

---

# PHẦN 10 — ĐÁNH GIÁ CHẤT LƯỢNG & ĐỘ ỔN ĐỊNH: WER, CER VÀ DRIFT

### 10.1 Công thức tính Word Error Rate (WER) và Character Error Rate (CER)
Dựa trên thuật toán quy hoạch động Levenshtein giữa chuỗi tham chiếu $R$ và chuỗi nhận dạng $H$:
$$\text{WER} = \frac{S + D + I}{N_R}$$
Trong đó:
* $S$ (*Substitutions*): Số từ bị nhận dạng sai thành từ khác.
* $D$ (*Deletions*): Số từ trong câu gốc bị bỏ sót.
* $I$ (*Insertions*): Số từ tự động chèn thêm không có trong câu gốc.
* $N_R$: Tổng số từ trong câu tham chiếu ($N_R = S + D + C$, với $C$ là số từ nhận dạng đúng - Hits).
* **CER (*Character Error Rate*):** Tương tự WER nhưng tính toán trên từng ký tự đơn lẻ. Đối với tiếng Việt, CER phản ánh độ chính xác của dấu thanh và âm tiết rất tốt.

### 10.2 Phân biệt rõ 3 khái niệm đo lường trong hệ thống:
1. **Model Accuracy (WER vs Reference):** So sánh giữa chuỗi nhận dạng $H$ với câu chuẩn do con người gõ $R$ (Ground-truth transcript).
2. **Dataset Quality:** Tỷ lệ nhãn sạch, không lỗi chính tả trong tập dữ liệu.
3. **Augmentation Consistency (Drift WER / Drift CER):** So sánh giữa chuỗi nhận dạng của âm thanh gốc $H_{\text{orig}}$ và chuỗi nhận dạng của âm thanh tăng cường $H_{\text{aug}}$:
   $$\text{Drift WER} = \text{Levenshtein}(H_{\text{orig}}, H_{\text{aug}})$$
   * Ý nghĩa: Nếu Drift WER gần $0\%$, mô hình PhoWhisper cực kỳ trơ và bền bỉ trước biến dạng âm thanh đó. Nếu Drift WER cao, biến dạng đó là điểm yếu của mô hình.

---

# PHẦN 11 — QUẢN TRỊ DỮ LIỆU & BẢO ĐẢM KHÔNG NHIỄM BẨN

Ứng dụng quản trị 6 bộ dữ liệu tiếng Việt quy mô lớn qua `datasets/registry.py` và `configs/dataset_mixture.yaml`:

| Tên Bộ dữ liệu | Bản quyền (License) | Hạng chất lượng | Tên split / Tỷ trọng | Trạng thái thực tế trong App hiện tại |
| :--- | :--- | :--- | :--- | :--- |
| **VIVOS** | CC-BY-SA-4.0 | Gold (Studio read) | `train`, `test` (Weight: 25%) | **Đã kiểm chứng cấu trúc**; có metadata trong Registry. |
| **ViMD** | Academic / Non-commercial | Gold (63 tỉnh thành) | `train`, `valid`, `test` (Weight: 35%) | **Đã kiểm chứng cấu trúc**; có metadata trong Registry. |
| **BUD500** | CC-BY-NC-4.0 | Gold (Phát thanh 500h) | `train`, `validation`, `test` (Weight: 20%) | **Khai báo cấu trúc trong Registry** (Registry only). |
| **VietMed** | Y tế phi thương mại | Gold (Hội thoại khám bệnh)| `train`, `dev`, `test` (Weight: 10%) | **Khai báo cấu trúc trong Registry** (Registry only). |
| **VietSuperSpeech**| Web Scraped Fair Use | Silver (Pseudo-labeled) | `train` only (Weight: 10%) | **Khai báo cấu trúc trong Registry**; Cấm dùng để test. |
| **VLSP Restricted**| Thỏa thuận DUA chính thức | Gold (Thi đấu học thuật) | `test`/`eval` (Quarantined) | **Khai báo cách ly tuyệt đối** (Weight: 0%). |

### Hai quy tắc an toàn thép (Hard Constraints in `datasets/mixture.py`):
1. **Speaker-Disjoint Partitioning:** Toàn bộ các câu nói của cùng một người nói (`speaker_id`) chỉ được phép xuất hiện ở tập `Train` HOẶC tập `Val`, không bao giờ xuất hiện ở cả hai. Điều này đảm bảo mô hình được đánh giá về khả năng tổng quát hóa trên giọng người lạ, không học vẹt giọng người quen.
2. **Zero Test Contamination:** Mọi mã nhận diện mẫu (`sample_id`) thuộc tập Test chính thức đều bị đưa vào danh sách cách ly (quarantine). Nếu thuật toán phát hiện bất kỳ mẫu nào xuất hiện trong quá trình tạo tập train, hệ thống sẽ ném ngoại lệ nghiêm trọng: `CRITICAL DATA CONTAMINATION DETECTED` và dừng chương trình ngay lập tức.

---

# PHẦN 12 — USER WORKFLOW & DEMO SCRIPT 3 PHÚT

### 12.1 Quy trình trải nghiệm người dùng 6 bước tại Tab Studio:
1. **Bước 1: Audio Input & Validation:** Người dùng tải file âm thanh hoặc bấm nút micro trên trình duyệt để ghi âm giọng nói. Cổng 7 tiêu chí ngay lập tức kiểm tra và hiển thị trạng thái vắn tắt: `Audio validation: PASS (Duration: ...s, RMS: ..., Peak: ...)`. Người dùng có thể bấm mở chi tiết để xem tường tận 7 bài test.
2. **Bước 2: Transcribe:** Bấm nút `Transcribe` màu đen nổi bật. Hệ thống chuyển đổi âm thanh, chạy giải mã an toàn qua PhoWhisper.
3. **Bước 3: Xem Transcript & Metrics:** Văn bản nhận dạng xuất hiện trong ô `Recognized Text`. Dòng chỉ số bên dưới thông báo: `Duration: ...s | Latency: ...s | RTF: ...`. Nếu có nhập câu tham chiếu, hệ thống hiển thị thêm WER và CER.
4. **Bước 4: Augmentation Studio (Tùy chọn):** Mở rộng accordion tăng cường âm thanh. Chọn số lượng $K$ (ví dụ $K=3$), chọn các loại biến đổi mong muốn, bấm `Generate augmentations`.
5. **Bước 5: Thẩm định Kết quả & Độ trôi:** Bảng so sánh xuất hiện đầy đủ thông tin: Tên biến thể, Phép biến đổi, Thời lượng, Transcript sinh ra, Trạng thái QC 10 điểm, và Drift WER/CER so với bản gốc. Có trình phát audio để nghe lại từng biến thể.
6. **Bước 6: Export:** Bấm nút `Export session`. Hệ thống kết xuất toàn bộ dữ liệu thành gói ZIP và cung cấp link tải về.

---

### 12.2 Kịch bản Demo trực tiếp 3 phút trước Giảng viên (Live Demo Script)

* **Phút 0:00 - 0:45: Thử nghiệm giọng nói thực tế**
  * *Hành động:* Mở tab `Studio`. Bấm chọn file mẫu giọng thật có sẵn trong repo: `thanh_voice.wav` (hoặc bấm Micro ghi âm trực tiếp câu: *"Xin chào tôi tên là Thanh"*).
  * *Lời nói:* *"Thưa thầy/cô, đầu tiên em đưa âm thanh giọng nói tiếng Việt vào hệ thống. Ngay lập tức cổng kiểm định 7 tiêu chí đã kiểm tra: âm thanh đạt 2.23 giây, 16kHz mono, biên độ đỉnh 0.54, RMS 0.1008, hoàn toàn hợp lệ."*
  * *Hành động:* Bấm `Transcribe`.
  * *Lời nói:* *"Mô hình PhoWhisper hoàn thành nhận dạng trong 0.49 giây, đạt hệ số thời gian thực RTF là 0.221 trên CPU. Kết quả nhận dạng chính xác 100%: 'xin chào tôi tên là thanh.' Cơ chế kẹp token động của tụi em đã tự động tính toán giới hạn an toàn 443 tokens, loại bỏ hoàn toàn nguy cơ lỗi tràn decoder."*

* **Phút 0:45 - 1:45: Tăng cường âm thanh & Kiểm tra độ bền vững**
  * *Hành động:* Mở Accordion `Augmentation Studio`. Chọn $K=3$, giữ nguyên các phép biến đổi (Gain, Noise, Time Shift, Reverb). Bấm `Generate augmentations`.
  * *Lời nói:* *"Bây giờ em sẽ tạo $K=3$ biến thể sóng âm vật lý 1D. Hệ thống chạy bộ lọc Gauss, convolution tạo vang phòng, và kiểm định qua hàng rào 10 bài kiểm tra chất lượng âm thanh kèm mã băm SHA-256."*
  * *Hành động:* Cuộn xuống bảng `Results & Consistency Diagnostics`.
  * *Lời nói:* *"Tại bảng chẩn đoán này, thầy cô có thể thấy toàn bộ 3 biến thể đều đạt chuẩn QC PASS. Độ trôi văn bản (Drift WER) cho thấy PhoWhisper giữ vững độ chính xác thế nào trước môi trường có tiếng ồn và dội âm."*

* **Phút 1:45 - 2:30: Quản trị dữ liệu & Đóng gói xuất bản**
  * *Hành động:* Bấm `Export session`, chỉ vào file ZIP tải về. Chuyển sang Tab `Dataset Provenance`, bấm nút `Simulate partition`.
  * *Lời nói:* *"Hệ thống tự động đóng gói toàn bộ âm thanh chuẩn 16-bit PCM, file text, metadata JSON và bảng manifest CSV thành một file ZIP duy nhất. Tại tab Quản trị dữ liệu, thuật toán của em mô phỏng phân chia tập dữ liệu đảm bảo 100% người nói giữa train và val là độc lập, đồng thời kích hoạt cơ chế cách ly tuyệt đối tập test."*

* **Phút 2:30 - 3:00: Tổng kết**
  * *Lời nói:* *"Tóm lại, ứng dụng của tụi em là một giải pháp hoàn chỉnh từ tiếp nhận, tiền thẩm định, phòng vệ suy luận, tăng cường âm thanh chuẩn vật lý cho đến đóng gói dữ liệu phục vụ nghiên cứu và sản xuất."*

---

# PHẦN 13 — XỬ LÝ HÀNG LOẠT (BATCH PROCESSING)

* **Cơ chế hoạt động:** Tại Tab `Batch Processing`, người dùng có thể tải lên cùng lúc hàng chục file âm thanh.
* **Xử lý tuần tự không nghẽn RAM (Streaming Queue):** Hệ thống duyệt qua từng file theo vòng lặp, nạp và giải phóng bộ nhớ từng file, tránh việc nạp đồng thời toàn bộ dữ liệu gây tràn RAM/VRAM máy chủ.
* **Cách ly sự cố (Failure Isolation):** 
  > *Nếu một file trong danh sách batch bị lỗi (ví dụ file rỗng, file hỏng, hoặc thời lượng dưới 0.2s), hệ thống sẽ làm gì?*  
  **Trả lời:** Hệ thống **KHÔNG BỊ SẬP (CRASH)**. Cổng kiểm định 7 tiêu chí sẽ bắt lỗi cục bộ của file đó, ghi nhận trạng thái `FAIL: [Nguyên nhân]` vào bảng kết quả, và **ngay lập tức tiếp tục xử lý các file tiếp theo trong hàng đợi**. Kết quả cuối cùng vẫn được xuất ra file CSV đầy đủ cho người dùng.

---

# PHẦN 14 — ĐÓNG GÓI XUẤT BẢN KẾT QUẢ (EXPORT)

Cấu trúc thư mục được xuất bản chuẩn xác bởi `export/exporter.py`:

```text
exports/
└── asr_session_1727012345/
    ├── original/
    │   └── original.wav               # Sóng âm gốc chuẩn 16-bit PCM 16kHz
    ├── augmented/
    │   ├── aug_01.wav                 # Biến thể sóng âm 1
    │   ├── aug_02.wav                 # Biến thể sóng âm 2
    │   └── ...
    ├── transcripts/
    │   ├── original.txt               # Văn bản nhận dạng của file gốc
    │   ├── reference.txt              # Văn bản tham chiếu (nếu người dùng có nhập)
    │   ├── aug_01.txt                 # Văn bản nhận dạng của biến thể 1
    │   └── ...
    ├── metadata/
    │   ├── original.json              # Siêu dữ liệu chi tiết file gốc (RMS, Peak, Duration)
    │   ├── aug_01.json                # Lịch sử chuỗi biến đổi, hạt giống seed, kết quả 10 bài QC
    │   └── ...
    ├── manifest.csv                   # Bảng kê tổng hợp toàn bộ các mẫu trong phiên
    ├── results.csv                    # Kết quả nhận dạng và độ trôi
    └── results.json                   # Báo cáo JSON đầy đủ cấu trúc cây
```
Sau khi tạo cây thư mục, mô-đun sử dụng thư viện `zipfile` (với chuẩn nén `ZIP_DEFLATED`) để nén toàn bộ thành file `exports/asr_session_1727012345.zip` để người dùng tải trực tiếp trên trình duyệt.

---

# PHẦN 15 — THIẾT KẾ GIAO DIỆN UI/UX

* **Định hướng thiết kế:** Giao diện được tái cấu trúc hoàn toàn theo phong cách **Minimal, Professional, Academic, Text-First**.
* **Loại bỏ 100% biểu tượng cảm xúc (No Emojis, No Unicode Icons):** Toàn bộ các icon trang trí rườm rà (`🎙️`, `⚡`, `🔍`, `⚙️`, `📊`, `✅`, `❌`) đều bị loại bỏ triệt để. Giao diện sử dụng typography hệ thống gọn gàng, thanh lịch như một công cụ kỹ thuật chuyên sâu của phòng lab.
* **Cấu trúc 4 Tab tinh gọn:**
  1. `Studio`: Toàn bộ quy trình 6 bước trên một màn hình duy nhất, giảm thiểu thao tác chuyển tab gây mất tập trung.
  2. `Batch Processing`: Bảng hàng đợi xử lý âm thanh số lượng lớn.
  3. `Dataset Provenance`: Sổ đăng ký nguồn gốc dữ liệu và trình giả lập chia tách người nói.
  4. `Settings & Diagnostics`: Quản lý nạp model, cấu hình giải mã, theo dõi phần cứng và xóa cache bộ nhớ.

---

# PHẦN 16 — TESTING & VERIFICATION (KIỂM THỬ PHẦN MỀM)

Ứng dụng sở hữu bộ kiểm thử tự động toàn diện chạy qua `pytest` trong thư mục `tests/`:

```text
============================= test session starts =============================
Platform: Win32 -- Python 3.14.0, pytest-9.1.1
Rootdir: D:\Vietnamese_ASR_Week6\vietnamese_asr_app
Collected 31 items: 31 PASSED in 12.65s (100% Success Rate)
```

### Bảng phân tích chi tiết 31 bài Test:

| File Test | Số lượng test | Nội dung kỹ thuật được kiểm chứng | Phân loại kiểm thử |
| :--- | :---: | :--- | :--- |
| `test_decoding_regression.py` | 6 tests | Giới hạn giải mã an toàn: Case A (prompt=4, req=448 $\to$ clamped 443); Case B (req=128 $\to$ 128); Case C (prompt dài 32); Case D (kiến trúc max=256, 100); Case E (tuple micro); Inference model thật không crash khi req=448. | **Integration / Regression Test** |
| `test_inference.py` | 7 tests | Chuẩn hóa tiếng Việt NFC, xóa dấu câu; Cổng 7 tiêu chí: bắt tín hiệu sạch (PASS), bắt tín hiệu im lặng (FAIL), bắt file quá ngắn (FAIL), bắt file quá dài (FAIL), bắt file chứa NaN (FAIL). | **Unit Test** |
| `test_qc.py` | 5 tests | Hàng rào 10 bài QC: Âm thanh chuẩn (PASS), Âm thanh im lặng (FAIL), Âm thanh chứa NaN/Inf (FAIL), Âm thanh bị cắt ngọn clipping vượt 0.1% (FAIL), Sai thời lượng (FAIL). | **Unit Test** |
| `test_transforms.py` | 5 tests | 5 phép biến đổi sóng âm: Gain, Additive Noise SNR, Time Shift, Frequency Filter, Synthetic Reverb (đảm bảo tính hữu hạn số học, không méo biên độ, có tính lặp lại theo seed). | **Unit Test** |
| `test_manifest.py` | 3 tests | Thuật toán phân chia tập dữ liệu: 0% rò rỉ người nói giữa train và val; Ngoại lệ chặn nhiễm bẩn tập test; Bộ lọc cận thời lượng 0.2s - 30s. | **Unit Test** |
| `test_metrics.py` | 4 tests | Độ chính xác thuật toán Levenshtein WER, CER chuỗi đồng nhất (0%), chuỗi có lỗi S/D/I; Đánh giá độ trôi văn bản (Drift WER/CER). | **Unit Test** |
| `test_export.py` | 1 test | Tạo cây thư mục phiên làm việc đầy đủ, lưu WAV PCM 16-bit, tạo JSON/CSV và nén thành công file ZIP. | **Integration Test** |

> **Lưu ý trung thực:** 31 bài test này là bài kiểm tra tính đúng đắn của logic phần mềm, xử lý ngoại lệ và độ ổn định suy luận (Software Engineering & Decoding Logic Verification). Đây **không phải** là báo cáo đo độ chính xác tổng thể (Benchmark Accuracy Report) của mô hình trên tập test 100 giờ.

---

# PHẦN 17 — HIỆU NĂNG THỰC TẾ (BENCHMARK ĐÃ ĐO ĐƯỢC)

Dưới đây là số liệu đo lường thực tế $100\%$ từ mã nguồn và file kiểm thử giọng thật `run_real_voice_test.py` trên môi trường máy tính báo cáo:

* **Môi trường phần cứng:** CPU Intel Core / AMD x86_64, RAM hệ thống, không có GPU CUDA kích hoạt trong bản build PyTorch hiện tại (`Torch: 2.10.0+cpu | Precision: float32`).
* **Mẫu âm thanh thử nghiệm:** `thanh_voice.wav` (Giọng đọc tiếng Việt tự nhiên: *"Xin chào tôi tên là Thanh"*).
  * Số lượng mẫu: $35,712$ mẫu tại tần số $16,000\text{ Hz}$.
  * Thời lượng âm thanh thực tế: $2.232\text{ giây}$.
  * Kênh: Mono 1 kênh.
  * Chỉ số tín hiệu: $\text{RMS} = 0.1008$, $\text{Peak} = 0.5370$.
* **Kết quả đo lường suy luận (Inference Metrics):**
  * Mô hình nạp: `vinai/PhoWhisper-tiny` (và `PhoWhisper-small`).
  * Thời gian nhận dạng (Latency): **$0.49\text{ giây}$** (trên CPU).
  * Hệ số thời gian thực (Real-Time Factor - RTF): **$0.221$** (RTF $< 1.0$ nghĩa là hệ thống chạy nhanh hơn tốc độ nói thực tế gấp $4.5$ lần ngay cả trên CPU thông thường).
  * Yêu cầu tokens ban đầu: $448$ tokens.
  * Token được kẹp an toàn: $443$ tokens (Tổng chiều dài tối đa: $447 \le 448$).
  * Kết quả nhận dạng thô: `'xin chào tôi tên là thanh.'`
  * Kết quả chuẩn hóa NFC: `'xin chào tôi tên là thanh'`
  * Độ chính xác từ vựng (Word Accuracy): **$100\%$** (nhận diện chính xác toàn bộ 6/6 từ).
  * Hiện tượng lỗi vị trí decoder: **HOÀN TOÀN BẰNG KHÔNG (ZERO EXCEPTIONS).**

---

# PHẦN 18 — PHÂN BIỆT RÕ RÀNG: APP NÀY VÀ THỰC NGHIỆM BENCHMARK CLOUD

| Tiêu chí so sánh | Dự án Thực nghiệm Benchmark Cloud (`ASR_FULL_BENCHMARK_CLOUD_v*.ipynb`) | Ứng dụng Studio Sản xuất (`vietnamese_asr_app/`) |
| :--- | :--- | :--- |
| **Mục tiêu cốt lõi** | Nghiên cứu khoa học, huấn luyện fine-tune mô hình PhoWhisper trên đám mây (Google Colab GPU), đo đạc WER/CER học thuật trên tập test chuẩn. | Xây dựng phần mềm ứng dụng hoàn chỉnh, phục vụ người dùng cuối, phòng vệ suy luận, kiểm chuẩn âm thanh và tăng cường dữ liệu. |
| **Môi trường thực thi** | Google Colab GPU (Tesla T4 / A100), Linux. | Máy tính cá nhân / Server ứng dụng cục bộ, giao diện Web Gradio. |
| **Dữ liệu sử dụng** | Tải và phân chia trực tiếp hàng nghìn file từ Hugging Face (`ViMD`, `VIVOS`) theo frozen manifest. | Xử lý âm thanh người dùng đưa vào thời gian thực qua Mic/Upload, kèm dữ liệu demo nội bộ (`thanh_voice.wav`). |
| **Model trọng tâm** | Quá trình huấn luyện fine-tune bị gián đoạn ở Epoch 5 (đã lưu checkpoint epoch 1 đến 4). | Sử dụng trực tiếp **Pretrained Model chính thức từ VinAI Research** (`vinai/PhoWhisper-small/tiny/base`). |

> **Câu hỏi mấu chốt:** *Model dùng trong app có phải chính xác là checkpoint vừa train trong benchmark notebook không?*  
> **Trả lời thẳng thắn:** **KHÔNG.** Model trong app hiện tại nạp trực tiếp pretrained weights của tác giả VinAI từ Hugging Face Hub. Ứng dụng app được thiết kế độc lập, tách biệt hoàn toàn để bảo toàn nguyên vẹn các artifact thực nghiệm benchmark và không làm xáo trộn kết quả nghiên cứu.

---

# PHẦN 19 — 25 CÂU HỎI VÀ TRẢ LỜI PHẢN BIỆN TRỰC TIẾP

**Q1: Tại sao nhóm chọn mô hình PhoWhisper mà không dùng Whisper gốc của OpenAI?**  
* *Trả lời:* Whisper gốc của OpenAI được huấn luyện đa ngôn ngữ, dữ liệu tiếng Việt chỉ chiếm một phần rất nhỏ dẫn đến hiện tượng sai dấu thanh và lỗi chính tả khi gặp từ ngữ địa phương. PhoWhisper của VinAI được huấn luyện chuyên sâu trên lượng lớn dữ liệu tiếng Việt chất lượng cao, giúp giảm đáng kể WER trên các tập dữ liệu bản địa như VIVOS và ViMD.  
* *Tại sao:* Thể hiện sự hiểu biết về đặc trưng ngôn ngữ đơn lập có thanh điệu.

**Q2: Mô hình này có phải do nhóm tự huấn luyện từ đầu không?**  
* *Trả lời:* Dạ không, mô hình là pretrained model của VinAI Research được nhóm tích hợp vào hệ thống. Đóng góp của nhóm là kỹ thuật hệ thống: xây dựng cổng kiểm soát âm thanh 7 tiêu chí, sửa lỗi tràn decoder Whisper, xây dựng pipeline tăng cường 1D và kiểm soát chất lượng 10 tiêu chí.  
* *Tại sao:* Thể hiện tính trung thực học thuật tuyệt đối.

**Q3: Nếu dùng model có sẵn thì đóng góp của nhóm là gì ngoài cái giao diện?**  
* *Trả lời:* Giao diện chỉ chiếm 10% công việc. 90% đóng góp kỹ thuật nằm ở tầng backend: phát hiện và khắc phục lỗi crash decoder bằng thuật toán kẹp token động, xử lý âm thanh dài $>30$s, xây dựng thuật toán tăng cường trên miền thời gian có kiểm chuẩn vật lý, và thuật toán phân chia tập dữ liệu đảm bảo không rò rỉ người nói.  
* *Tại sao:* Bảo vệ khối lượng công việc kỹ thuật đồ sộ của dự án.

**Q4: Mô hình nhận đầu vào là gì?**  
* *Trả lời:* Mô hình nhận ma trận 80 kênh log-Mel spectrogram được tính toán từ tín hiệu âm thanh 1D có tần số lấy mẫu chuẩn 16,000 Hz với cửa sổ FFT 25ms và bước nhảy 10ms.  
* *Tại sao:* Trả lời đúng bản chất xử lý tín hiệu của Whisper.

**Q5: Tại sao bắt buộc âm thanh phải đưa về 16 kHz?**  
* *Trả lời:* Vì theo định lý lấy mẫu Nyquist, tần số 16 kHz cho phép bảo toàn dải tần âm thanh tiếng nói con người lên tới 8 kHz (vốn chứa toàn bộ các formants $F_1, F_2, F_3$ quyết định nguyên âm và phụ âm), đồng thời đây là kích thước đầu vào cố định của bộ lọc Mel trong PhoWhisper.  
* *Tại sao:* Đúng kiến thức cơ sở xử lý tín hiệu số.

**Q6: Tại sao bắt buộc phải chuyển sang Mono 1 kênh?**  
* *Trả lời:* Mô hình ASR nhận diện nội dung ngữ nghĩa tiếng nói, không cần thông tin vị trí không gian âm thanh. Ép về Mono giúp giảm 50% khối lượng tính toán và loại bỏ hiện tượng triệt tiêu pha giữa hai micro.  
* *Tại sao:* Tối ưu hóa tính toán và chuẩn hóa dữ liệu.

**Q7: Bộ giải mã sinh transcript như thế nào?**  
* *Trả lời:* Decoder hoạt động theo cơ chế tự hồi quy (autoregressive). Bắt đầu từ các token mồi chỉ định ngôn ngữ tiếng Việt và tác vụ nhận dạng, tại mỗi bước, mạng dự đoán phân phối xác suất từ tiếp theo, chọn từ qua thuật toán Greedy (hoặc Beam Search), rồi đưa từ đó ngược lại đầu vào cho bước kế tiếp cho đến khi gặp token kết thúc `<|endoftranscript|>`.  
* *Tại sao:* Nắm vững cơ chế sinh của Transformer Decoder.

**Q8: Mô hình Whisper khác gì các mô hình ASR truyền thống như HMM-GMM hay CTC?**  
* *Trả lời:* Mô hình truyền thống cần tách biệt mô hình âm học (Acoustic Model), mô hình phát âm (Pronunciation Lexicon) và mô hình ngôn ngữ (Language Model) với quá trình gióng hàng phức tạp. Whisper là kiến trúc Sequence-to-Sequence đầu-cuối (End-to-End), học trực tiếp ánh xạ từ phổ âm thanh sang ký tự văn bản.  
* *Tại sao:* Hiểu rõ sự phát triển của công nghệ ASR hiện đại.

**Q9: Tại sao nhóm lại làm Waveform Augmentation mà không dùng trực tiếp SpecAugment?**  
* *Trả lời:* SpecAugment che các dải trên spectrogram trong quá trình huấn luyện GPU, nhưng không tạo ra âm thanh nghe được trong thực tế. Nhóm biến đổi trên sóng âm 1D (Waveform) để tạo ra các file âm thanh vật lý hoàn chỉnh, có thể phát lại kiểm tra, đóng gói xuất bản và dùng để đánh giá độ trôi (Drift) của bất kỳ hệ thống nào.  
* *Tại sao:* Nêu bật tính ứng dụng thực tiễn của phần mềm.

**Q10: Tăng cường âm thanh (Augmentation) có nguy cơ làm hỏng dữ liệu không?**  
* *Trả lời:* Có, nếu tăng âm lượng quá mức sẽ gây rè đứt sóng (clipping), hoặc cộng nhiễu quá lớn làm mất hẳn tiếng nói. Do đó, nhóm bắt buộc mọi file biến đổi phải đi qua hàng rào 10 bài kiểm tra Audio QC (giới hạn clipping $<0.1\%$, RMS $>1e-5$). Nếu vi phạm, hệ thống sẽ đánh dấu FAIL và loại bỏ.  
* *Tại sao:* Chứng minh hệ thống có cơ chế kiểm soát chất lượng tự động.

**Q11: Tại sao cần tính toán mã băm SHA-256 trong bài QC số 10?**  
* *Trả lời:* Để đảm bảo tính toàn vẹn dữ liệu và khả năng truy vết nguồn gốc (Data Provenance). Mỗi file âm thanh biến thể có một chữ ký mật mã duy nhất, đảm bảo tính lặp lại của thực nghiệm khoa học.  
* *Tại sao:* Tư duy kỹ thuật phần mềm chuẩn chỉ.

**Q12: WER và CER khác nhau thế nào, khi nào nên dùng cái nào?**  
* *Trả lời:* WER tính trên đơn vị từ, còn CER tính trên đơn vị ký tự. Trong tiếng Việt, một từ sai dấu thanh (ví dụ "nhà" thành "nhá") chỉ tính là 1 lỗi ký tự trên CER, nhưng bị tính là sai nguyên 1 từ trên WER. CER phù hợp để đánh giá âm vị chi tiết, còn WER phản ánh sát hơn mức độ hiểu văn bản của người dùng.  
* *Tại sao:* Nắm vững bản chất ngôn ngữ học tính toán.

**Q13: Mô hình này có xử lý được tiếng địa phương (Dialect) không?**  
* *Trả lời:* PhoWhisper được huấn luyện trên dữ liệu đa dạng nên nhận dạng tốt giọng Bắc và Nam chuẩn. Với các phương ngữ nặng miền Trung, độ chính xác có thể giảm. Nhóm đã thiết kế sẵn mô-đun tích hợp tập dữ liệu ViMD (chứa 63 tỉnh thành) trong cấu hình để sẵn sàng cho việc thích ứng miền sau này.  
* *Tại sao:* Đánh giá thực tế, không tâng bốc mô hình.

**Q14: Ứng dụng này có nhận dạng Real-time trực tiếp dạng streaming (vừa nói vừa hiện chữ) không?**  
* *Trả lời:* Hiện tại chưa, ứng dụng hoạt động theo cơ chế Pseudo-real-time (ghi âm theo đoạn/utterance rồi giải mã ngay lập tức). Tuy nhiên với hệ số RTF đạt 0.221 trên CPU, một câu nói 2 giây chỉ mất chưa tới 0.5 giây để hiện kết quả, hoàn toàn đáp ứng tốt nhu cầu tương tác.  
* *Tại sao:* Trả lời đúng kiến trúc hiện tại, không ngụy biện.

**Q15: Nếu một file trong danh sách Batch bị hỏng thì hệ thống có dừng lại không?**  
* *Trả lời:* Dạ không. Cổng kiểm định 7 tiêu chí sẽ bắt lỗi riêng của file đó, ghi nhận thông báo lỗi vào hàng kết quả tương ứng, và vòng lặp streaming sẽ tiếp tục xử lý các file lành lặn tiếp theo mà không làm crash hệ thống.  
* *Tại sao:* Nêu bật tính năng chịu lỗi (fault-tolerant).

**Q16: Khi file âm thanh dài hơn 30 giây thì app xử lý ra sao?**  
* *Trả lời:* Cổng 7 tiêu chí sẽ kích hoạt chế độ chunking. Hệ thống chia sóng âm thành các cửa sổ tuần tự 30 giây ($480,000$ mẫu), đưa từng cửa sổ qua Whisper dưới chế độ `inference_mode`, rồi nối chuỗi văn bản lại theo thứ tự thời gian.  
* *Tại sao:* Giải quyết đúng bài toán giới hạn cửa sổ ngữ cảnh của Whisper.

**Q17: Tại sao cần kiểm tra RMS trong cổng kiểm định âm thanh?**  
* *Trả lời:* RMS đo năng lượng hữu hiệu của tín hiệu. Nếu RMS $< 1e-5$, file âm thanh thực chất là khoảng lặng hoàn toàn (silence). Việc chặn file im lặng giúp tránh việc mô hình tự sinh các từ lặp lại ảo giác vô nghĩa (hallucination).  
* *Tại sao:* Hiểu rõ hiện tượng hallucination của các mô hình Sequence-to-Sequence.

**Q18: Kiểm tra Clipping ratio là làm gì?**  
* *Trả lời:* Là đếm tỷ lệ các mẫu âm thanh chạm ngưỡng biên độ tuyệt đối $\ge 0.999$. Nếu tỷ lệ này vượt quá $0.1\%$, sóng âm đã bị biến dạng thành sóng vuông (square wave), làm mất thông tin hài âm của giọng nói, hệ thống sẽ từ chối xử lý.  
* *Tại sao:* Đúng kiến thức vật lý âm thanh.

**Q19: Nhóm có tự huấn luyện mô hình ASR nào trong app chưa?**  
* *Trả lời:* Dạ trong thư mục app này, nhóm chưa chạy full training mà chỉ chạy kiểm thử dry-run cho script `train.py` để xác thực cơ chế lưu checkpoint và phục hồi trạng thái hạt giống ngẫu nhiên (RNG state).  
* *Tại sao:* Tuyệt đối trung thực, tránh bị giảng viên bắt bẻ.

**Q20: Nhóm có tinh chỉnh (Fine-tuning) PhoWhisper trong app này chưa?**  
* *Trả lời:* Dạ chưa, trong app nhóm sử dụng trực tiếp trọng số gốc của PhoWhisper. Việc nghiên cứu thực nghiệm fine-tune được tiến hành ở một nhánh nghiên cứu benchmark riêng biệt trên môi trường đám mây.  
* *Tại sao:* Phân định ranh giới rõ ràng.

**Q21: Mô hình trong app đã được đánh giá chính thức trên toàn bộ tập test VIVOS/ViMD chưa?**  
* *Trả lời:* Trong app đã có script `scripts/evaluate.py` hỗ trợ đánh giá tự động trên file manifest. Nhóm đã kiểm tra tích hợp thành công trên các mẫu kiểm thử giọng thật, nhưng chưa chạy toàn bộ hàng chục nghìn mẫu test tập trung trên máy cá nhân vì giới hạn phần cứng CPU.  
* *Tại sao:* Giải thích thuyết phục dựa trên giới hạn tài nguyên.

**Q22: Kết quả nào trong app đã được kiểm chứng thực tế 100%?**  
* *Trả lời:* Toàn bộ 31 bài test tự động (`pytest tests/`), quy trình nhận dạng giọng thật trên file `thanh_voice.wav` với RTF 0.221, thuật toán kẹp token chống tràn decoder, quy trình sinh biến thể 1D và kiểm định 10 bài QC.  
* *Tại sao:* Nêu các bằng chứng thực tế đã chạy có log ghi lại.

**Q23: Hạn chế lớn nhất hiện tại của ứng dụng là gì?**  
* *Trả lời:* Hạn chế hiện tại là chưa chạy trên môi trường GPU máy chủ nội bộ (đang chạy CPU float32), chưa hỗ trợ nhận dạng streaming theo thời gian thực (từng từ ngắt quãng), và thuật toán cắt file 30 giây dài hiện tại cắt cố định theo thời gian chứ chưa sử dụng thuật toán dò tìm khoảng lặng (VAD) để cắt theo ranh giới câu nói.  
* *Tại sao:* Người làm kỹ thuật giỏi là người biết rõ hạn chế của hệ thống mình làm.

**Q24: Nếu có thêm 1 tháng để phát triển, nhóm sẽ làm gì tiếp theo?**  
* *Trả lời:* Nhóm sẽ: 1. Tích hợp thuật toán Voice Activity Detection (Silero VAD) để phân đoạn âm thanh thông minh theo hơi thở người nói; 2. Đưa mô hình lên cụm GPU và lượng tử hóa (Quantization INT8/FP16 qua CTranslate2/ONNX) để tăng tốc gấp 4 lần; 3. Chạy toàn bộ quy trình fine-tune PhoWhisper trên hỗn hợp đa phương ngữ ViMD.  
* *Tại sao:* Thể hiện tầm nhìn phát triển sản phẩm chuyên nghiệp.

**Q25: Tóm lại giá trị lớn nhất mà dự án này mang lại là gì?**  
* *Trả lời:* Là một giải pháp công nghệ toàn diện và an toàn. Dự án biến một mô hình nghiên cứu thô thành một phần mềm có khả năng phòng thủ trước các lỗi dữ liệu thực tế, cung cấp công cụ tạo dữ liệu âm thanh nhân tạo có kiểm định vật lý, và tuân thủ các chuẩn mực khoa học nghiêm ngặt.  
* *Tại sao:* Câu chốt ấn tượng, làm nổi bật giá trị đề tài.

---

# PHẦN 20 — CHIẾN LƯỢC TRẢ LỜI CÁC CÂU HỎI XOÁY

> [!TIP]
> Khi giảng viên hỏi xoáy để dồn ép sinh viên, hãy giữ bình tĩnh, thừa nhận sự thật một cách tự tin và hướng câu trả lời về giá trị kỹ thuật phần mềm (Software Engineering).

### Xoáy 1: *“Nhóm em có thực sự xây dựng model không, hay chỉ tải thư viện về gọi vài dòng code?”*
* **Cách đáp trả:**  
  *"Dạ thưa thầy, về mặt mô hình nền tảng (Foundation Model), nhóm em không tự xây dựng mạng nơ-ron từ đầu vì việc huấn luyện một mô hình hàng trăm triệu tham số như PhoWhisper đòi hỏi hàng nghìn giờ dữ liệu bản quyền và cụm máy chủ GPU hàng tỷ đồng của các viện nghiên cứu lớn như VinAI. Tuy nhiên, việc đưa mô hình đó vào thực tế không chỉ là 'gọi vài dòng code'. Nếu chỉ gọi code mặc định của Hugging Face, chương trình sẽ lập tức sập khi gặp lỗi tràn $448$ tokens của Whisper, sẽ sinh ra từ lặp vô nghĩa khi người dùng nạp file im lặng, hoặc mất dữ liệu khi file quá 30 giây. Nhóm em tập trung giải quyết bài toán kỹ thuật hệ thống: xây dựng các thuật toán tiền thẩm định, phòng vệ suy luận, kiểm chuẩn dữ liệu và công cụ tăng cường âm thanh vật lý đạt chuẩn kiểm thử tự động."*

### Xoáy 2: *“Nếu chỉ dùng pretrained model thì yếu tố AI của đề tài này nằm ở đâu?”*
* **Cách đáp trả:**  
  *"Dạ, yếu tố AI của đề tài nằm ở việc tích hợp, làm chủ và thẩm định mô hình học sâu trong điều kiện biên (edge cases). Cụ thể: nhóm em làm chủ cơ chế mã hóa log-Mel spectrogram, cơ chế giải mã tự hồi quy của Transformer, thuật toán tính toán ngân sách token an toàn, thuật toán đo đạc độ trôi ngữ nghĩa văn bản (Drift WER) khi âm thanh bị biến dạng âm học, và mô hình hóa bài toán phân chia dữ liệu huấn luyện độc lập người nói (Speaker-disjoint) - một bài toán cốt lõi trong học máy."*

### Xoáy 3: *“Tại sao nhóm không tự train lấy một model nhỏ cho riêng mình để chứng minh năng lực?”*
* **Cách đáp trả:**  
  *"Dạ thưa cô, trong mã nguồn `scripts/train.py`, nhóm em đã xây dựng đầy đủ kiến trúc pipeline huấn luyện, có cơ chế quản lý checkpoint, lưu trữ optimizer AdamW và khôi phục 100% trạng thái hạt giống ngẫu nhiên (RNG) đã được kiểm chứng qua bài test dry-run. Nhưng với góc độ ứng dụng thực tế cho người Việt, việc cố tình tự train một model nhỏ trên vài chục tiếng dữ liệu sẽ cho kết quả nhận dạng rất tệ (WER có thể lên tới 40-50%), không có giá trị thực tiễn. Nhóm em lựa chọn cách tiếp cận chuẩn mực của công nghiệp hiện nay: đứng trên vai người khổng lồ, sử dụng mô hình nền tảng tốt nhất của Việt Nam và dồn tâm huyết xây dựng tầng giải pháp ứng dụng hoàn thiện, an toàn."*

---

# PHẦN 21 — KỊCH BẢN THUYẾT TRÌNH 5 PHÚT

*(Chuẩn bị slide hoặc màn hình demo sẵn sàng tại `http://127.0.0.1:7860`)*

**Kính chào quý thầy cô và hội đồng, sau đây em xin đại diện nhóm trình bày dự án: "Vietnamese Speech-to-Text & Audio Augmentation Studio".**

* **Phút 1: Bài toán và Lý do phát triển**  
  Nhận dạng tiếng nói tiếng Việt là bài toán có độ phức tạp cao do tính chất ngôn ngữ có 6 thanh điệu và sự đa dạng của 63 phương ngữ. Hiện nay, nhiều giải pháp demo ASR chỉ dừng lại ở mức gọi mô hình mẫu trong điều kiện lý tưởng. Khi gặp âm thanh thực tế như tạp âm, tiếng vang, file rỗng hoặc file vượt quá 30 giây, hệ thống rất dễ bị sập hoặc sinh văn bản ảo giác. Đề tài của tụi em được phát triển với mục tiêu giải quyết trọn vẹn bài toán kỹ thuật hệ thống xung quanh mô hình ASR: từ tiếp nhận, tiền kiểm tra tín hiệu, phòng vệ suy luận an toàn, đến tăng cường dữ liệu âm thanh và quản trị nguồn dữ liệu khoa học.

* **Phút 2: Bản chất Mô hình và Đóng góp Kỹ thuật**  
  Về mô hình nhận dạng, hệ thống tích hợp dòng mô hình PhoWhisper của VinAI Research, cụ thể là bản `PhoWhisper-small` với 244 triệu tham số. Đây là pretrained model tiên tiến nhất cho tiếng Việt hiện nay. Đóng góp trọng tâm của nhóm em nằm ở tầng kỹ thuật ứng dụng:
  Thứ nhất, tụi em phát hiện và xử lý dứt điểm lỗi tràn kiến trúc 448 vị trí giải mã của Whisper bằng thuật toán kẹp token động `compute_safe_max_new_tokens`.
  Thứ hai, xây dựng Cổng kiểm định âm thanh 7 tiêu chí tự động loại bỏ âm thanh hỏng, file im lặng và file vượt ngưỡng biên độ.
  Thứ ba, xây dựng Studio tăng cường dữ liệu trực tiếp trên sóng âm 1D với 7 phép biến đổi vật lý, kèm Hàng rào 10 bài kiểm tra chất lượng âm thanh nghiêm ngặt có băm mã SHA-256.

* **Phút 3: Kiến trúc và Quy trình Hoạt động**  
  Kiến trúc hệ thống được chia làm 4 khối rõ rệt: Tầng giao diện người dùng tối giản (Text-first UI); Tầng tiền xử lý âm thanh chuẩn 16kHz Mono; Tầng suy luận nhận dạng nạp mô hình dạng Singleton Cache để tránh nạp lại bộ nhớ; và Tầng kiểm chuẩn đo đạc độ trôi văn bản (Drift WER/CER) bằng khoảng cách Levenshtein.

* **Phút 4: Kết quả Kiểm chứng Thực tế**  
  Toàn bộ hệ thống phần mềm của tụi em được kiểm chứng độc lập qua 31 bài unit và integration test tự động bằng `pytest`, đạt tỷ lệ thành công 100%. Trên máy tính thử nghiệm chạy CPU, hệ thống nhận dạng file âm thanh thực tế đạt hệ số thời gian thực RTF là 0.221, nghĩa là chỉ mất chưa đầy nửa giây để nhận dạng xong một câu nói 2.2 giây với độ chính xác từ vựng tuyệt đối.

* **Phút 5: Kết luận**  
  Tóm lại, dự án không chỉ mang lại một công cụ nhận dạng tiếng Việt nhanh, chính xác cho người dùng cuối mà còn là một nền tảng tạo dữ liệu và thẩm định độ bền vững của mô hình ASR cho các kỹ sư dữ liệu. Em xin chân thành cảm ơn quý thầy cô và sẵn sàng lắng nghe các câu hỏi phản biện.

---

# PHẦN 22 — KỊCH BẢN THUYẾT TRÌNH 10 PHÚT CHUYÊN SÂU

*(Dành cho buổi báo cáo có thời lượng dài, giảng viên yêu cầu đi sâu vào chi tiết kỹ thuật)*

1. **Mở đầu & Tổng quan vấn đề (1.5 phút):** Phân tích sâu về các thách thức trong ASR tiếng Việt (thanh điệu, biến âm phương ngữ, từ đồng âm). Chỉ ra khoảng cách giữa mô hình nghiên cứu lý thuyết trong phòng lab và bài toán ứng dụng thực tế.
2. **Kiến trúc mô hình PhoWhisper & Lựa chọn công nghệ (2 phút):** Giải thích cấu trúc Encoder-Decoder Transformer của Whisper. Tại sao chọn ma trận 80 kênh log-Mel. Phân tích sự vượt trội của PhoWhisper so với Whisper gốc khi xử lý nguyên âm và phụ âm tiếng Việt.
3. **Phát hiện & Giải quyết lỗi kiến trúc Decoder (1.5 phút):** Đi sâu vào mã nguồn `asr/inference.py`. Giải thích vì sao `decoder_input_ids` chiếm 4 tokens, vì sao yêu cầu 448 tokens sẽ làm sập chương trình. Trình bày công thức toán học kẹp an toàn với margin=1.
4. **Hệ thống Audio Validation Gate & QC Pipeline (2 phút):** Trình bày chi tiết bảng 7 tiêu chí đầu vào và bảng 10 tiêu chí QC đầu ra. Giải thích tại sao phải đo RMS để diệt khoảng lặng, tại sao giới hạn tỷ lệ clipping dưới 0.1%, và ý nghĩa của việc băm SHA-256 trên chuỗi byte sóng âm.
5. **Waveform Augmentation & Drift Measurement (1.5 phút):** Giải thích 7 phép biến đổi 1D miền thời gian. Nêu rõ sự khác biệt giữa đo đạc độ chính xác model (WER vs Ref) và đo đạc độ trôi ngữ nghĩa (Drift WER vs Original).
6. **Demo trực tiếp & Kết quả kiểm thử (1 phút):** Thao tác trực tiếp trên giao diện: ghi âm micro, chạy nhận dạng, sinh biến thể, xuất file ZIP. Chiếu kết quả 31 bài test tự động.
7. **Hạn chế và Hướng phát triển (0.5 phút):** Trình bày trung thực về giới hạn CPU hiện tại, kế hoạch tích hợp Silero VAD và lượng tử hóa mô hình trong tương lai.

---

# PHẦN 23 — ONE-PAGE CHEAT SHEET

*(In ra giấy hoặc mở trên điện thoại đọc lướt trước khi bước vào phòng thi)*

* **Dự án:** `vietnamese_asr_app` — Vietnamese Speech-to-Text & Data Augmentation Studio.
* **Mục tiêu:** Ứng dụng nhận dạng tiếng Việt hoàn chỉnh, chống crash, tiền thẩm định tín hiệu, tăng cường sóng âm 1D có QC.
* **Model sử dụng:** Pretrained `vinai/PhoWhisper-small` (244M params, Hugging Face Hub). Không tự train from scratch.
* **Đóng góp kỹ thuật:** 7-point Audio Gate; Sửa lỗi tràn 448 tokens (`safe max_new_tokens`); Chunking 30s; 7 phép tăng cường 1D; 10-check QC; Băm SHA-256; Phân chia không rò rỉ người nói.
* **Chuẩn âm thanh đầu vào:** 16,000 Hz, 1 kênh Mono, định dạng số thực `float32` trong khoảng $[-1.0, 1.0]$.
* **Cổng thẩm định 7 điểm:** Signal exist, Finite real, Mono, Duration $\ge 0.2$s, Duration $\le 30$s, RMS $\ge 1e-5$, Peak $\le 1.2$.
* **Bug decoder đã fix:** Whisper kẹp cứng 448 vị trí. Có 4 tokens prompt $\implies$ Kẹp an toàn $\min(\text{req}, 448 - 4 - 1) = 443$.
* **Hàng rào 10 bài QC:** Decodable, Finite, No NaN, No Inf, Peak $\le 1.05$, Clipping $\le 0.1\%$, RMS $\ge 1e-5$, $0.2\text{s} \le \text{Dur} \le 30\text{s}$, 16kHz Mono, SHA-256.
* **Các phép tăng cường 1D:** Gain, Calibrated Noise SNR, Time Shift, Time Stretch, Pitch Shift, Room Reverb, Bandpass (300-3400Hz).
* **Chỉ số đo lường:** WER (Word Error Rate), CER (Character Error Rate), Drift WER (Độ trôi văn bản sau biến đổi âm thanh).
* **Quản trị dữ liệu:** 6 dataset (VIVOS, ViMD, BUD500, VietMed, VietSuperSpeech, VLSP); Khóa tập test tuyệt đối; Độc lập người nói.
* **Kiểm thử tự động:** Bộ 31 bài tests `pytest` trong `tests/` — Đạt $31/31$ PASSED ($100\%$).
* **Hiệu năng thực tế:** Giọng thật `thanh_voice.wav` (2.23s) chạy trên CPU mất $0.49$s $\implies$ $\text{RTF} = 0.221$ (nhanh gấp 4.5 lần giọng nói thật).
* **Khác biệt với Benchmark:** Benchmark là notebook train đám mây; App là phần mềm độc lập dùng trọng số pretrained chính thức của VinAI.

---

# PHẦN 24 — FACT SHEET

| Thành phần kỹ thuật | Giá trị thực tế trong Source Code | Nguồn chứng minh trong Repo |
| :--- | :--- | :--- |
| **Tên Model mặc định** | `vinai/PhoWhisper-small` (hỗ trợ cả tiny, base, medium) | `configs/model.yaml` |
| **Nguồn gốc Model** | Pretrained tải từ Hugging Face Hub (Tác giả: VinAI Research) | `asr/model.py : ASRModelManager` |
| **Kiến trúc mạng** | Transformer Encoder-Decoder (OpenAI Whisper architecture) | `transformers.WhisperForConditionalGeneration` |
| **Tần số lấy mẫu (Sample rate)** | 16,000 Hz | `configs/app.yaml`, `asr/audio.py` |
| **Số kênh âm thanh** | 1 kênh (Mono) | `asr/audio.py : load_audio` |
| **Kiểu dữ liệu xử lý** | `numpy.float32` chuẩn hóa $[-1.0, 1.0]$ | `asr/audio.py : load_audio` |
| **Ngôn ngữ & Tác vụ** | Language: `"vi"`, Task: `"transcribe"` | `configs/model.yaml`, `asr/inference.py` |
| **Prompt Tokens ban đầu** | 4 tokens (`<|startoftranscript|>`, `<|vi|>`, `<|transcribe|>`, `<|notimestamps|>`) | `asr/inference.py` |
| **Giới hạn vị trí Decoder** | 448 tokens (`max_target_positions`) | `model.config.max_target_positions` |
| **Ngân sách Token tối đa an toàn** | 443 tokens (khi prompt=4, margin=1) | `asr/inference.py : compute_safe_max_new_tokens` |
| **Thời lượng xử lý tối đa/chunk** | 30.0 giây (480,000 mẫu tại 16kHz) | `configs/app.yaml`, `asr/inference.py` |
| **Cổng thẩm định đầu vào** | 7 tiêu chí tự động | `asr/audio.py : validate_audio` |
| **Số phép biến đổi sóng âm** | 7 phép biến đổi 1D miền thời gian | `augmentation/transforms.py` |
| **Kiểm soát chất lượng âm thanh** | 10 bài kiểm tra vật lý & mã băm SHA-256 | `augmentation/qc.py : run_10_point_qc` |
| **Độ trôi nhận dạng** | Drift WER và Drift CER tính bằng khoảng cách Levenshtein | `evaluation/metrics.py` |
| **Bộ kiểm thử phần mềm** | 31 bài test tự động (100% Passed) | Thư mục `tests/` |
| **Thiết bị đo lường thực tế** | CPU x86_64, PyTorch CPU float32 | `run_real_voice_test.py` output log |
| **Hệ số RTF đo trên CPU** | 0.221 (file 2.23s nhận dạng trong 0.49s) | `run_real_voice_test.py` output log |
| **Trạng thái Full Training trong App** | Chưa thực hiện (`NOT IMPLEMENTED`) | Thư mục `checkpoints/` |
| **Trạng thái Dry-run Resume Test** | Đã thực hiện thành công (`VERIFIED` qua test_resume_smoke.pt) | `scripts/train.py --dry_run` |

---

# PHẦN 25 — LỜI DẶN DÒ: NHỮNG ĐIỀU NÊN NÓI VÀ TUYỆT ĐỐI TRÁNH NÓI

### 25.1 Những điều ĐÃ ĐƯỢC KIỂM CHỨNG (Verified Facts - Tự tin nói):
1. Nhóm đã tích hợp thành công mô hình `PhoWhisper` chạy ổn định trên CPU/GPU qua thư viện Transformers.
2. Nhóm đã phát hiện ra lỗi tràn kích thước decoder $448$ của Whisper và đã tự viết hàm kẹp an toàn động giải quyết triệt để lỗi này.
3. Nhóm đã xây dựng cổng thẩm định âm thanh 7 tiêu chí và hàng rào 10 bài kiểm tra chất lượng âm thanh kèm mã băm SHA-256.
4. Nhóm đã xây dựng pipeline biến đổi sóng âm vật lý 1D với 7 phép biến đổi thời gian thực, có khả năng lặp lại theo seed.
5. Hệ thống đã được kiểm chứng qua 31 bài test tự động và đã chạy nhận dạng thành công trên giọng đọc thật tiếng Việt với RTF 0.221.

### 25.2 Những điều TUYỆT ĐỐI KHÔNG ĐƯỢC NÓI (Claims You MUST NOT Say):
1. **KHÔNG NÓI:** *"Nhóm em tự huấn luyện/tự train ra model PhoWhisper này."* $\to$ Giảng viên sẽ hỏi ngay train bao nhiêu triệu ảnh/tiếng, ở cluster nào, mất bao nhiêu tiền điện, và bạn sẽ trượt ngay lập tức vì gian dối.
2. **KHÔNG NÓI:** *"Nhóm em đã fine-tune lại model trong app này trên tập VIVOS/ViMD."* $\to$ Trong thư mục app này chỉ có model pretrained từ Hugging Face và file test smoke dummy 20KB, không có model checkpoint fine-tuned.
3. **KHÔNG NÓI:** *"Hệ thống đã được đo đạc benchmark hoàn chỉnh trên GPU A100."* $\to$ Trong log thực tế chỉ có CPU, chưa chạy GPU benchmark trong app.
4. **KHÔNG NÓI:** *"Hệ thống nhận dạng streaming thời gian thực từng từ theo micro."* $\to$ Hệ thống là dạng pseudo-streaming (nhận diện theo câu ngắn ngay sau khi ngắt thu âm).
5. **KHÔNG NÓI:** *"Nhóm đã huấn luyện trên cả 6 tập dữ liệu trong Registry."* $\to$ Các tập BUD500, VietMed, VietSuperSpeech, VLSP mới chỉ ở mức đăng ký siêu dữ liệu và thiết kế bộ lọc, chưa nạp vào huấn luyện trong app.

### 25.3 Cách dùng thuật ngữ chuẩn xác để tránh bị hiểu nhầm:
* Thay vì nói *"Nhóm em xây dựng model ASR"*, hãy nói: **"Nhóm em xây dựng hệ thống ứng dụng ASR dựa trên mô hình nền tảng PhoWhisper."**
* Thay vì nói *"Nhóm em train thử nghiệm thành công"*, hãy nói: **"Nhóm em đã chạy kiểm thử dry-run xác thực cơ chế checkpoint và khôi phục trạng thái ngẫu nhiên."**
* Thay vì nói *"Nhóm em đánh giá model đạt chuẩn"*, hãy nói: **"Nhóm em đã kiểm thử 31 ca kiểm thử phần mềm tự động và xác thực thành công trên mẫu giọng nói thực tế."**

---

```text
====================================================================================================
AUDIT STATUS BLOCK
- Code inspected:              100% full source in vietnamese_asr_app/ (app.py, asr/, augmentation/,
                               datasets/, evaluation/, export/, scripts/, tests/, configs/)
- Documentation inspected:     README.md, run_metadata.json, historical audit logs
- Tests inspected:             31 automated pytest cases (All 31 PASSED in 12.65s)
- Model loading verified:      VERIFIED (vinai/PhoWhisper-small/tiny loaded via ASRModelManager)
- Inference verified:          VERIFIED (Real voice "Xin chào tôi tên là Thanh", Latency 0.49s, RTF 0.221)
- Training verified:           DRY-RUN ONLY (scripts/train.py --dry_run verified via test_resume_smoke.pt)
- Fine-tuning verified:        NOT IMPLEMENTED IN THIS APP (Pretrained PhoWhisper used directly)
- End-to-end verified:         VERIFIED (Full 6-step Studio workflow, 10-check QC, Zip packaging)
====================================================================================================
```
