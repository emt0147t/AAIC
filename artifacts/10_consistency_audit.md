# 10 — Final Experimental Consistency Audit (Gate 7)

> **Purpose**: Cross-artifact audit identifying and resolving discrepancies, evolving assumptions, and conflicting records across Gates 0 through 6.
> **Rule**: Do not silently correct contradictions; explicitly log and resolve every issue.

---

## 1. Audit Summary Matrix

| Audit Dimension | Status | Notes |
|:---|:---:|:---|
| **Training Data Selection** | CONSISTENT | VIVOS train + ViMD train (Config B: ~95h gold) across Gate 3–6 |
| **Validation Data Selection** | CONSISTENT | ViMD valid (10.26h, 1,900 utt) across Gate 4–6 |
| **Primary Test Candidate** | CONSISTENT | ViMD test (10.87h, 2,026 utt), NEF across all models |
| **Secondary Reference Test** | CONSISTENT | VIVOS test (0.75h, 760 utt), CONFIRMED_CONTAMINATED for PhoWhisper |
| **Primary Model Candidate** | CONSISTENT | `vinai/PhoWhisper-tiny` (37.8M params) |
| **Fallback Model Candidate** | CONSISTENT | `vinai/PhoWhisper-base` (74M params) |
| **ViMD Sample Rate** | RESOLVED DISCREPANCY | Assumed 16 kHz in G1–G4; verified 44.1 kHz in G5–G6 |
| **FOSD License & Format** | RESOLVED DISCREPANCY | Mistakenly cited as CC BY 4.0 & WAV; verified FPT License & MP3 |
| **VietSuperSpeech Status** | CONSISTENT DISCREPANCY | Version unresolved (32k vs 52k vs 67k); excluded from frozen protocol |
| **Augmentation Methodology** | CONSISTENT | Labeled "reconstruction based on report methodology" across G5–G7 |
| **Pilot Metrics vs Benchmark** | CONSISTENT | Strictly labeled "ENGINEERING PILOT — NOT FOR FINAL BENCHMARK" |

---

## 2. Detailed Discrepancies and Contradictions

### Issue 1: ViMD Native Audio Sample Rate (16 kHz vs 44.1 kHz)
- **ISSUE**: Initial dataset profiles assumed ViMD was 16,000 Hz (standard speech recognition benchmark convention), whereas actual audio decoding proved it is 44,100 Hz.
- **SOURCE A**: `03_data_quality.md` (Table 1 lists format as `WAV ?k`); `07_clean_asr_protocol.md` initial draft (`Sample rate: 16,000 Hz; ViMD expected 16kHz based on standard practice`).
- **SOURCE B**: `08_augmentation_analysis.md` and `09_pilot_report.md` (`Decoded audio verified at 44,100 Hz; librosa.resample converts 44.1 kHz -> 16.0 kHz`).
- **REQUIRED ACTION**: Explicitly freeze the requirement that all ViMD audio must undergo software resampling ($44.1\text{ kHz} \rightarrow 16.0\text{ kHz}$) before feature extraction in both training and evaluation pipelines.

### Issue 2: FOSD License Classification (CC BY 4.0 vs FPT Public License)
- **ISSUE**: Earlier gate drafts inherited a "CC BY 4.0" label from secondary references, but original Mendeley record inspection showed a custom license.
- **SOURCE A**: `03_data_quality.md` early draft (Table row: `License: CC BY 4.0 (Mendeley)`).
- **SOURCE B**: Mendeley Data record `10.17632/k9sxg2twv4.4` and Gate 4 update in `04_data_selection_analysis.md` (`FPT Public License — custom permissive license with patent/trademark exclusion`).
- **REQUIRED ACTION**: Maintain `FPT Public License` across all final documentation; reject the CC BY 4.0 label.

### Issue 3: FOSD Audio Encoding (WAV vs MP3)
- **ISSUE**: Gate 2 quality audit listed FOSD as WAV PCM 16-bit, but direct Mendeley repository inspection revealed recordings are compressed MP3 files.
- **SOURCE A**: `03_data_quality.md` (Table row: `Audio format: WAV PCM 16-bit mono, 16kHz`).
- **SOURCE B**: Mendeley file listing and Gate 4 verification in `04_data_selection_analysis.md` (`Audio format: MP3; requires decoding and conversion to WAV for ASR ingestion`).
- **REQUIRED ACTION**: Record MP3 format and transcoding requirement as an operational barrier; maintain FOSD exclusion from the frozen provisional mixture.

### Issue 4: VIVOS Hugging Face Access & Mirror Redundancy
- **ISSUE**: Canonical repository `AILAB-VNUHCM/vivos` on Hugging Face returns HTTP 401 on legacy loading scripts and packages audio as a monolithic 1.47 GB archive.
- **SOURCE A**: `03_data_quality.md` (`Access: Public HuggingFace, not gated`).
- **SOURCE B**: Gate 4 API test (`401 Client Error on script download`) and Gate 6 pilot (`thanhduycao/vivos_ng_only` parquet mirror used for lightweight 10.8 MB streaming).
- **REQUIRED ACTION**: Clarify canonical access: authoritative archival source is Zenodo record `7068130` (1.47 GB tar.gz); for lightweight pilot streaming, community parquet mirrors (`thanhduycao/vivos_ng_only`) provide identical prompts and audio without 401 failures.

### Issue 5: VietSuperSpeech Release Discrepancy (32,267 vs 52,023 vs 67,405)
- **ISSUE**: Three contradictory numbers exist for utterance count and total hours across documentation and manifests.
- **SOURCE A**: Hugging Face README (`32,267 samples / 103.18 hours / 3 sources`).
- **SOURCE B**: Published arXiv Paper (`52,023 samples / 267.39 hours / 4 sources`); Current HF commit `cbf624a` (`67,405 manifest entries / ~300+ hours`).
- **REQUIRED ACTION**: Maintain classification `VIETSUPERSPEECH = VERSION_UNRESOLVED`. Strictly exclude VietSuperSpeech from any frozen ASR test set or gold training mixture. Mark Config C as `NOT READY TO FREEZE`.

### Issue 6: VIVOS Test Contamination vs Evaluation Role
- **ISSUE**: Risk of treating VIVOS test as a general generalization benchmark despite known training contamination.
- **SOURCE A**: Historical reports and benchmarks treating VIVOS test as standard Vietnamese ASR test.
- **SOURCE B**: `06_model_contamination_audit.md` and PhoWhisper paper (arXiv:2406.02555, Table 1: VIVOS test explicitly listed in PhoWhisper training data pool).
- **REQUIRED ACTION**: Enforce strict distinction: VIVOS test is **CONFIRMED_CONTAMINATED** for PhoWhisper and Zipformer. It is labeled **SECONDARY REFERENCE ONLY** to measure in-domain ceiling, never generalization.

### Issue 7: ViMD Dataset License & Model Redistribution
- **ISSUE**: ViMD license is `CC BY-NC-ND 4.0` (NonCommercial, NoDerivatives), which carries restrictions on derivative works.
- **SOURCE A**: Hugging Face canonical repo `nguyendv02/ViMD_Dataset` metadata (`license: cc-by-nc-nd-4.0`).
- **SOURCE B**: `07_clean_asr_protocol.md` License Constraints section.
- **REQUIRED ACTION**: Acknowledge that while training for internal research/evaluation is permissible, publicly distributing model weights fine-tuned on ViMD may be restricted by the NoDerivatives clause. This keeps the protocol classified as **PROVISIONAL** until legal clarification is obtained.

### Issue 8: Pilot Metrics Interpretation
- **ISSUE**: Risk of citing Gate 6 pilot WER (~64%) as a model capability benchmark.
- **SOURCE A**: Raw evaluation output in `five_sample_predictions.csv` (mean WER 0.6400, CER 0.4971).
- **SOURCE B**: Pilot report `09_pilot_report.md` section 1.
- **REQUIRED ACTION**: Permanently stamp all pilot numbers with: `ENGINEERING PILOT — NOT FOR FINAL BENCHMARK`. Explicitly document that 1 epoch on 100 samples verifies code execution paths only, not acoustic convergence.
