# Vietnamese Speech AI Studio — Research & Operational Workflow

## 1. Overview
The Vietnamese Speech AI Studio bridges rigorous scientific speech research with an interactive engineering interface. It exposes the complete research pipeline:

```
USER AUDIO (Microphone / Upload)
    ↓
7-POINT VALIDATION GATE (Finite, Mono, Dur 0.2–30s, RMS >= 1e-5, Peak <= 1.2)
    ↓
16 kHz MONO PREPROCESSING
    ↓
BASE PhoWhisper INFERENCE (Greedy / Beam Search, Token Clamping)
    ↓
OPTIONAL 1D WAVEFORM AUGMENTATION (7 Transforms, K=1–20, 10-Point QC)
    ↓
OPTIONAL SYNTHETIC SPEECH GENERATION (Gwen-TTS 0.6B, Approved Voices, Lineage Provenance)
    ↓
OPTIONAL PEFT LoRA ADAPTER (Dynamic Stats, Checkpoint Loader, Read-Only Frozen Full E1)
    ↓
EVALUATION & BENCHMARKING (Ground-Truth Gated Accuracy vs Prediction Drift)
    ↓
NEUTRAL COMPARATIVE METRICS (Absolute & Relative Deltas, No Promotional Claims)
    ↓
NUMERICAL REPORTING & MULTI-FORMAT EXPORT (Section 10 JSON, Section 12 MD, CSV, ZIP)
```

---

## 2. Detailed Pipeline Stages

### Stage 1: Audio Ingestion & 7-Point Audio Validation
1. **Signal Existence:** Confirms non-empty audio array.
2. **Finite Real Numbers:** Verifies `np.all(np.isfinite(audio))`.
3. **Mono Channel:** Ensures 1D single-channel representation.
4. **Lower Duration Bound:** Enforces duration $\ge 0.20$ seconds.
5. **Upper Duration Bound:** Enforces duration $\le 30.00$ seconds (or invokes sequential chunking).
6. **RMS Signal Level:** Enforces $\text{RMS} \ge 1 \times 10^{-5}$ (rejects digital silence).
7. **Dynamic Peak Range:** Checks $|\text{peak}| \le 1.20$ (flags catastrophic clipping).

### Stage 2: PhoWhisper Inference & Decoding Safety
- Employs `vinai/PhoWhisper-small` (or user-selected tiny/base/medium).
- Enforces Whisper decoder architectural safety constraint:
  $$\text{prompt\_length} + \text{effective\_max\_new\_tokens} + 1 \le 448$$
- Calculates precise latency (seconds) and Real-Time Factor (RTF).
- Computes NFC unicode normalized transcript with Vietnamese punctuation stripped.

### Stage 3: Waveform Augmentation Studio
- **Transformations:** Gain ($\pm 12$ dB), Additive Calibrated Noise (SNR 10–35 dB), Time Shift ($\pm 0.5$ s), Time Stretch ($0.85$–$1.15 \times$), Pitch Shift ($\pm 2$ semitones), Synthetic Reverb ($0.1$–$0.5$ s decay), Bandpass Filter ($300$–$3400$ Hz).
- **10-Point QC Protocol:** Non-empty, finite, no NaNs, no Infs, range $\le 1.0$, clipping ratio $< 0.1\%$, RMS $> 1 \times 10^{-5}$, duration $0.2$–$30.0$ s, 16 kHz mono format, deterministic SHA-256 hash.

### Stage 4: Synthetic Audio Synthesis & Provenance Lineage
- **Backend:** `g-group-ai-lab/gwen-tts-0.6B` loaded on demand (lazy loading).
- **Voice Policy:** Strictly uses built-in reference voices (`spk_synth_01` to `spk_synth_09`). Zero real human speaker cloning.
- **Provenance Logging:** Captures `synthetic_id`, `source_text_id`, `source_dataset`, `source_split`, transcript hash, voice ID, timestamp, 10-point QC pass/fail status, and audio SHA-256 hash.

### Stage 5: LoRA Model Adaptation & Safety Gates
- **Parameter Accounting:** Dynamic calculation from instantiated models yields:
  - Total Parameters: 37,908,096
  - Trainable LoRA Parameters: 147,456
  - Frozen Base Parameters: 37,760,640
  - Trainable Ratio: 0.3890%
- **Execution Safety Gate:** For Full E1, verifies CUDA availability, PEFT 0.21.0, pinned model revision `cc51d32be916efebde04ff549854fa1741cb5c02`, and frozen test manifest CRLF hash `efe54856d12613f109cebc9482a6f224ca4e2370d0ff3f8a2fe4ea0a013b60ca`. Refuses to launch Full E1 on CPU workstation.

### Stage 6: Accuracy vs Prediction Drift Evaluation
- **Reference Provided:** Computes exact Levenshtein Word Error Rate (WER) and Character Error Rate (CER), reporting substitutions, deletions, insertions, and exact match status.
- **Reference Omitted:** Strictly withholds accuracy claims. Instead, evaluates consistency across variants or models and labels metrics as `PREDICTION DRIFT / CONSISTENCY`.

### Stage 7: Reporting & Export Packaging
- **Section 10 Machine-Readable JSON:** Standardized JSON structure recording model, adapter, dataset, metrics, training budget, and synthetic metadata.
- **Section 12 Human-Readable Markdown:** Distinguishes `MEASURED` (T4 CUDA preflight: 0.942s/step, 726 MB VRAM), `ESTIMATED` (2,358s duration), `PROJECTED`, and `N/A` (uncompleted metrics).
- **Multi-Format Bundles:** Exports CSV summaries, JSON objects, Markdown reports, and complete downloadable ZIP packages.
