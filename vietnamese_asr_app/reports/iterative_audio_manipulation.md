# Iterative Audio Manipulation Specification & Research Report

## 1. Objective
The Iterative Audio Manipulation pipeline provides controlled, cumulative waveform-space perturbation of original recorded speech:
$$x_0 \xrightarrow{T_1} x_1 \xrightarrow{T_2} x_2 \xrightarrow{T_3} \dots \xrightarrow{T_K} x_K$$
where each intermediate state $x_i = T_i(x_{i-1})$ is derived directly from its predecessor.

The scientific objectives are:
- To explore PhoWhisper ASR robustness under compound physical acoustic distortions.
- To study transcription stability and cumulative linguistic drift under sequential perturbations.
- To discover "best-preserving" manipulated variants through candidate search.

---

## 2. Fundamental Difference from Synthetic Audio
| Dimension | Iterative Audio Manipulation | Synthetic Audio (TTS) |
| :--- | :--- | :--- |
| **Source of Truth** | Real original recorded speech waveform ($x_0$) | Text transcript string |
| **Generation Mechanism** | Time-domain 1D DSP transforms on audio samples | Neural text-to-speech synthesis (Gwen-TTS 0.6B) |
| **Speaker Identity** | 100% preserved from original speaker | Synthesized from reference speaker profile (`spk_synth_01`..`09`) |
| **Speech Content** | Exactly identical to original recording | Rendered from phonetic text tokens |
| **Audio Lineage** | Direct cumulative parent-child waveform chain | Text prompt $\to$ neural spectrogram vocoder |

> [!IMPORTANT]
> **Strict Non-Synthetic Policy:** Iterative audio manipulation uses **zero TTS**, **zero neural vocoding**, and **zero voice cloning**.

---

## 3. Difference from Independent Augmentation
In the existing independent augmentation mode:
$$x_0 \to T_1(x_0), \quad x_0 \to T_2(x_0), \quad x_0 \to T_3(x_0)$$
each augmented sample is an isolated sibling transformed directly from the source $x_0$.

In **Iterative Audio Manipulation**, each step accumulates upon previous transformations:
$$x_0 \xrightarrow{T_1} x_1 \xrightarrow{T_2} x_2 \dots \xrightarrow{T_K} x_K$$
Every state $x_i$ ($i \ge 1$) has exactly one parent $x_{i-1}$, and all intermediate states $x_1, \dots, x_K$ are saved, evaluated, and traceable.

---

## 4. Definition of $K$
$K$ represents the **number of cumulative transformation iterations**:
- $K = 1$: Produces $x_1 = T_1(x_0)$ (1 derived audio).
- $K = 3$: Produces $x_1, x_2, x_3$ where $x_1 = T_1(x_0), x_2 = T_2(x_1), x_3 = T_3(x_2)$ (3 derived audios).
- $K = 5$: Produces $x_1, x_2, x_3, x_4, x_5$ (5 derived audios).

$K$ is **never** interpreted as TTS sample count, number of independent copies, or speaker count.

---

## 5. Waveform-Space Transformation Operators
All operations strictly reuse existing tested 1D waveform transforms from `augmentation/transforms.py`:
1. **Gain (`apply_gain`):** Linear amplitude scaling with safe headroom limiting to prevent clipping.
2. **Additive Noise (`apply_additive_noise`):** Calibrated Gaussian noise added at a specified SNR in decibels.
3. **Time Shift (`apply_time_shift`):** Circular or zero-padded temporal shift in seconds.
4. **Time Stretch (`apply_time_stretch`):** Phase-vocoder tempo scaling preserving pitch.
5. **Pitch Shift (`apply_pitch_shift`):** Resampling pitch alteration by fractional semitones.
6. **Synthetic Reverb (`apply_synthetic_reverb`):** Room impulse response convolution modeling acoustic reverberation.
7. **Bandpass Filtering (`apply_frequency_filter`):** Butterworth IIR filter modeling telephone or acoustic channels.

---

## 6. Conservative Parameter Bounds (Cumulative Drift Control)
To prevent runaway distortion when applying compound operators across $K$ steps, individual iterations operate under strictly bounded conservative ranges:

| Operator | Parameter | Conservative Range per Step |
| :--- | :--- | :--- |
| `gain` | `gain_db` | $[-2.5, +2.5]$ dB |
| `additive_noise` | `snr_db` | $[22.0, 35.0]$ dB |
| `time_shift` | `shift_sec` | $[-0.05, +0.05]$ s |
| `time_stretch` | `rate` | $[0.96, 1.04] \times$ |
| `pitch_shift` | `n_steps` | $[-0.8, +0.8]$ semitones |
| `synthetic_reverb` | `decay_sec`, `wet_mix` | Decay: $[0.08, 0.20]$ s, Wet: $[0.08, 0.20]$ |
| `bandpass_filter` | `low_cut`, `high_cut` | Low: $[180, 320]$ Hz, High: $[3300, 3800]$ Hz |

---

## 7. 10-Point Audio Quality Control (QC) Policy
After **every** transformation step $T_i$, the candidate waveform $x_i$ is evaluated by the full 10-point QC suite (`run_10_point_qc`):
1. Decodable non-empty float32 array
2. Finite real values only
3. Zero NaNs
4. Zero Infs
5. Peak amplitude $\le 1.0$
6. Clipping sample ratio $< 0.1\%$
7. Signal RMS level $\ge 1 \times 10^{-5}$
8. Duration within bounds ($0.2$–$30.0$ s)
9. Format conformity (16 kHz mono)
10. Deterministic SHA-256 byte stream hash

### Failure Isolation Policy:
If a candidate $x_i$ fails any QC check:
- It is immediately rejected and marked `FAILED`.
- The failure reasons are logged into metadata.
- The pipeline **does not** advance the chain from the invalid audio.
- The previous valid state $x_{i-1}$ is preserved.

---

## 8. Deterministic Seed & Reproducibility
For an experiment with master seed $S$:
- Step $i$ derives sub-seed $S_i = S + (i \times 1000)$.
- Candidate $j$ in search mode derives $S_{i, j} = S_i + (j \times 37)$.
Given the same $(x_0, K, S, \text{config})$, the pipeline produces bitwise-identical audio arrays across runs.

---

## 9. Search Strategies

### Mode 1: Random Iterative Manipulation
At each iteration $i$, a single operator and bounded parameters are sampled deterministically:
$$x_i = T_i(x_{i-1})$$

### Mode 2: Best-Preserving Iterative Search
At each iteration $i$, $M$ distinct candidate transformations are generated from $x_{i-1}$:
$$c_1 = T_{i,1}(x_{i-1}), \quad c_2 = T_{i,2}(x_{i-1}), \quad \dots, \quad c_M = T_{i,M}(x_{i-1})$$
All candidates passing 10-point QC are transcribed through PhoWhisper and scored. The highest-scoring candidate is chosen:
$$x_i = \arg\max_{j} \text{Score}(c_j)$$

---

## 10. Scoring Methodology & Information Gating

### Case A: Ground-Truth Reference Transcript Available
When a reference transcript is supplied:
- Evaluates Levenshtein Word Error Rate (WER) and Character Error Rate (CER).
- Objective: Lower error is better.
$$\text{Score} = -(10 \cdot \text{WER} + \text{CER})$$
- Tie-breaking: Higher signal RMS / lower distortion.
- Label: `ACCURACY_WER_CER`.

### Case B: No Reference Transcript Available
When no reference transcript is supplied:
- **Accuracy metrics are strictly withheld.** No synthetic WER/CER is fabricated.
- Measures transcription consistency against the parent state $x_{i-1}$:
$$\text{Score} = (100 - \text{Drift\_WER}) - 0.1 \cdot \text{Drift\_CER}$$
- Explicit label: `ASR_CONSISTENCY_QUALITY_PROXY`.

---

## 11. Complete Parent-Child Lineage
Every generated intermediate artifact records:
- `original_audio_id`: Identifier of $x_0$.
- `iteration`: Index $i$.
- `parent_iteration`: Index of immediate parent ($i-1$).
- `manipulation_name`: Applied operator.
- `parameters`: Exact operator arguments.
- `random_seed`: Seed used for sampling.
- `duration_sec`, `rms`, `peak_abs`: Physical signal properties.
- `qc_passed`, `qc_status`, `failed_checks`: 10-point QC results.
- `asr_raw_transcript`: Output from PhoWhisper.
- `wer`, `cer` (if reference) or `consistency_drift_wer` (if no reference).
- `score`, `score_type`: Formal scoring breakdown.
- `audio_sha256`, `filename`: Storage metadata.

---

## 12. Multi-Format Output Artifacts
Every run exports:
- `wavs/iteration_000.wav` ($x_0$ original source copy)
- `wavs/iteration_001.wav` ($x_1$) ... `wavs/iteration_K.wav` ($x_K$)
- `iterative_manipulation_metadata.json`: Machine-readable lineage tree.
- `iterative_manipulation_manifest.csv`: Tabular spreadsheet for analysis.
- `.zip` Archive: Downloadable bundle packaging all WAVs, JSON, and CSV.

---

## 13. Scientific Limitations & Ethical Boundaries
- **No Accuracy Improvement Guarantee:** Iterative manipulation modifies the acoustic waveform. Without a reference transcript, higher consistency does not guarantee ground-truth correctness.
- **Speech Degradation Floor:** As $K \to \infty$, compounding perturbations will eventually degrade acoustic phonetic intelligibility. Conservative bounds mitigate this up to $K=10$.
- **Original Audio Integrity:** The input audio $x_0$ is never modified in-place; it is preserved as an immutable root artifact.
