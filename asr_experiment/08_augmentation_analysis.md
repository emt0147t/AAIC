# 08 — Augmentation Analysis (Gate 5)

> LABEL: **augmentation reconstruction based on the weekly report methodology**
>
> No original MixMatch/ReMixMatch implementation was found in the workspace.
> All augmentations below are reconstructed from the methodology described in the weekly report.

> [!IMPORTANT]
> **ViMD sample rate = 44,100 Hz** (verified from decoded audio). This is NOT 16kHz.
> ASR preprocessing must resample ViMD audio from 44,100 Hz → 16,000 Hz before feature extraction.
> This resolves a Gate 4 unresolved item.

---

## Terminology Separation

| Concept | Domain | Description |
|---------|--------|-------------|
| **Waveform augmentation** | Signal (time domain) | Directly modifies the raw audio waveform samples |
| **Spectrogram augmentation** | Feature space | Modifies the time-frequency representation after STFT/mel transform |
| **MixUp** | Training strategy | Convex combination of two training examples and their labels |
| **Pseudo-label generation** | Semi-supervised | Using model predictions on unlabeled data as surrogate labels |
| **Sharpening** | Label refinement | Lowering temperature of pseudo-label distribution to reduce entropy |
| **Distribution alignment** | ReMixMatch | Rescaling pseudo-label distribution to match labeled data marginals |
| **Augmentation anchoring** | ReMixMatch | Using weak augmentation output as anchor for pseudo-label, strong augmentation for training input |

---

## Augmentations Applied

### 1. Amplitude Perturbation — WAVEFORM-SPACE

- **What it changes**: Scales the entire waveform by a random gain factor (dB)
- **What remains invariant**: Duration, frequency content (spectral shape), temporal structure, pitch
- **Why useful for SSL**: Forces the model to be invariant to recording volume differences, which vary across microphones, distances, and recording conditions
- **Representation**: Waveform → Waveform
- **RECONSTRUCTION PARAMETERS**: Uniform random gain in [-6dB, +6dB]
- **Library**: numpy
- **Limitations**: Does not simulate frequency-dependent gain (e.g., room acoustics); a very simple augmentation

### 2. Time Shift — WAVEFORM-SPACE

- **What it changes**: Circularly shifts the waveform by a random number of samples (up to ±100ms)
- **What remains invariant**: Duration, frequency content, amplitude, spectral shape
- **Why useful for SSL**: Forces the model to be invariant to exact alignment/onset of speech; simulates slight timing differences in segmentation
- **Representation**: Waveform → Waveform
- **RECONSTRUCTION PARAMETERS**: max_shift_ms=100, circular shift (np.roll)
- **Library**: numpy
- **Limitations**: Circular shift wraps end→beginning, which may create a small discontinuity; in practice this is negligible for short shifts relative to utterance length

### 3. Noise Injection — WAVEFORM-SPACE

- **What it changes**: Adds additive white Gaussian noise at a controlled signal-to-noise ratio
- **What remains invariant**: Duration, temporal structure (approximately), spectral shape of speech (speech remains dominant)
- **Why useful for SSL**: Forces the model to be robust to background noise, which is ubiquitous in real-world speech; noise robustness is critical for deployment
- **Representation**: Waveform → Waveform
- **RECONSTRUCTION PARAMETERS**: SNR=20dB (moderate, speech clearly audible above noise)
- **Library**: numpy (Gaussian noise generation)
- **Limitations**: White Gaussian noise is not realistic background noise (which is typically colored/structured); does not simulate babble, music, or environmental noise

### 4. Time Masking — FEATURE-SPACE

- **What it changes**: Zeros out contiguous blocks of time frames in the log-mel spectrogram (SpecAugment)
- **What remains invariant**: Frequency content in non-masked regions; overall spectral shape
- **Why useful for SSL**: Forces the model to predict masked temporal regions from surrounding context; prevents over-reliance on any single time segment; inspired by BERT-style masking
- **Representation**: Log-mel spectrogram → Masked log-mel spectrogram
- **RECONSTRUCTION PARAMETERS**: max_T=20 frames, num_masks=2
- **Library**: numpy (SpecAugment-style)
- **NO WAV OUTPUT**: This is a feature-space transformation. The spectrogram cannot be meaningfully inverted back to a waveform because phase information is discarded. Saving a "masked WAV" would be scientifically misleading.
- **Limitations**: The mask boundaries are sharp rectangles; does not simulate realistic temporal dropouts

### 5. Frequency Masking — FEATURE-SPACE

- **What it changes**: Zeros out contiguous blocks of mel frequency bins in the log-mel spectrogram (SpecAugment)
- **What remains invariant**: Temporal structure; frequency content in non-masked bands
- **Why useful for SSL**: Forces the model to recognize speech from partial spectral information; prevents over-reliance on specific frequency bands (e.g., formant regions)
- **Representation**: Log-mel spectrogram → Masked log-mel spectrogram
- **RECONSTRUCTION PARAMETERS**: max_F=15 mel bins, num_masks=2
- **Library**: numpy (SpecAugment-style)
- **NO WAV OUTPUT**: Feature-space transformation only.
- **Limitations**: Mel-scale masking affects perceptually-weighted bands, not raw Hz; mask width in mel bins is not directly interpretable in Hz

---

## MixMatch / ReMixMatch Analysis

### Implementation Status: NO ORIGINAL IMPLEMENTATION FOUND

The workspace contains no Python/notebook implementation of MixMatch or ReMixMatch. The weekly report (`demo11.pdf`) describes these algorithms conceptually but no executable code was located.

### MixMatch (Berthelot et al., 2019) — Report Description

The report describes MixMatch as combining:
1. **K augmented views** of each unlabeled sample (K=2 typically)
2. **Pseudo-labeling**: Model predicts on each augmented view
3. **Averaging**: Average predictions across K views to reduce noise
4. **Sharpening**: Apply temperature scaling (T < 1) to averaged prediction to reduce entropy
5. **MixUp**: Create convex combinations of labeled/unlabeled examples: `x' = λ·x_i + (1-λ)·x_j`, `y' = λ·y_i + (1-λ)·y_j` where λ ~ Beta(α, α)

The augmented views in the speech domain would use waveform augmentations (amplitude perturbation, time shift, noise) and/or spectrogram augmentations (time/frequency masking).

### ReMixMatch (Berthelot et al., 2020) — Report Description

The report describes ReMixMatch as extending MixMatch with:
1. **Distribution alignment**: Rescale the pseudo-label distribution to match the marginal distribution of labeled data classes
2. **Augmentation anchoring**: Use a weakly-augmented view to generate pseudo-labels, but train on a strongly-augmented view
3. **Additional strongly-augmented view**: One extra strongly-augmented unlabeled sample added to loss

These are **training strategy concepts**, not audio augmentations per se. They operate on the output of augmentation pipelines, not on audio directly.

### What Can Be Reconstructed

Only the **individual augmentation primitives** (amplitude perturbation, time shift, noise injection, time masking, frequency masking) can be reconstructed and visualized. The full MixMatch/ReMixMatch training loop requires:
- A model
- Labeled and unlabeled data splits
- A training loop with combined loss
- Hyperparameters (K, T, α, λ)

None of these are reconstructed in this gate. The augmentation primitives shown here are the **building blocks** that would feed into such a training loop.

---

## Parameter Distinction

| Parameter | Type | Value | Source |
|-----------|------|-------|--------|
| Amplitude gain range | RECONSTRUCTION | ±6 dB | Conservative default |
| Time shift max | RECONSTRUCTION | 100 ms | Conservative default |
| Noise SNR | RECONSTRUCTION | 20 dB | Conservative (clearly audible speech) |
| Time mask max width | RECONSTRUCTION | 20 frames | SpecAugment paper default |
| Time mask count | RECONSTRUCTION | 2 | SpecAugment paper default |
| Freq mask max width | RECONSTRUCTION | 15 mel bins | SpecAugment paper default |
| Freq mask count | RECONSTRUCTION | 2 | SpecAugment paper default |
| n_fft | RECONSTRUCTION | 512 | Standard for 16kHz |
| hop_length | RECONSTRUCTION | 160 | 10ms hop for 16kHz |
| n_mels | RECONSTRUCTION | 80 | Standard for ASR |
| Random seed | RECONSTRUCTION | 42 | Fixed for reproducibility |

> These are NOT confirmed as the parameters used in the historical experiment.
> They are conservative defaults chosen to make transformations visible while remaining acoustically plausible.

---

## Output Structure

```
artifacts/augmentation_demo/
├── sample_000/
│   ├── original.wav
│   ├── amplitude.wav              # waveform-space
│   ├── time_shift.wav             # waveform-space
│   ├── noise.wav                  # waveform-space
│   ├── original_spectrogram.png
│   ├── time_mask_spectrogram.png  # feature-space (NO WAV)
│   ├── frequency_mask_spectrogram.png  # feature-space (NO WAV)
│   ├── comparison.png
│   └── metadata.json
├── sample_001/
│   └── ... (same structure)
├── sample_002/
│   └── ... (same structure)
├── summary_comparison.png
└── summary.csv
```

---

## Report-Ready Summary Table

The following table can be directly copied into the weekly report:

| Sample ID | Augmentation | Representation | Parameters | Duration (s) | Sample Rate | Output File | Nature |
|:---|:---|:---|:---|:---:|:---:|:---|:---:|
| `sample_000` | Original Speech | Waveform | N/A (Baseline) | 9.239 | 44,100 Hz | `original.wav` | Original |
| `sample_000` | Amplitude Perturbation | Waveform-space | Gain = -1.506 dB | 9.239 | 44,100 Hz | `amplitude.wav` | RECONSTRUCTION |
| `sample_000` | Time Shift | Waveform-space | Shift = +64.85 ms | 9.239 | 44,100 Hz | `time_shift.wav` | RECONSTRUCTION |
| `sample_000` | Noise Injection | Waveform-space | White noise, SNR = 20.0 dB | 9.239 | 44,100 Hz | `noise.wav` | RECONSTRUCTION |
| `sample_000` | Time Masking | Feature-space | $T=20$ frames, $n=2$ masks | 9.239 | 44,100 Hz | `time_mask_spectrogram.png` | RECONSTRUCTION |
| `sample_000` | Frequency Masking | Feature-space | $F=15$ mel bins, $n=2$ masks | 9.239 | 44,100 Hz | `frequency_mask_spectrogram.png` | RECONSTRUCTION |
| `sample_001` | Original Speech | Waveform | N/A (Baseline) | 7.358 | 44,100 Hz | `original.wav` | Original |
| `sample_001` | Amplitude Perturbation | Waveform-space | Gain = -1.506 dB | 7.358 | 44,100 Hz | `amplitude.wav` | RECONSTRUCTION |
| `sample_001` | Time Shift | Waveform-space | Shift = +64.85 ms | 7.358 | 44,100 Hz | `time_shift.wav` | RECONSTRUCTION |
| `sample_001` | Noise Injection | Waveform-space | White noise, SNR = 20.0 dB | 7.358 | 44,100 Hz | `noise.wav` | RECONSTRUCTION |
| `sample_001` | Time Masking | Feature-space | $T=20$ frames, $n=2$ masks | 7.358 | 44,100 Hz | `time_mask_spectrogram.png` | RECONSTRUCTION |
| `sample_001` | Frequency Masking | Feature-space | $F=15$ mel bins, $n=2$ masks | 7.358 | 44,100 Hz | `frequency_mask_spectrogram.png` | RECONSTRUCTION |
| `sample_002` | Original Speech | Waveform | N/A (Baseline) | 3.651 | 44,100 Hz | `original.wav` | Original |
| `sample_002` | Amplitude Perturbation | Waveform-space | Gain = -1.506 dB | 3.651 | 44,100 Hz | `amplitude.wav` | RECONSTRUCTION |
| `sample_002` | Time Shift | Waveform-space | Shift = +64.85 ms | 3.651 | 44,100 Hz | `time_shift.wav` | RECONSTRUCTION |
| `sample_002` | Noise Injection | Waveform-space | White noise, SNR = 20.0 dB | 3.651 | 44,100 Hz | `noise.wav` | RECONSTRUCTION |
| `sample_002` | Time Masking | Feature-space | $T=20$ frames, $n=2$ masks | 3.651 | 44,100 Hz | `time_mask_spectrogram.png` | RECONSTRUCTION |
| `sample_002` | Frequency Masking | Feature-space | $F=15$ mel bins, $n=2$ masks | 3.651 | 44,100 Hz | `frequency_mask_spectrogram.png` | RECONSTRUCTION |

---

## Audio Quality Verification Summary

All 9 generated WAV files across the 3 samples were verified against the quality criteria:
- **File Existence & Decodability**: 100% of files exist on disk and decode successfully via `scipy.io.wavfile` / `soundfile`.
- **Sample Rate & Channels**: Exactly 44,100 Hz, 1 channel (mono PCM 16-bit) matching the original source.
- **Duration Invariance**: Duration before and after waveform augmentation is strictly identical (sample_000: 9.239s; sample_001: 7.358s; sample_002: 3.651s).
- **Numerical Integrity**: Zero NaN values, zero Inf values.
- **Clipping Prevention**: Peak values strictly within `[-1.0, 1.0]`. No accidental clipping detected (`clipped = False`).
- **Signal Presence**: Energy verified via RMS calculation (RMS range: 0.057 - 0.070). No silent or degenerate outputs (`silent = False`).

