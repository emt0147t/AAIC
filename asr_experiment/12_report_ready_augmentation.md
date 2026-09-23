# 12 — Report-Ready Audio Augmentation Technical Summary

> **METHODOLOGY STATEMENT**:  
> **The Gate 5 experiment was a reconstruction based on the weekly report methodology, not a reproduction of the historical MixMatch/ReMixMatch implementation.**  
> A complete search of the workspace revealed no historical implementation or code scripts. All transformations were reconstructed from theoretical principles described in the weekly report.

---

## 1. Domain and Concept Separation

Audio augmentation methods operate across distinct physical and mathematical spaces. Conflating time-domain signal manipulation with spectrogram masking or loss-level interpolation obscures acoustic invariance properties.

```
Audio Input (Waveform) 
       │
       ├─ [Waveform-Space Augmentation] ───> amplitude.wav, time_shift.wav, noise.wav
       │
       ▼ (STFT / Log-Mel Feature Extraction)
Spectrogram Representation 
       │
       ├─ [Feature-Space Augmentation]  ───> time_mask.png, freq_mask.png (NO WAV)
       │
       ▼ (Acoustic Model Encoder / Decoder)
Latent Embeddings & Logits
       │
       ├─ [Loss / Representation Level] ───> MixUp (Linear interpolation)
       │
       ▼ (Semi-Supervised Training Strategy)
Optimization Objective
       │
       └─ [SSL Training Strategy]       ───> Augmentation Anchoring, Sharpening
```

### Categorization Framework

| Category | Transformations | Mathematical Domain | Acoustic Invariance Enforced | Output Artifact |
|:---|:---|:---|:---|:---:|
| **WAVEFORM-SPACE** | Amplitude Perturbation, Time Shifting, Noise Injection | Discrete time-domain signal $x[n]$ | Volume variation, temporal alignment jitter, environmental SNR | 16-bit PCM WAV |
| **FEATURE-SPACE** | Time Masking, Frequency Masking (SpecAugment) | 2D time-frequency matrix $S[f, t]$ | Temporal dropouts, formant occlusion, spectral bias | PNG Spectrogram (Zero Fake WAVs) |
| **LOSS / REPRESENTATION** | MixUp | Latent feature vectors and target labels | Smooth decision boundaries between acoustic classes | Training Loss Term |
| **SSL STRATEGY** | Augmentation Anchoring, Distribution Alignment | Pseudo-label generation and assignment | Consistent prediction across weak and strong views | Training Loop |

---

## 2. Technical Breakdown of Reconstructed Primitives

### 1. Amplitude Perturbation (Waveform-Space)
- **Mathematical Operation**: $x_{\text{aug}}[n] = \alpha \cdot x[n]$, where $\alpha = 10^{\Delta \text{dB} / 20}$ with $\Delta \text{dB} \sim \mathcal{U}(-6.0, +6.0)$.
- **What Changes**: Signal root-mean-square (RMS) energy and peak amplitude.
- **What Remains Invariant**: Fundamental frequency ($F_0$), harmonic relationships, formants, speech rate, duration.
- **SSL Rationale**: Eliminates sensitivity to recording gain, microphone distance, and speaker projection volume.
- **Artifact**: `amplitude.wav` (PCM 16-bit mono, verified duration invariant).

### 2. Time Shifting (Waveform-Space)
- **Mathematical Operation**: Circular buffer rotation $x_{\text{aug}}[n] = x[(n + \delta) \pmod N]$, where $|\delta| \le \Delta t_{\max} \cdot f_s$ with $\Delta t_{\max} = 100\text{ ms}$.
- **What Changes**: Absolute temporal sample index of phonemic transitions.
- **What Remains Invariant**: Complete frequency spectrum, spectral tilt, amplitude envelope, duration.
- **SSL Rationale**: Prevents neural encoders from over-fitting to fixed temporal frame offsets or utterance onset positions.
- **Artifact**: `time_shift.wav` (PCM 16-bit mono).

### 3. Controlled Noise Injection (Waveform-Space)
- **Mathematical Operation**: $x_{\text{aug}}[n] = x[n] + \sigma \cdot \epsilon[n]$, where $\epsilon[n] \sim \mathcal{N}(0, 1)$ and $\sigma = \sqrt{P_{\text{signal}} / 10^{\text{SNR} / 10}}$ with $\text{SNR} = 20.0\text{ dB}$.
- **What Changes**: Signal-to-noise ratio; introduces broad-spectrum stochastic noise floor.
- **What Remains Invariant**: Phonetic structure, pitch contour, intelligible speech cues.
- **SSL Rationale**: Regularizes representations against stationary ambient background noise common in field recordings.
- **Artifact**: `noise.wav` (PCM 16-bit mono).

### 4. Time Masking (Feature-Space)
- **Mathematical Operation**: In log-mel spectrogram $S \in \mathbb{R}^{F \times T}$, zeroing columns $S[:, t_0 : t_0 + t] = S_{\min}$, where $t \le T_{\max} = 20$ frames ($n=2$ masks).
- **What Changes**: Erases temporal segments across all frequency bins.
- **What Remains Invariant**: All unmasked temporal frames; global frequency structure.
- **SSL Rationale**: Encourages bidirectional temporal context integration (BERT-like phonetic reconstruction).
- **Scientific Integrity**: **No fake WAV is exported**. Spectrogram inversion without phase is acoustically destructive; exported strictly as `time_mask_spectrogram.png`.

### 5. Frequency Masking (Feature-Space)
- **Mathematical Operation**: Zeroing rows $S[f_0 : f_0 + f, :] = S_{\min}$, where $f \le F_{\max} = 15$ mel channels ($n=2$ masks).
- **What Changes**: Suppresses specific mel-frequency sub-bands across the entire duration.
- **What Remains Invariant**: Full temporal envelope; unmasked frequency bands.
- **SSL Rationale**: Forces the acoustic model to utilize redundant phonetic information across multiple formants.
- **Scientific Integrity**: Exported strictly as `frequency_mask_spectrogram.png` (zero fake WAVs).

---

## 3. Semi-Supervised Learning Concepts (Loss & Strategy Level)

These components described in the weekly report are **algorithmic learning policies**, not audio file transforms:

1. **MixUp Interpolation**:
   - Computes convex combinations: $\tilde{x} = \lambda x_i + (1 - \lambda) x_j$ and $\tilde{y} = \lambda y_i + (1 - \lambda) y_j$ where $\lambda \sim \text{Beta}(\alpha, \alpha)$.
   - Operates in feature/representation space, regularizing classification boundaries.
2. **Augmentation Anchoring (ReMixMatch)**:
   - Uses a **weakly augmented view** (e.g. slight amplitude jitter) to produce a stable pseudo-label distribution.
   - Enforces consistency by training the model to predict that anchored target on multiple **strongly augmented views** (e.g. combined heavy SpecAugment + noise).
3. **Pseudo-Label Sharpening**:
   - Reduces entropy of averaged predictions via temperature scaling: $p_i = q_i^{1/T} / \sum_j q_j^{1/T}$ ($T < 1.0$), pushing predictions toward one-hot confidence.

---

## 4. Report-Ready Demonstration Results (3 Utterances)

| Sample ID | Augmentation Primitive | Category | Parameter Specification | Duration (s) | Sample Rate | Decodable? | Artifact Name |
|:---|:---|:---|:---|:---:|:---:|:---:|:---|
| `sample_000` | Baseline Speech | Waveform | Unprocessed source | 9.239 | 44,100 Hz | Yes | `original.wav` |
| `sample_000` | Amplitude Perturbation | Waveform-Space | Gain = -1.506 dB | 9.239 | 44,100 Hz | Yes | `amplitude.wav` |
| `sample_000` | Time Shift | Waveform-Space | Circular shift = +64.85 ms | 9.239 | 44,100 Hz | Yes | `time_shift.wav` |
| `sample_000` | Noise Injection | Waveform-Space | Gaussian noise, 20 dB SNR | 9.239 | 44,100 Hz | Yes | `noise.wav` |
| `sample_000` | Time Masking | Feature-Space | $T=20$ frames, $n=2$ masks | 9.239 | 44,100 Hz | N/A | `time_mask_spectrogram.png` |
| `sample_000` | Frequency Masking | Feature-Space | $F=15$ mel bins, $n=2$ masks | 9.239 | 44,100 Hz | N/A | `frequency_mask_spectrogram.png` |
| `sample_001` | Baseline Speech | Waveform | Unprocessed source | 7.358 | 44,100 Hz | Yes | `original.wav` |
| `sample_001` | Amplitude Perturbation | Waveform-Space | Gain = -1.506 dB | 7.358 | 44,100 Hz | Yes | `amplitude.wav` |
| `sample_001` | Time Shift | Waveform-Space | Circular shift = +64.85 ms | 7.358 | 44,100 Hz | Yes | `time_shift.wav` |
| `sample_001` | Noise Injection | Waveform-Space | Gaussian noise, 20 dB SNR | 7.358 | 44,100 Hz | Yes | `noise.wav` |
| `sample_001` | Time Masking | Feature-Space | $T=20$ frames, $n=2$ masks | 7.358 | 44,100 Hz | N/A | `time_mask_spectrogram.png` |
| `sample_001` | Frequency Masking | Feature-Space | $F=15$ mel bins, $n=2$ masks | 7.358 | 44,100 Hz | N/A | `frequency_mask_spectrogram.png` |
| `sample_002` | Baseline Speech | Waveform | Unprocessed source | 3.651 | 44,100 Hz | Yes | `original.wav` |
| `sample_002` | Amplitude Perturbation | Waveform-Space | Gain = -1.506 dB | 3.651 | 44,100 Hz | Yes | `amplitude.wav` |
| `sample_002` | Time Shift | Waveform-Space | Circular shift = +64.85 ms | 3.651 | 44,100 Hz | Yes | `time_shift.wav` |
| `sample_002` | Noise Injection | Waveform-Space | Gaussian noise, 20 dB SNR | 3.651 | 44,100 Hz | Yes | `noise.wav` |
| `sample_002` | Time Masking | Feature-Space | $T=20$ frames, $n=2$ masks | 3.651 | 44,100 Hz | N/A | `time_mask_spectrogram.png` |
| `sample_002` | Frequency Masking | Feature-Space | $F=15$ mel bins, $n=2$ masks | 3.651 | 44,100 Hz | N/A | `frequency_mask_spectrogram.png` |

---

## 5. Quality Verification Summary
- **Decodability**: 100% of the 9 generated WAV files decode cleanly (`scipy.io.wavfile`).
- **Duration Preservation**: Exact duration preservation ($\Delta t = 0.000\text{ s}$).
- **Numeric Stability**: Zero NaN or Inf values across all audio arrays.
- **Amplitude Safeguards**: Peak amplitudes remain strictly in $[-0.53, +0.64]$ with zero threshold clipping (`clipped = False`).
- **Signal Presence**: Healthy speech energy maintained across all files ($\text{RMS} \approx 0.057 - 0.069$, `silent = False`).
