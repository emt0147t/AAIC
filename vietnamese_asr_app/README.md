# 🎙️ Vietnamese Speech-to-Text (ASR) & Data Augmentation Studio

A production-grade Vietnamese Speech Recognition application and data curation system built around the **PhoWhisper** architecture, multi-corpus dataset mixtures, and waveform-space audio augmentation.

---

## 🌟 Key Features

### 1. Vietnamese ASR Model & Inference Engine
- **PhoWhisper Family**: Seamless switching between `vinai/PhoWhisper-tiny` (~39M), `vinai/PhoWhisper-base` (~74M), `vinai/PhoWhisper-small` (~244M), and `vinai/PhoWhisper-medium` (~769M).
- **Decoding Options**: Greedy decoding and configurable Beam Search ($N \in [1, 8]$).
- **Comprehensive Normalization**: Deterministic NFC Unicode normalization, lowercasing, punctuation stripping, and whitespace collapsing.
- **7-Point Input Audio Validation Gate**: Verifies format decodability, duration ($[0.2s, 30.0s]$), non-zero RMS signal ($> 10^{-5}$), finite values, and absence of NaNs/Infs before inference.
- **Performance Profiling**: Real-time reporting of latency (s) and Real-Time Factor (RTF).

### 2. Waveform-Space K-Augmentation Studio
- **Authentic 1D Waveform Transforms**: Generates real acoustic variants without phase artifacts:
  1. Amplitude Gain ($\pm 6\text{ dB}$ with headroom safety)
  2. Calibrated SNR Additive Gaussian / Ambient Noise ($10\text{ dB} - 30\text{ dB}$)
  3. Time Shift ($\pm 0.1 - 0.5\text{s}$)
  4. Mild Time Stretch ($0.9\times - 1.1\times$ speed adjustment)
  5. Pitch Perturbation ($\pm 1 - 2$ semitones)
  6. Synthetic Room Impulse / Reverb (exponential decay convolution)
  7. Butterworth Bandpass Filtering ($300\text{ Hz} - 3400\text{ Hz}$ telephone acoustics)
- **Execution Modes**:
  - **Random K Mode**: Generates $K \in [1, 20]$ variants with deterministic pseudo-random seeds (`seed + sample_idx + aug_idx`).
  - **Exact Plan Mode**: Interactive parameter control over exact operational chains.
- **10-Check Audio Quality Control (QC) Pipeline**:
  - Every generated waveform is strictly evaluated on 10 quality checks: decodability, finite values, no NaNs, no Infs, amplitude bounds $[-1, 1]$, clipping ratio $< 0.1\%$, RMS $> 10^{-5}$, duration boundaries, 16kHz mono format, and cryptographic SHA-256 integrity hash.
- **ASR Robustness & Consistency Analysis**:
  - Live ASR transcription across all augmented variants.
  - Computes transcript drift WER/CER vs original audio and reference text (if provided).

### 3. Multi-Dataset Mixture & Data Governance (6 Corpora)
- Authoritative provenance registry and contamination safeguards across:
  1. **VIVOS**: Clean studio read speech (Gold human, train split only, strict test quarantine).
  2. **ViMD**: Multi-dialect 63 provinces speech (~102.56h, Gold human).
  3. **BUD500**: Broadcast diverse topics (Gold human, CC-BY-NC-4.0).
  4. **VietMed**: Medical teleconsultation speech (Gold human, domain adaptation).
  5. **VietSuperSpeech**: Large-scale spontaneous podcast speech (Silver pseudo-labeled by Zipformer-30M).
  6. **VLSP Restricted**: Competition benchmark (Restricted evaluation data).
- **Speaker-Disjoint Partitioner**: Mathematically guarantees zero speaker leakage between train and validation splits.
- **Zero Test Contamination**: Strictly rejects quarantined test sample IDs from training or validation manifests.

### 4. Memory-Efficient Batch Audio Processing
- Sequential stream processing of multi-file queues without bulk RAM consumption.
- Generates summary manifests and downloadable batch CSV results.

### 5. Structured Export & Packaging
- Exports complete sessions to `exports/{session_id}/`:
  - `original/original.wav`
  - `augmented/aug_*.wav`
  - `transcripts/*.txt`
  - `metadata/*.json`
  - `manifest.csv` and `results.json`
  - Single-click standalone `.zip` bundle.

### 6. Dynamic Hardware Diagnostics
- Auto-detects NVIDIA CUDA GPU vs CPU.
- Automatically configures `float16` on GPU and `float32` on CPU with VRAM tracking.

---

## 📁 Project Structure

```text
vietnamese_asr_app/
├── app.py                      # Interactive Gradio 7-Tab Application
├── configs/
│   ├── app.yaml                # Application & audio QC thresholds
│   ├── model.yaml              # PhoWhisper model specifications
│   └── dataset_mixture.yaml    # 6-dataset mixture & partition configuration
├── asr/
│   ├── audio.py                # 7-point audio validation & 16kHz resampling
│   ├── inference.py            # PhoWhisper inference & RTF calculation
│   ├── model.py                # Model loader & hardware detection
│   └── normalization.py        # Vietnamese NFC text normalization
├── augmentation/
│   ├── pipeline.py             # Random K & Exact Plan orchestrator
│   ├── qc.py                   # 10-point Audio Quality Control pipeline
│   └── transforms.py           # Waveform transformations (gain, noise, reverb, etc.)
├── datasets/
│   ├── mixture.py              # Sampler, speaker-disjoint partitioner & manifest builder
│   ├── provenance.py           # Provenance schemas & quality tiers
│   └── registry.py             # Registry for 6 candidate corpora
├── evaluation/
│   └── metrics.py              # Levenshtein WER, CER & drift consistency
├── export/
│   └── exporter.py             # Artifact packaging & ZIP bundle exporter
├── scripts/
│   ├── audit_dataset.py        # CLI dataset provenance & schema inspector
│   ├── build_manifest.py       # CLI manifest bootstrap tool
│   ├── evaluate.py             # CLI ASR evaluation script
│   └── train.py                # CLI training script with resumability & dry-run
├── tests/                      # 25 automated pytest unit tests
│   ├── test_export.py
│   ├── test_inference.py
│   ├── test_manifest.py
│   ├── test_metrics.py
│   ├── test_qc.py
│   └── test_transforms.py
├── requirements.txt            # Dependency specifications
└── README.md                   # Documentation
```

---

## 🚀 Quickstart Guide

### 1. Installation
Ensure Python 3.10+ is installed:

```bash
cd vietnamese_asr_app
pip install -r requirements.txt
```

### 2. Launch the Gradio Web Studio
Start the interactive local web application:

```bash
python app.py
```
Open your browser at `http://127.0.0.1:7860`.

### 3. Running Automated Tests
Run the comprehensive test suite:

```bash
python -m pytest tests
```
*All 25 unit tests verify QC gates, transforms, speaker disjointness, and metrics.*

---

## 💻 CLI Commands & Workflows

### Dataset Audit & Governance
Inspect provenance, licenses, and quality tiers of candidate datasets:

```bash
python scripts/audit_dataset.py --all
```

### Build Frozen Dataset Mixture Manifests
Partition multi-corpus datasets with speaker disjointness:

```bash
python scripts/build_manifest.py --config configs/dataset_mixture.yaml --out_dir manifests
```

### Training Verification (Dry-Run Mode)
Verify training initialization, protocol fingerprinting, and checkpoint saving/resuming without training:

```bash
python scripts/train.py --config configs/dataset_mixture.yaml --dry_run
```

### Resume Training from Checkpoint
```bash
python scripts/train.py --config configs/dataset_mixture.yaml --resume_from checkpoints/checkpoint_epoch_2.pt
```

### Benchmark Evaluation CLI
Evaluate PhoWhisper on a manifest CSV:

```bash
python scripts/evaluate.py --manifest manifests/test_manifest.csv --model_id vinai/PhoWhisper-small
```
