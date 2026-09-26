# Synthetic Vietnamese Speech Pilot Report

> [!NOTE]
> **Status**: `SYNTHETIC PILOT: PASSED`  
> **Benchmark Label**: `SYNTHETIC SPEECH PILOT — DATA GENERATION ONLY`  
> This task generates and validates an auditable synthetic speech dataset for future E2 experimentation. No ASR retraining or performance claims are made in this report.

---

## 1. TTS Backend Model
- **Model ID**: `g-group-ai-lab/gwen-tts-0.6B`
- **Model Revision**: `a83e1f08656d48217a83f0ad422d1610d5b5c963`
- **Architecture**: Qwen3-TTS-0.6B finetuned on 1,000h natural Vietnamese speech
- **Generation API**: `Qwen3TTSModel.generate_voice_clone`

## 2. Environment & Hardware
- **Execution Device**: `cpu`
- **GPU Name**: `N/A (CPU Mode)`
- **CUDA Available**: `False`
- **Python Version**: `3.14.0`
- **PyTorch Version**: `2.10.0+cpu`
- **Transformers Version**: `4.57.6`
- **Qwen-TTS Version**: `0.1.1`

## 3. Source Transcripts
- **Source Manifest**: `manifests/train_manifest.csv`
- **Total Source Transcripts**: `22`
- **Source Datasets**: VIVOS train (10) + ViMD clean train (12)
- **Selection Factor K**: `K = 1` (exactly 1 synthetic utterance per real training transcript)

## 4. Generation Counts & QC Results
- **Total Generated**: `22` utterances
- **QC Passed**: `22` utterances
- **QC Failed**: `0` utterances
- **Pass Rate**: `100.0%`
- **Failure Reasons**: None (0 failed checks across all 10 criteria)

## 5. Audio Duration Statistics
- **Total Audio Duration**: `124.56 seconds` (`2.08 minutes`)
- **Mean Utterance Duration**: `5.66 seconds`
- **Min Duration**: `1.76 seconds`
- **Max Duration**: `14.72 seconds`

## 6. Sample Rate & Format Standardization
- **Raw TTS Output**: `24,000 Hz`
- **Standardized ASR Output**: `16,000 Hz`
- **Channels**: `1 (Mono)`
- **Container / Format**: `PCM_16 WAV`
- **Finite Check**: `100% finite samples (zero NaNs, zero Infs)`

## 7. Synthetic Speaker / Voice Distribution
Built-in reference speakers from Gwen-TTS were used under explicit synthetic IDs (`spk_synth_01` .. `spk_synth_09`), preventing any cloning of real speakers from VIVOS, ViMD, validation, or test sets.

| Synthetic Speaker ID | Reference Voice Name | Gender | Utterances Generated |
| :--- | :--- | :--- | :--- |
| `spk_synth_01` | Yến Nhi (`yen_nhi`) | female | 3 |
| `spk_synth_02` | Mỹ Vân (`my_van`) | female | 3 |
| `spk_synth_03` | Ái Vy (`ai_vy`) | female | 3 |
| `spk_synth_04` | An Nhi (`an_nhi`) | female | 3 |
| `spk_synth_05` | Diệu Linh (`dieu_linh`) | female | 2 |
| `spk_synth_06` | Khánh Toàn (`khanh_toan`) | male | 2 |
| `spk_synth_07` | Trần Lâm (`tran_lam`) | male | 2 |
| `spk_synth_08` | NSND Hà Phương (`nsnd_ha_phuong`) | male | 2 |
| `spk_synth_09` | NSND Kim Cúc (`nsnd_kim_cuc`) | female | 2 |

## 8. Transcript Leakage Audit
- **Synthetic Transcripts $\cap$ Test Transcripts**: `0` (`0%` overlap)
- **Synthetic Transcripts $\cap$ Validation Transcripts**: `0` (`0%` overlap)
- **Source Lineage**: 100% of synthetic transcripts trace directly to `manifests/train_manifest.csv`.

## 9. Manifest Checksums & Cryptographic Hashes
- `manifests/synthetic_manifest.csv`: `21ce3ad0b72ab70734e357a50d35eb1d37d527601a648904f1517afdc2883254`
- `manifests/synthetic_qc_passed.csv`: `21ce3ad0b72ab70734e357a50d35eb1d37d527601a648904f1517afdc2883254`
- `data/synthetic/metadata.csv`: `b0f498fafb8982a6e8f7e8b13a17ed7bb539103036ff4f1ffbc7c930d529f2e4`

## 10. Generation Efficiency
- **Total Generation Time**: `971.17 seconds` (`16.19 minutes`)
- **Mean Real-Time Factor (RTF)**: `7.80` (CPU execution)
- **Git Commit**: `a2dced84868f5988b3c8e783301f814064d393ca`

## 11. Exact Generation Command
```bash
python scripts/generate_synthetic_pilot.py
```

---
> [!IMPORTANT]
> **Label**: `SYNTHETIC SPEECH PILOT — DATA GENERATION ONLY`  
> Dataset is ready and quarantined for controlled E2 LoRA parameter-efficient training.
