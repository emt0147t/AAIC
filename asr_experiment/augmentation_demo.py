"""
Gate 5 — Audio Augmentation Reconstruction Demo
================================================
LABEL: "augmentation reconstruction based on the weekly report methodology"
NO original MixMatch/ReMixMatch implementation found in workspace.

Uses 3-5 Vietnamese utterances. CPU only. No full dataset download.
"""
import sys, json, os, io, csv, struct
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import torch

# ===== CONSTANTS =====
SEED = 42
SR = 16000  # target sample rate
OUT_ROOT = r"d:\Vietnamese_ASR_Week6\asr_experiment\augmentation_demo"
np.random.seed(SEED)
torch.manual_seed(SEED)

# ===== STEP 1: Obtain a few audio samples via HF streaming =====
def get_vimd_samples(n=3):
    """Stream n samples from ViMD, extract raw audio bytes as WAV."""
    from datasets import load_dataset, Audio
    print(f"Streaming {n} samples from ViMD (decode=False)...")
    ds = load_dataset("nguyendv02/ViMD_Dataset", split="test", streaming=True)
    # Get raw bytes (no decode)
    ds = ds.cast_column("audio", Audio(decode=False))
    
    samples = []
    for i, item in enumerate(ds):
        if i >= n:
            break
        audio_data = item["audio"]
        raw_bytes = audio_data.get("bytes", b"")
        path = audio_data.get("path", f"sample_{i}")
        text = item.get("text", "")
        speaker = item.get("speakerID", "unknown")
        region = item.get("region", "unknown")
        province = item.get("province_name", "unknown")
        
        samples.append({
            "idx": i,
            "path": path,
            "raw_bytes": raw_bytes,
            "text": text,
            "speaker": speaker,
            "region": region,
            "province": province,
        })
        print(f"  [{i}] {path} spk={speaker} region={region} bytes={len(raw_bytes)}")
    
    return samples

def decode_wav_bytes(raw_bytes):
    """Decode WAV bytes to numpy array + sample rate using scipy."""
    from scipy.io import wavfile
    buf = io.BytesIO(raw_bytes)
    try:
        sr, data = wavfile.read(buf)
        # Normalize to float32 [-1, 1]
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.float32:
            pass
        else:
            data = data.astype(np.float32)
        # Mono
        if data.ndim > 1:
            data = data.mean(axis=1)
        return sr, data
    except Exception as e:
        print(f"  WAV decode failed: {e}, trying raw parse...")
        return None, None

def save_wav(filepath, audio, sr):
    """Save float32 audio as 16-bit PCM WAV."""
    from scipy.io import wavfile
    # Clip to [-1, 1] and convert to int16
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    wavfile.write(filepath, sr, audio_int16)

# ===== STEP 2: Augmentation Functions =====

def amplitude_perturbation(audio, gain_db=6.0, seed=42):
    """WAVEFORM-SPACE: Scale amplitude by a random gain factor."""
    rng = np.random.RandomState(seed)
    # Random gain between -gain_db and +gain_db
    gain = rng.uniform(-gain_db, gain_db)
    factor = 10 ** (gain / 20.0)
    return audio * factor, {"gain_db": round(gain, 3), "factor": round(factor, 4)}

def time_shift(audio, max_shift_ms=100, sr=16000, seed=42):
    """WAVEFORM-SPACE: Circular shift audio by random amount."""
    rng = np.random.RandomState(seed)
    max_samples = int(max_shift_ms * sr / 1000)
    shift = rng.randint(-max_samples, max_samples)
    shifted = np.roll(audio, shift)
    return shifted, {"shift_samples": int(shift), "shift_ms": round(shift / sr * 1000, 2)}

def noise_injection(audio, snr_db=20.0, seed=42):
    """WAVEFORM-SPACE: Add Gaussian noise at specified SNR."""
    rng = np.random.RandomState(seed)
    signal_power = np.mean(audio ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), len(audio)).astype(np.float32)
    noisy = audio + noise
    return noisy, {"snr_db": snr_db, "signal_rms": round(float(np.sqrt(signal_power)), 6),
                   "noise_rms": round(float(np.sqrt(noise_power)), 6)}

def compute_spectrogram(audio, sr=16000, n_fft=512, hop_length=160, n_mels=80):
    """Compute log-mel spectrogram."""
    import librosa
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmin=20, fmax=8000
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel

def time_mask_spectrogram(spec, T=20, num_masks=2, seed=42):
    """FEATURE-SPACE: Apply SpecAugment time masking to spectrogram."""
    rng = np.random.RandomState(seed)
    spec_masked = spec.copy()
    _, tau = spec.shape
    params = []
    for _ in range(num_masks):
        t = rng.randint(0, min(T, tau))
        t0 = rng.randint(0, max(1, tau - t))
        spec_masked[:, t0:t0+t] = spec.min()
        params.append({"t0": int(t0), "width": int(t), "max_T": T})
    return spec_masked, {"masks": params, "num_masks": num_masks, "max_T": T}

def freq_mask_spectrogram(spec, F=15, num_masks=2, seed=42):
    """FEATURE-SPACE: Apply SpecAugment frequency masking to spectrogram."""
    rng = np.random.RandomState(seed)
    spec_masked = spec.copy()
    n_mels, _ = spec.shape
    params = []
    for _ in range(num_masks):
        f = rng.randint(0, min(F, n_mels))
        f0 = rng.randint(0, max(1, n_mels - f))
        spec_masked[f0:f0+f, :] = spec.min()
        params.append({"f0": int(f0), "width": int(f), "max_F": F})
    return spec_masked, {"masks": params, "num_masks": num_masks, "max_F": F}

# ===== STEP 3: Visualization =====

def plot_comparison(audio_orig, sr, augmented_dict, spec_orig, spec_augmented_dict,
                    sample_id, out_dir):
    """Create a compact comparison figure for one sample."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Count total subplots: original waveform + original spec + each augmentation
    waveform_augs = {k: v for k, v in augmented_dict.items() if v is not None}
    spec_augs = {k: v for k, v in spec_augmented_dict.items() if v is not None}
    
    n_rows = 2 + len(waveform_augs) + len(spec_augs)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 2.2 * n_rows))
    
    # Original waveform
    ax = axes[0]
    t = np.arange(len(audio_orig)) / sr
    ax.plot(t, audio_orig, linewidth=0.3, color='steelblue')
    ax.set_title(f"{sample_id} — Original Waveform ({sr} Hz, {len(audio_orig)/sr:.1f}s)", fontsize=9)
    ax.set_ylabel("Amplitude", fontsize=8)
    ax.set_xlim(0, t[-1])
    ax.tick_params(labelsize=7)
    
    # Original spectrogram
    ax = axes[1]
    ax.imshow(spec_orig, aspect='auto', origin='lower', cmap='magma',
              extent=[0, len(audio_orig)/sr, 0, sr/2/1000])
    ax.set_title(f"{sample_id} — Original Log-Mel Spectrogram", fontsize=9)
    ax.set_ylabel("Freq (kHz)", fontsize=8)
    ax.tick_params(labelsize=7)
    
    row = 2
    # Waveform augmentations
    for name, aug_audio in waveform_augs.items():
        ax = axes[row]
        t_aug = np.arange(len(aug_audio)) / sr
        ax.plot(t_aug, aug_audio, linewidth=0.3, color='darkorange')
        ax.set_title(f"{name} (waveform-space)", fontsize=9)
        ax.set_ylabel("Amplitude", fontsize=8)
        ax.set_xlim(0, t_aug[-1])
        ax.tick_params(labelsize=7)
        row += 1
    
    # Spectrogram augmentations (feature-space)
    for name, aug_spec in spec_augs.items():
        ax = axes[row]
        ax.imshow(aug_spec, aspect='auto', origin='lower', cmap='magma',
                  extent=[0, len(audio_orig)/sr, 0, sr/2/1000])
        ax.set_title(f"{name} (feature-space)", fontsize=9)
        ax.set_ylabel("Freq (kHz)", fontsize=8)
        ax.tick_params(labelsize=7)
        row += 1
    
    axes[-1].set_xlabel("Time (s)", fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, "comparison.png")
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return path

def plot_summary(all_specs, out_path):
    """One summary figure comparing all augmentation types across samples."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    aug_types = ["original", "time_mask", "freq_mask", "amplitude", "time_shift", "noise"]
    n_samples = len(all_specs)
    
    fig, axes = plt.subplots(n_samples, len(aug_types), figsize=(2.5*len(aug_types), 2.2*n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]
    
    for i, (sample_id, specs) in enumerate(all_specs.items()):
        for j, aug_name in enumerate(aug_types):
            ax = axes[i, j]
            if aug_name in specs and specs[aug_name] is not None:
                ax.imshow(specs[aug_name], aspect='auto', origin='lower', cmap='magma')
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(aug_name.replace("_", " ").title(), fontsize=8, fontweight='bold')
            if j == 0:
                ax.set_ylabel(sample_id, fontsize=7, rotation=0, labelpad=50, va='center')
    
    plt.suptitle("Augmentation Comparison (Log-Mel Spectrograms)", fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)

# ===== STEP 4: Quality Check =====

def quality_check(filepath, audio, sr):
    """Verify WAV file quality."""
    checks = {
        "file_exists": os.path.isfile(filepath),
        "sample_rate": sr,
        "channels": 1,
        "duration_s": round(len(audio) / sr, 3),
        "n_samples": len(audio),
        "has_nan": bool(np.any(np.isnan(audio))),
        "has_inf": bool(np.any(np.isinf(audio))),
        "amplitude_min": round(float(np.min(audio)), 6),
        "amplitude_max": round(float(np.max(audio)), 6),
        "rms": round(float(np.sqrt(np.mean(audio**2))), 6),
        "clipped": bool(np.sum(np.abs(audio) >= 0.999) > 10),
        "silent": bool(np.sqrt(np.mean(audio**2)) < 1e-5),
    }
    # Verify can be decoded
    try:
        from scipy.io import wavfile
        sr2, data2 = wavfile.read(filepath)
        checks["decodable"] = True
        checks["decoded_sr"] = sr2
    except Exception as e:
        checks["decodable"] = False
        checks["decode_error"] = str(e)
    
    return checks

# ===== MAIN =====

def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    
    # Get 3 samples from ViMD test split
    print("=" * 60)
    print("STEP 1: Fetching audio samples")
    print("=" * 60)
    raw_samples = get_vimd_samples(n=3)
    
    # Decode WAV bytes
    print("\n" + "=" * 60)
    print("STEP 2: Decoding audio")
    print("=" * 60)
    samples = []
    for s in raw_samples:
        sr, audio = decode_wav_bytes(s["raw_bytes"])
        if sr is None:
            print(f"  SKIP sample {s['idx']}: decode failed")
            continue
        print(f"  Sample {s['idx']}: sr={sr}, duration={len(audio)/sr:.2f}s, samples={len(audio)}")
        s["sr"] = sr
        s["audio"] = audio
        samples.append(s)
    
    if not samples:
        print("ERROR: No samples could be decoded!")
        return
    
    # Import matplotlib/librosa (installed earlier)
    import librosa
    
    # Process each sample
    print("\n" + "=" * 60)
    print("STEP 3: Generating augmentations")
    print("=" * 60)
    
    all_specs_for_summary = {}
    summary_rows = []
    
    for s in samples:
        sample_id = f"sample_{s['idx']:03d}"
        sample_dir = os.path.join(OUT_ROOT, sample_id)
        os.makedirs(sample_dir, exist_ok=True)
        
        audio = s["audio"]
        sr = s["sr"]
        
        print(f"\n--- {sample_id} (sr={sr}, dur={len(audio)/sr:.2f}s, spk={s['speaker']}) ---")
        
        # Save original
        orig_path = os.path.join(sample_dir, "original.wav")
        save_wav(orig_path, audio, sr)
        
        # Compute original spectrogram
        spec_orig = compute_spectrogram(audio, sr=sr)
        
        metadata = {
            "sample_id": sample_id,
            "source_dataset": "ViMD (nguyendv02/ViMD_Dataset, test split)",
            "source_path": s["path"],
            "speaker": s["speaker"],
            "region": s["region"],
            "province": s["province"],
            "transcript": s["text"][:200],
            "original_sr": sr,
            "original_duration_s": round(len(audio) / sr, 3),
            "original_samples": len(audio),
            "seed": SEED,
            "label": "augmentation reconstruction based on the weekly report methodology",
            "augmentations": {},
            "quality_checks": {},
        }
        
        waveform_augmented = {}
        spec_augmented = {}
        all_sample_specs = {"original": spec_orig}
        
        # --- WAVEFORM-SPACE AUGMENTATIONS ---
        
        # 04: Amplitude perturbation
        aug_amp, params_amp = amplitude_perturbation(audio, gain_db=6.0, seed=SEED)
        amp_path = os.path.join(sample_dir, "amplitude.wav")
        save_wav(amp_path, aug_amp, sr)
        waveform_augmented["04_amplitude_perturbation"] = aug_amp
        metadata["augmentations"]["04_amplitude_perturbation"] = {
            "representation": "waveform-space",
            "input": "float32 waveform",
            "output": "float32 waveform → 16-bit PCM WAV",
            "parameters": params_amp,
            "seed": SEED,
            "library": "numpy",
            "description": "Random gain between -6dB and +6dB",
            "output_file": "amplitude.wav"
        }
        qc = quality_check(amp_path, aug_amp, sr)
        metadata["quality_checks"]["amplitude"] = qc
        print(f"  amplitude_perturbation: gain={params_amp['gain_db']}dB OK={not qc['has_nan']}")
        
        spec_amp = compute_spectrogram(aug_amp, sr=sr)
        all_sample_specs["amplitude"] = spec_amp
        
        summary_rows.append({
            "sample_id": sample_id, "augmentation": "amplitude_perturbation",
            "representation": "waveform-space", "parameter_summary": f"gain={params_amp['gain_db']}dB",
            "duration_before": round(len(audio)/sr, 3), "duration_after": round(len(aug_amp)/sr, 3),
            "sample_rate": sr, "output_path": "amplitude.wav", "reconstructed_or_rerun": "RECONSTRUCTION"
        })
        
        # 05: Time shift
        aug_ts, params_ts = time_shift(audio, max_shift_ms=100, sr=sr, seed=SEED)
        ts_path = os.path.join(sample_dir, "time_shift.wav")
        save_wav(ts_path, aug_ts, sr)
        waveform_augmented["05_time_shift"] = aug_ts
        metadata["augmentations"]["05_time_shift"] = {
            "representation": "waveform-space",
            "input": "float32 waveform",
            "output": "float32 waveform → 16-bit PCM WAV",
            "parameters": params_ts,
            "seed": SEED,
            "library": "numpy (np.roll)",
            "description": f"Circular shift by {params_ts['shift_ms']}ms",
            "output_file": "time_shift.wav"
        }
        qc = quality_check(ts_path, aug_ts, sr)
        metadata["quality_checks"]["time_shift"] = qc
        print(f"  time_shift: shift={params_ts['shift_ms']}ms OK={not qc['has_nan']}")
        
        spec_ts = compute_spectrogram(aug_ts, sr=sr)
        all_sample_specs["time_shift"] = spec_ts
        
        summary_rows.append({
            "sample_id": sample_id, "augmentation": "time_shift",
            "representation": "waveform-space", "parameter_summary": f"shift={params_ts['shift_ms']}ms",
            "duration_before": round(len(audio)/sr, 3), "duration_after": round(len(aug_ts)/sr, 3),
            "sample_rate": sr, "output_path": "time_shift.wav", "reconstructed_or_rerun": "RECONSTRUCTION"
        })
        
        # 06: Noise injection
        aug_noise, params_noise = noise_injection(audio, snr_db=20.0, seed=SEED)
        noise_path = os.path.join(sample_dir, "noise.wav")
        save_wav(noise_path, aug_noise, sr)
        waveform_augmented["06_noise_injection"] = aug_noise
        metadata["augmentations"]["06_noise_injection"] = {
            "representation": "waveform-space",
            "input": "float32 waveform",
            "output": "float32 waveform → 16-bit PCM WAV",
            "parameters": params_noise,
            "seed": SEED,
            "library": "numpy (Gaussian noise)",
            "description": f"Additive Gaussian noise at {params_noise['snr_db']}dB SNR",
            "output_file": "noise.wav"
        }
        qc = quality_check(noise_path, aug_noise, sr)
        metadata["quality_checks"]["noise"] = qc
        print(f"  noise_injection: snr={params_noise['snr_db']}dB OK={not qc['has_nan']}")
        
        spec_noise = compute_spectrogram(aug_noise, sr=sr)
        all_sample_specs["noise"] = spec_noise
        
        summary_rows.append({
            "sample_id": sample_id, "augmentation": "noise_injection",
            "representation": "waveform-space", "parameter_summary": f"snr={params_noise['snr_db']}dB",
            "duration_before": round(len(audio)/sr, 3), "duration_after": round(len(aug_noise)/sr, 3),
            "sample_rate": sr, "output_path": "noise.wav", "reconstructed_or_rerun": "RECONSTRUCTION"
        })
        
        # --- FEATURE-SPACE AUGMENTATIONS ---
        
        # 02: Time masking (SpecAugment) — FEATURE-SPACE ONLY
        spec_tm, params_tm = time_mask_spectrogram(spec_orig, T=20, num_masks=2, seed=SEED)
        spec_augmented["02_time_mask"] = spec_tm
        all_sample_specs["time_mask"] = spec_tm
        
        # Save spectrogram image (NO WAV — this is feature-space)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        ax.imshow(spec_tm, aspect='auto', origin='lower', cmap='magma')
        ax.set_title("Time Mask (feature-space transformation)", fontsize=9)
        ax.set_ylabel("Mel bin", fontsize=8)
        ax.set_xlabel("Time frame", fontsize=8)
        plt.tight_layout()
        tm_spec_path = os.path.join(sample_dir, "time_mask_spectrogram.png")
        fig.savefig(tm_spec_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        metadata["augmentations"]["02_time_mask"] = {
            "representation": "feature-space (log-mel spectrogram)",
            "input": "log-mel spectrogram (80 mels, n_fft=512, hop=160)",
            "output": "masked log-mel spectrogram (PNG image)",
            "parameters": params_tm,
            "seed": SEED,
            "library": "numpy (SpecAugment-style)",
            "description": "SpecAugment time masking: 2 masks, max width 20 frames",
            "output_file": "time_mask_spectrogram.png",
            "NO_WAV": "feature-space transformation — no WAV generated"
        }
        print(f"  time_mask: {params_tm['num_masks']} masks, max_T={params_tm['max_T']} [FEATURE-SPACE]")
        
        summary_rows.append({
            "sample_id": sample_id, "augmentation": "time_mask",
            "representation": "feature-space", "parameter_summary": f"T={params_tm['max_T']},n={params_tm['num_masks']}",
            "duration_before": round(len(audio)/sr, 3), "duration_after": round(len(audio)/sr, 3),
            "sample_rate": sr, "output_path": "time_mask_spectrogram.png", "reconstructed_or_rerun": "RECONSTRUCTION"
        })
        
        # 03: Frequency masking (SpecAugment) — FEATURE-SPACE ONLY
        spec_fm, params_fm = freq_mask_spectrogram(spec_orig, F=15, num_masks=2, seed=SEED)
        spec_augmented["03_freq_mask"] = spec_fm
        all_sample_specs["freq_mask"] = spec_fm
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        ax.imshow(spec_fm, aspect='auto', origin='lower', cmap='magma')
        ax.set_title("Frequency Mask (feature-space transformation)", fontsize=9)
        ax.set_ylabel("Mel bin", fontsize=8)
        ax.set_xlabel("Time frame", fontsize=8)
        plt.tight_layout()
        fm_spec_path = os.path.join(sample_dir, "frequency_mask_spectrogram.png")
        fig.savefig(fm_spec_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        metadata["augmentations"]["03_freq_mask"] = {
            "representation": "feature-space (log-mel spectrogram)",
            "input": "log-mel spectrogram (80 mels, n_fft=512, hop=160)",
            "output": "masked log-mel spectrogram (PNG image)",
            "parameters": params_fm,
            "seed": SEED,
            "library": "numpy (SpecAugment-style)",
            "description": "SpecAugment frequency masking: 2 masks, max width 15 mel bins",
            "output_file": "frequency_mask_spectrogram.png",
            "NO_WAV": "feature-space transformation — no WAV generated"
        }
        print(f"  freq_mask: {params_fm['num_masks']} masks, max_F={params_fm['max_F']} [FEATURE-SPACE]")
        
        summary_rows.append({
            "sample_id": sample_id, "augmentation": "freq_mask",
            "representation": "feature-space", "parameter_summary": f"F={params_fm['max_F']},n={params_fm['num_masks']}",
            "duration_before": round(len(audio)/sr, 3), "duration_after": round(len(audio)/sr, 3),
            "sample_rate": sr, "output_path": "frequency_mask_spectrogram.png", "reconstructed_or_rerun": "RECONSTRUCTION"
        })
        
        # Save original spectrogram image
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        ax.imshow(spec_orig, aspect='auto', origin='lower', cmap='magma')
        ax.set_title("Original Log-Mel Spectrogram", fontsize=9)
        ax.set_ylabel("Mel bin", fontsize=8)
        ax.set_xlabel("Time frame", fontsize=8)
        plt.tight_layout()
        fig.savefig(os.path.join(sample_dir, "original_spectrogram.png"), dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        # Comparison figure
        print(f"  Creating comparison figure...")
        plot_comparison(audio, sr, waveform_augmented, spec_orig, spec_augmented,
                       sample_id, sample_dir)
        
        # Save metadata
        # Remove audio arrays from metadata (not JSON serializable)
        with open(os.path.join(sample_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        
        all_specs_for_summary[sample_id] = all_sample_specs
        
        # Add original row to summary
        summary_rows.append({
            "sample_id": sample_id, "augmentation": "original",
            "representation": "waveform", "parameter_summary": "none",
            "duration_before": round(len(audio)/sr, 3), "duration_after": round(len(audio)/sr, 3),
            "sample_rate": sr, "output_path": "original.wav", "reconstructed_or_rerun": "N/A"
        })
    
    # Summary comparison figure
    print("\n" + "=" * 60)
    print("STEP 4: Creating summary figure")
    print("=" * 60)
    summary_fig_path = os.path.join(OUT_ROOT, "summary_comparison.png")
    plot_summary(all_specs_for_summary, summary_fig_path)
    print(f"  Saved: {summary_fig_path}")
    
    # Summary CSV
    csv_path = os.path.join(OUT_ROOT, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sample_id", "augmentation", "representation", "parameter_summary",
            "duration_before", "duration_after", "sample_rate", "output_path",
            "reconstructed_or_rerun"
        ])
        writer.writeheader()
        writer.writerows(sorted(summary_rows, key=lambda x: (x["sample_id"], x["augmentation"])))
    print(f"  Saved: {csv_path}")
    
    print("\n" + "=" * 60)
    print("GATE 5 AUGMENTATION DEMO COMPLETE")
    print(f"Output: {OUT_ROOT}")
    print(f"Samples processed: {len(samples)}")
    print(f"WAV augmentations: amplitude, time_shift, noise (3 per sample)")
    print(f"Feature-space augmentations: time_mask, freq_mask (2 per sample)")
    print(f"GPU used: NO (CPU only)")
    print("=" * 60)

if __name__ == "__main__":
    main()
