"""Real Waveform Audio Transformations.

All transforms operate strictly in 1D waveform time-domain and produce
decodable, valid audio without spectrogram phase synthesis artifacts.
"""

from typing import Optional, Union
import librosa
import numpy as np
from scipy import signal


def apply_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    """Apply linear amplitude gain in decibels with safe headroom protection.

    Args:
        audio: 1D float32 audio.
        gain_db: Gain in decibels (e.g. -6.0 to +6.0).
    """
    factor = 10.0 ** (gain_db / 20.0)
    out = audio * factor
    peak = np.max(np.abs(out)) if out.size > 0 else 0.0
    if peak > 0.99:
        # Soft-limiting tanh saturation to preserve harmonic transformation without clipping
        # and prevent degenerate identity collapse when applying positive gain to near-peak audio
        tanh_peak = np.tanh(peak)
        if tanh_peak > 1e-7:
            out = (np.tanh(out) / tanh_peak) * 0.98
        else:
            out = np.clip(out, -0.98, 0.98)
    return out.astype(np.float32)


def apply_additive_noise(
    audio: np.ndarray,
    snr_db: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Add calibrated Gaussian noise at exact Signal-to-Noise Ratio (SNR).

    Args:
        audio: 1D float32 audio.
        snr_db: Target SNR in decibels (e.g. 10.0 to 30.0).
        rng: Seeded numpy RandomGenerator for deterministic results.
    """
    if rng is None:
        rng = np.random.default_rng()

    signal_power = np.mean(audio**2)
    if signal_power < 1e-9:
        return audio.copy()

    # Calculate required noise power
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise_std = np.sqrt(noise_power)

    noise = rng.normal(0.0, noise_std, size=len(audio)).astype(np.float32)
    out = audio + noise

    # Safe normalization if peak exceeds 0.99
    peak = np.max(np.abs(out)) if out.size > 0 else 0.0
    if peak > 0.99:
        tanh_peak = np.tanh(peak)
        if tanh_peak > 1e-7:
            out = (np.tanh(out) / tanh_peak) * 0.98
        else:
            out = np.clip(out, -0.98, 0.98)
    return out.astype(np.float32)


def apply_time_shift(
    audio: np.ndarray,
    shift_sec: float,
    sample_rate: int = 16000,
    mode: str = "zero",  # "zero" or "roll"
) -> np.ndarray:
    """Shift audio in time by shift_sec (positive = right, negative = left).

    Args:
        audio: 1D float32 audio.
        shift_sec: Shift duration in seconds.
        sample_rate: Audio sample rate.
        mode: 'zero' (zero-padded) or 'roll' (circular).
    """
    shift_samples = int(round(shift_sec * sample_rate))
    if shift_samples == 0:
        return audio.copy()

    n = len(audio)
    if mode == "roll":
        return np.roll(audio, shift_samples).astype(np.float32)

    # Zero-padded mode
    out = np.zeros_like(audio)
    if shift_samples > 0:
        if shift_samples < n:
            out[shift_samples:] = audio[: n - shift_samples]
    else:
        abs_shift = abs(shift_samples)
        if abs_shift < n:
            out[: n - abs_shift] = audio[abs_shift:]
    return out.astype(np.float32)


def apply_time_stretch(
    audio: np.ndarray,
    rate: float,
) -> np.ndarray:
    """Mild time-stretch without changing pitch using librosa phase vocoder.

    Args:
        audio: 1D float32 audio.
        rate: Speed factor (e.g. 0.9 = slower, 1.1 = faster).
    """
    if abs(rate - 1.0) < 1e-3:
        return audio.copy()

    # Clamp rate to safe range [0.8, 1.25]
    safe_rate = float(np.clip(rate, 0.8, 1.25))
    stretched = librosa.effects.time_stretch(y=audio, rate=safe_rate)
    return stretched.astype(np.float32)


def apply_pitch_shift(
    audio: np.ndarray,
    n_steps: float,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Pitch shift audio by n_steps semitones.

    Args:
        audio: 1D float32 audio.
        n_steps: Number of semitones (-2.0 to +2.0).
        sample_rate: Audio sample rate.
    """
    if abs(n_steps) < 1e-2:
        return audio.copy()

    safe_steps = float(np.clip(n_steps, -3.0, 3.0))
    shifted = librosa.effects.pitch_shift(y=audio, sr=sample_rate, n_steps=safe_steps)
    return shifted.astype(np.float32)


def apply_synthetic_reverb(
    audio: np.ndarray,
    sample_rate: int = 16000,
    decay_time_sec: float = 0.25,
    wet_mix: float = 0.25,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Apply synthetic room impulse response convolution.

    Args:
        audio: 1D float32 audio.
        sample_rate: Audio sample rate.
        decay_time_sec: Room reverberation time (RT60 approximation).
        wet_mix: Ratio of wet reverberant signal (0.0 to 0.5).
        rng: Seeded random generator for impulse response noise.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    ir_len = int(sample_rate * decay_time_sec)
    if ir_len <= 1:
        return audio.copy()

    # Create exponentially decaying noise impulse response
    t = np.linspace(0, decay_time_sec, ir_len, dtype=np.float32)
    decay_curve = np.exp(-3.0 * t / max(decay_time_sec, 1e-4))
    noise = rng.normal(0.0, 1.0, size=ir_len).astype(np.float32)
    ir = noise * decay_curve
    # Direct path spike
    ir[0] += 1.0
    ir /= np.sum(np.abs(ir)) + 1e-8

    # Fast FFT convolution
    wet = signal.fftconvolve(audio, ir, mode="same")
    wet_mix_safe = float(np.clip(wet_mix, 0.0, 0.6))
    out = (1.0 - wet_mix_safe) * audio + wet_mix_safe * wet

    peak = np.max(np.abs(out)) if out.size > 0 else 0.0
    if peak > 0.99:
        tanh_peak = np.tanh(peak)
        if tanh_peak > 1e-7:
            out = (np.tanh(out) / tanh_peak) * 0.98
        else:
            out = np.clip(out, -0.98, 0.98)
    return out.astype(np.float32)


def apply_frequency_filter(
    audio: np.ndarray,
    sample_rate: int = 16000,
    filter_type: str = "bandpass",  # "bandpass", "highpass", "lowpass"
    low_cut: float = 300.0,
    high_cut: float = 3400.0,
    order: int = 4,
) -> np.ndarray:
    """Apply Butterworth IIR filter (e.g. telephone bandpass 300-3400 Hz).

    Args:
        audio: 1D float32 audio.
        sample_rate: Audio sample rate.
        filter_type: 'bandpass', 'highpass', or 'lowpass'.
        low_cut: Low cutoff frequency in Hz.
        high_cut: High cutoff frequency in Hz.
        order: Filter order.
    """
    nyquist = 0.5 * sample_rate

    if filter_type == "bandpass":
        low = max(low_cut / nyquist, 1e-4)
        high = min(high_cut / nyquist, 0.99)
        if low >= high:
            return audio.copy()
        sos = signal.butter(order, [low, high], btype="band", output="sos")
    elif filter_type == "highpass":
        low = max(low_cut / nyquist, 1e-4)
        sos = signal.butter(order, low, btype="highpass", output="sos")
    elif filter_type == "lowpass":
        high = min(high_cut / nyquist, 0.99)
        sos = signal.butter(order, high, btype="lowpass", output="sos")
    else:
        return audio.copy()

    filtered = signal.sosfiltfilt(sos, audio)
    peak = np.max(np.abs(filtered)) if filtered.size > 0 else 0.0
    if peak > 0.99:
        tanh_peak = np.tanh(peak)
        if tanh_peak > 1e-7:
            filtered = (np.tanh(filtered) / tanh_peak) * 0.98
        else:
            filtered = np.clip(filtered, -0.98, 0.98)
    return filtered.astype(np.float32)
