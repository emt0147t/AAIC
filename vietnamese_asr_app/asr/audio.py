"""Audio Preprocessing and Validation Module.

Provides format decoding, 16kHz mono resampling, 7-point audio validation,
and lossless 16-bit PCM WAV serialization.
"""

from dataclasses import dataclass
import io
import os
from typing import Optional, Tuple, Union
import numpy as np
import soundfile as sf
import librosa


@dataclass
class AudioValidationResult:
    is_valid: bool
    error_message: Optional[str] = None
    duration_sec: float = 0.0
    sample_rate: int = 16000
    channels: int = 1
    rms: float = 0.0
    peak_abs: float = 0.0
    num_samples: int = 0


def load_audio(
    source: Union[str, bytes, io.BytesIO, Tuple[int, np.ndarray]],
    target_sr: int = 16000,
) -> Tuple[np.ndarray, int]:
    """Load audio from file path, raw bytes, or Gradio microphone tuple into 16kHz mono float32.

    Args:
        source: Audio file path, byte buffer, or (sample_rate, numpy_array) from Gradio.
        target_sr: Target sample rate (default 16000 Hz).

    Returns:
        (audio_array, sample_rate) where audio_array is 1D float32 normalized in [-1.0, 1.0].
    """
    if isinstance(source, tuple):
        # Gradio microphone format: (sr, data)
        sr, y = source
        if y.dtype.kind in ("i", "u"):
            # Integer PCM, convert to float32 in [-1.0, 1.0]
            max_val = np.iinfo(y.dtype).max
            y = y.astype(np.float32) / max_val
        else:
            y = y.astype(np.float32)

        if y.ndim > 1:
            y = np.mean(y, axis=1)

        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        return y, target_sr

    elif isinstance(source, (bytes, io.BytesIO)):
        buf = io.BytesIO(source) if isinstance(source, bytes) else source
        buf.seek(0)
        y, sr = sf.read(buf, dtype="float32")
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        return y, target_sr

    elif isinstance(source, str):
        if not os.path.exists(source):
            raise FileNotFoundError(f"Audio file not found: {source}")
        y, sr = librosa.load(source, sr=target_sr, mono=True, dtype=np.float32)
        return y, target_sr

    else:
        raise ValueError(f"Unsupported audio input source type: {type(source)}")


def validate_audio(
    audio: np.ndarray,
    sample_rate: int = 16000,
    min_duration_sec: float = 0.2,
    max_duration_sec: float = 30.0,
    min_rms: float = 1e-5,
    allow_chunking: bool = False,
) -> AudioValidationResult:
    """Execute 7-point input validation on audio waveform.

    1. Signal existence and array validity
    2. Finite check (no NaNs, no Infs)
    3. Mono 1D array verification
    4. Duration lower-bound check (>= min_duration_sec)
    5. Duration upper-bound check (<= max_duration_sec or chunking supported)
    6. Non-silent signal check (RMS >= min_rms)
    7. Numerical range sanity check (values roughly within [-1.0, 1.0])
    """
    if audio is None or len(audio) == 0:
        return AudioValidationResult(is_valid=False, error_message="Audio signal is empty or None")

    if not np.all(np.isfinite(audio)):
        return AudioValidationResult(is_valid=False, error_message="Audio contains NaN or Infinite values")

    if audio.ndim != 1:
        return AudioValidationResult(is_valid=False, error_message=f"Expected mono 1D audio, got shape {audio.shape}")

    num_samples = len(audio)
    duration_sec = float(num_samples) / float(sample_rate)

    if duration_sec < min_duration_sec:
        return AudioValidationResult(
            is_valid=False,
            error_message=f"Audio duration ({duration_sec:.2f}s) is below minimum threshold ({min_duration_sec}s)",
            duration_sec=duration_sec,
            num_samples=num_samples,
        )

    if duration_sec > max_duration_sec and not allow_chunking:
        return AudioValidationResult(
            is_valid=False,
            error_message=f"Audio duration ({duration_sec:.2f}s) exceeds maximum threshold ({max_duration_sec}s)",
            duration_sec=duration_sec,
            num_samples=num_samples,
        )

    rms = float(np.sqrt(np.mean(audio**2)))
    if rms < min_rms:
        return AudioValidationResult(
            is_valid=False,
            error_message=f"Audio appears completely silent: RMS={rms:.2e} < min={min_rms:.2e}",
            duration_sec=duration_sec,
            rms=rms,
            num_samples=num_samples,
        )

    peak = float(np.max(np.abs(audio)))
    if peak > 1.2:
        return AudioValidationResult(
            is_valid=False,
            error_message=f"Audio signal severely exceeds normal normalized range (peak={peak:.2f})",
            duration_sec=duration_sec,
            peak_abs=peak,
            rms=rms,
            num_samples=num_samples,
        )

    return AudioValidationResult(
        is_valid=True,
        error_message=None,
        duration_sec=duration_sec,
        sample_rate=sample_rate,
        channels=1,
        rms=rms,
        peak_abs=peak,
        num_samples=num_samples,
    )


def save_wav_pcm16(audio: np.ndarray, file_path: str, sample_rate: int = 16000) -> None:
    """Save float32 audio array to 16-bit PCM WAV."""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    # Clip safely to avoid overflow wrap-around before int16 conversion
    clipped = np.clip(audio, -1.0, 1.0)
    int_data = (clipped * 32767.0).astype(np.int16)
    sf.write(file_path, int_data, sample_rate, subtype="PCM_16")
