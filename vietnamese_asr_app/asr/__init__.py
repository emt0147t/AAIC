"""Vietnamese ASR core package."""

from .audio import AudioValidationResult, load_audio, save_wav_pcm16, validate_audio
from .inference import ASRInferenceEngine, DecodingConfigurationError, TranscriptionResult
from .model import ASRModelManager, HardwareDiagnostics, detect_hardware
from .normalization import normalize_vietnamese_text

__all__ = [
    "AudioValidationResult",
    "load_audio",
    "save_wav_pcm16",
    "validate_audio",
    "ASRInferenceEngine",
    "DecodingConfigurationError",
    "TranscriptionResult",
    "ASRModelManager",
    "HardwareDiagnostics",
    "detect_hardware",
    "normalize_vietnamese_text",
]
