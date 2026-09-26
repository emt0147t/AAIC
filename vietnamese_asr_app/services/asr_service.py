"""ASR Service for Vietnamese Speech AI Studio.

Encapsulates PhoWhisper model lifecycle, audio validation (7-point gate),
safe decoding parameter clamping, and LoRA adapter dynamic attaching/detaching.
Ensures transparent distinction between BASE MODEL and LoRA-ADAPTED MODEL.
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

from asr.audio import AudioValidationResult, load_audio, validate_audio
from asr.inference import ASRInferenceEngine, TranscriptionResult, DecodingConfigurationError
from asr.model import ASRModelManager, HardwareDiagnostics, detect_hardware
from asr.normalization import normalize_vietnamese_text

PINNED_MODEL_REVISIONS = {
    "vinai/PhoWhisper-tiny": "cc51d32be916efebde04ff549854fa1741cb5c02",
    "vinai/PhoWhisper-base": "main",
    "vinai/PhoWhisper-small": "main",
    "vinai/PhoWhisper-medium": "main",
}


class ASRService:
    """Core service for managing ASR inference, base models, and LoRA adapters."""

    def __init__(self, default_model_id: str = "vinai/PhoWhisper-small"):
        self.model_id = default_model_id
        self.engine = ASRInferenceEngine(model_id=default_model_id, default_max_new_tokens=192)
        self._active_adapter_path: Optional[str] = None
        self._active_adapter_name: Optional[str] = None
        self._base_model_ref: Optional[WhisperForConditionalGeneration] = None

    @property
    def is_adapter_active(self) -> bool:
        return self._active_adapter_path is not None

    @property
    def active_adapter_name(self) -> str:
        return self._active_adapter_name or "None (Base Model)"

    @property
    def active_adapter_path(self) -> Optional[str]:
        return self._active_adapter_path

    @property
    def model_revision(self) -> str:
        return PINNED_MODEL_REVISIONS.get(self.model_id, "main")

    def get_hardware_info(self) -> HardwareDiagnostics:
        return detect_hardware()

    def get_compact_status(self) -> str:
        hw = self.get_hardware_info()
        vram_part = f" | VRAM: {hw.vram_allocated_gb:.2f}GB / {hw.vram_total_gb:.2f}GB" if hw.cuda_available else ""
        adapter_part = f" | Adapter: {self.active_adapter_name}" if self.is_adapter_active else " | Mode: BASE MODEL"
        return (
            f"Model: {self.model_id} (rev: {self.model_revision[:8]}){adapter_part} | "
            f"Device: {hw.device} ({hw.gpu_name}){vram_part} | Precision: {hw.recommended_precision}"
        )

    def set_base_model(self, model_id: str) -> None:
        """Switch base model architecture, unloading any currently attached adapter."""
        if self.model_id != model_id:
            self.unload_adapter()
            self.model_id = model_id
            self.engine.model_id = model_id
            self.engine._model = None
            self.engine._processor = None
            self._base_model_ref = None

    def load_adapter(self, checkpoint_or_adapter_dir: str, adapter_name: Optional[str] = None) -> Dict[str, Any]:
        """Attach a LoRA adapter to the base model.
        
        Validates path, checks for adapter config and weights, and wraps the base model.
        """
        # Determine actual adapter directory
        adapter_path = checkpoint_or_adapter_dir
        if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
            nested = os.path.join(checkpoint_or_adapter_dir, "adapter_model")
            if os.path.exists(os.path.join(nested, "adapter_config.json")):
                adapter_path = nested
            else:
                raise FileNotFoundError(
                    f"LoRA adapter could not be loaded: missing adapter_config.json in {checkpoint_or_adapter_dir}"
                )

        # Ensure base engine is loaded
        self.engine._ensure_loaded()
        assert self.engine._model is not None

        # Cache clean base model if not already cached
        if not isinstance(self.engine._model, PeftModel):
            self._base_model_ref = self.engine._model

        # Load PeftModel onto base model
        base_to_wrap = self._base_model_ref if self._base_model_ref is not None else self.engine._model
        peft_model = PeftModel.from_pretrained(
            base_to_wrap,
            adapter_path,
            is_trainable=False,
        )
        peft_model.eval()

        self.engine._model = peft_model
        self._active_adapter_path = adapter_path
        self._active_adapter_name = adapter_name or os.path.basename(checkpoint_or_adapter_dir.rstrip("/\\"))

        return {
            "status": "LOADED",
            "adapter_name": self._active_adapter_name,
            "adapter_path": self._active_adapter_path,
            "model_mode": "BASE + LoRA",
        }

    def unload_adapter(self) -> None:
        """Detach LoRA adapter and restore clean base model."""
        if self._base_model_ref is not None:
            self.engine._model = self._base_model_ref
            self._base_model_ref = None
        elif isinstance(self.engine._model, PeftModel):
            self.engine._model = self.engine._model.get_base_model()

        self._active_adapter_path = None
        self._active_adapter_name = None

    def validate_audio_file(self, audio_file_path: str) -> Tuple[Optional[np.ndarray], int, AudioValidationResult, str]:
        """Load audio and run 7-point validation gate."""
        if not audio_file_path or not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file does not exist: {audio_file_path}")

        audio_16k, sr = load_audio(audio_file_path, target_sr=16000)
        val_res = validate_audio(audio_16k, sample_rate=sr, allow_chunking=True)
        details = self.format_7point_details(audio_16k, sr, val_res)
        return audio_16k, sr, val_res, details

    @staticmethod
    def format_7point_details(audio_16k: np.ndarray, sr: int, val_res: AudioValidationResult) -> str:
        """Format detailed text for the 7-point validation gate."""
        num_samples = len(audio_16k) if audio_16k is not None else 0
        dur = float(num_samples) / float(sr) if sr > 0 else 0.0
        finite_ok = np.all(np.isfinite(audio_16k)) if num_samples > 0 else False
        mono_ok = (audio_16k.ndim == 1) if audio_16k is not None else False
        lower_dur_ok = dur >= 0.2
        upper_dur_ok = dur <= 30.0
        rms_ok = val_res.rms >= 1e-5
        peak_ok = val_res.peak_abs <= 1.2

        lines = [
            f"1. Signal existence:    {'PASS' if num_samples > 0 else 'FAIL'} ({num_samples:,} samples)",
            f"2. Finite values:        {'PASS' if finite_ok else 'FAIL'} (all values finite real numbers)",
            f"3. Mono channel:         {'PASS' if mono_ok else 'FAIL'} (1 channel)",
            f"4. Lower duration bound: {'PASS' if lower_dur_ok else 'FAIL'} ({dur:.2f} s >= 0.20 s)",
            f"5. Upper duration bound: {'PASS' if upper_dur_ok else 'PASS (Chunking supported)'} ({dur:.2f} s <= 30.00 s)",
            f"6. RMS signal level:     {'PASS' if rms_ok else 'FAIL'} ({val_res.rms:.4f} >= 0.00001)",
            f"7. Dynamic range:        {'PASS' if peak_ok else 'FAIL'} (Peak {val_res.peak_abs:.2f} <= 1.20)",
        ]
        return "\n".join(lines)

    def transcribe(
        self,
        audio_16k: np.ndarray,
        strategy: str = "greedy",
        num_beams: int = 4,
        language: str = "vi",
        task: str = "transcribe",
        max_new_tokens: Optional[int] = None,
    ) -> TranscriptionResult:
        """Execute transcription through the loaded model/adapter engine."""
        return self.engine.transcribe(
            audio_16k=audio_16k,
            strategy=strategy,
            num_beams=num_beams,
            language=language,
            task=task,
            max_new_tokens=max_new_tokens,
        )
