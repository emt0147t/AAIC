"""Model Management and Hardware Detection Module.

Manages PhoWhisper model loading, caching, and hardware resource detection.
"""

from dataclasses import dataclass
import gc
from typing import Dict, Optional, Tuple
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


@dataclass
class HardwareDiagnostics:
    device: str
    cuda_available: bool
    gpu_name: Optional[str] = None
    vram_total_gb: Optional[float] = None
    vram_allocated_gb: Optional[float] = None
    torch_version: str = torch.__version__
    recommended_precision: str = "float32"

    def summary(self) -> str:
        if self.cuda_available:
            return (
                f"Device: {self.device} ({self.gpu_name}) | "
                f"VRAM: {self.vram_allocated_gb:.2f}GB / {self.vram_total_gb:.2f}GB | "
                f"Torch: {self.torch_version} | Precision: {self.recommended_precision}"
            )
        return (
            f"Device: CPU (CUDA not available in active torch build) | "
            f"Torch: {self.torch_version} | Precision: {self.recommended_precision}"
        )


def detect_hardware(preferred_device: str = "auto") -> HardwareDiagnostics:
    """Detect local compute capabilities and return diagnostics."""
    cuda_avail = torch.cuda.is_available()
    
    if preferred_device == "cpu" or not cuda_avail:
        return HardwareDiagnostics(
            device="cpu",
            cuda_available=cuda_avail,
            gpu_name="N/A (CPU Mode)",
            vram_total_gb=None,
            vram_allocated_gb=None,
            torch_version=torch.__version__,
            recommended_precision="float32",
        )
    
    # CUDA is available and requested
    dev_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    total_vram = props.total_memory / (1024**3)
    allocated_vram = torch.cuda.memory_allocated(0) / (1024**3)
    
    return HardwareDiagnostics(
        device="cuda:0",
        cuda_available=True,
        gpu_name=dev_name,
        vram_total_gb=round(total_vram, 2),
        vram_allocated_gb=round(allocated_vram, 2),
        torch_version=torch.__version__,
        recommended_precision="float16",
    )


class ASRModelManager:
    """Singleton model cache manager for PhoWhisper / Whisper models."""

    _instances: Dict[str, Tuple[WhisperForConditionalGeneration, WhisperProcessor]] = {}
    _current_device: Optional[str] = None

    @classmethod
    def get_model_and_processor(
        cls,
        model_id: str = "vinai/PhoWhisper-small",
        preferred_device: str = "auto",
    ) -> Tuple[WhisperForConditionalGeneration, WhisperProcessor, HardwareDiagnostics]:
        """Load or retrieve cached model and processor."""
        hw = detect_hardware(preferred_device)
        cache_key = f"{model_id}_{hw.device}"

        if cache_key in cls._instances:
            model, processor = cls._instances[cache_key]
            return model, processor, hw

        # Clear any other models if switching to avoid OOM
        cls.clear_cache()

        print(f"[ASRModelManager] Loading model {model_id} onto {hw.device}...")
        processor = WhisperProcessor.from_pretrained(model_id)

        dtype = torch.float16 if hw.device.startswith("cuda") else torch.float32
        model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            dtype=dtype,
        )
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.forced_decoder_ids = None
        model.to(hw.device)
        model.eval()

        cls._instances[cache_key] = (model, processor)
        cls._current_device = hw.device
        return model, processor, hw

    @classmethod
    def clear_cache(cls) -> None:
        """Release loaded models and free memory."""
        cls._instances.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
