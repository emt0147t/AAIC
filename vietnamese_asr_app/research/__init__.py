"""Vietnamese ASR Research Layer: PEFT / LoRA Fine-Tuning & Controlled Experiments.

This package provides:
- LoRA model preparation, parameter counting, and freeze integrity checks
- WhisperDataCollatorWithPadding for sequence-to-sequence ASR training
- QuarantineEnforcer preventing train/synthetic test contamination
- Adapter-only and trainer state checkpoint serialization/restoration
- Reproducible experiment configuration schemas for E0, E1, and E2
"""

__version__ = "0.1.0"
