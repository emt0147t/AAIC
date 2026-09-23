"""Audio Augmentation Studio package."""

from .pipeline import AugmentedAudioSample, AugmentationStudioEngine
from .qc import QualityCheckResult, run_10_point_qc
from .transforms import (
    apply_additive_noise,
    apply_frequency_filter,
    apply_gain,
    apply_pitch_shift,
    apply_synthetic_reverb,
    apply_time_shift,
    apply_time_stretch,
)

__all__ = [
    "AugmentedAudioSample",
    "AugmentationStudioEngine",
    "QualityCheckResult",
    "run_10_point_qc",
    "apply_additive_noise",
    "apply_frequency_filter",
    "apply_gain",
    "apply_pitch_shift",
    "apply_synthetic_reverb",
    "apply_time_shift",
    "apply_time_stretch",
]
