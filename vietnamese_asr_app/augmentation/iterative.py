"""Iterative Audio Manipulation Core Module.

Implements cumulative waveform-space transformations:
x_0 -> x_1 -> x_2 -> ... -> x_K
where x_i = T_i(x_{i-1}).

Strictly operates in 1D waveform time-domain without TTS, voice cloning,
or linguistic content alteration. Enforces 10-point QC after every step.
"""

from dataclasses import asdict, dataclass
import hashlib
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

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


@dataclass
class IterativeStepRecord:
    iteration: int
    parent_iteration: int
    manipulation_name: str
    parameters: Dict[str, Any]
    random_seed: int
    duration_sec: float
    rms: float
    peak_abs: float
    qc_passed: bool
    qc_status: str
    failed_checks: List[str]
    asr_raw_transcript: str
    asr_normalized_transcript: str
    wer: Optional[float]
    cer: Optional[float]
    consistency_drift_wer: Optional[float]
    consistency_drift_cer: Optional[float]
    score: float
    score_type: str  # "ACCURACY_WER_CER" or "ASR_CONSISTENCY_QUALITY_PROXY"
    selected_as_best: bool
    audio_sha256: str
    filename: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["duration_sec"] = round(self.duration_sec, 3)
        d["rms"] = round(self.rms, 5)
        d["peak_abs"] = round(self.peak_abs, 4)
        if self.wer is not None:
            d["wer"] = round(self.wer, 2)
        if self.cer is not None:
            d["cer"] = round(self.cer, 2)
        if self.consistency_drift_wer is not None:
            d["consistency_drift_wer"] = round(self.consistency_drift_wer, 2)
        if self.consistency_drift_cer is not None:
            d["consistency_drift_cer"] = round(self.consistency_drift_cer, 2)
        d["score"] = round(self.score, 4)
        return d


class IterativeAudioManipulator:
    """Waveform manipulator executing bounded, cumulative transformations with QC gating."""

    # Conservative, safe bounds per single iteration to prevent runaway degradation
    OPERATOR_PARAM_BOUNDS = {
        "gain": {"gain_db_range": (-2.5, 2.5)},
        "additive_noise": {"snr_db_range": (22.0, 35.0)},
        "time_shift": {"shift_sec_range": (-0.05, 0.05)},
        "time_stretch": {"rate_range": (0.96, 1.04)},
        "pitch_shift": {"n_steps_range": (-0.8, 0.8)},
        "synthetic_reverb": {"decay_sec_range": (0.08, 0.20), "wet_mix_range": (0.08, 0.20)},
        "bandpass_filter": {"low_cut_range": (180.0, 320.0), "high_cut_range": (3300.0, 3800.0)},
    }

    AVAILABLE_OPERATORS = [
        "gain",
        "additive_noise",
        "time_shift",
        "time_stretch",
        "pitch_shift",
        "synthetic_reverb",
        "bandpass_filter",
    ]

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def apply_operator(
        self,
        audio: np.ndarray,
        op_name: str,
        params: Dict[str, Any],
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Apply a single waveform operator using conservative parameters."""
        if audio is None or len(audio) == 0:
            return np.array([], dtype=np.float32)

        audio_in = audio.astype(np.float32)

        if op_name == "gain":
            gain_db = float(params.get("gain_db", 0.0))
            return apply_gain(audio_in, gain_db=gain_db)

        elif op_name == "additive_noise":
            snr_db = float(params.get("snr_db", 28.0))
            return apply_additive_noise(audio_in, snr_db=snr_db, rng=rng)

        elif op_name == "time_shift":
            shift_sec = float(params.get("shift_sec", 0.0))
            return apply_time_shift(audio_in, shift_sec=shift_sec, sample_rate=self.sample_rate)

        elif op_name == "time_stretch":
            rate = float(params.get("rate", 1.0))
            return apply_time_stretch(audio_in, rate=rate)

        elif op_name == "pitch_shift":
            n_steps = float(params.get("n_steps", 0.0))
            return apply_pitch_shift(audio_in, n_steps=n_steps, sample_rate=self.sample_rate)

        elif op_name == "synthetic_reverb":
            decay_sec = float(params.get("decay_sec", 0.15))
            wet_mix = float(params.get("wet_mix", 0.15))
            return apply_synthetic_reverb(
                audio_in,
                sample_rate=self.sample_rate,
                decay_time_sec=decay_sec,
                wet_mix=wet_mix,
                rng=rng,
            )

        elif op_name == "bandpass_filter":
            low_cut = float(params.get("low_cut", 250.0))
            high_cut = float(params.get("high_cut", 3500.0))
            return apply_frequency_filter(
                audio_in,
                sample_rate=self.sample_rate,
                filter_type="bandpass",
                low_cut=low_cut,
                high_cut=high_cut,
            )

        else:
            raise ValueError(f"Unknown manipulation operator: '{op_name}'")

    def sample_random_operator(
        self,
        rng: np.random.Generator,
        allowed_operators: Optional[List[str]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Sample a valid operator and conservative bounded parameters deterministically."""
        pool = allowed_operators or self.AVAILABLE_OPERATORS
        op_name = str(rng.choice(pool))
        bounds = self.OPERATOR_PARAM_BOUNDS[op_name]
        params: Dict[str, Any] = {}

        if op_name == "gain":
            params["gain_db"] = round(float(rng.uniform(*bounds["gain_db_range"])), 2)
        elif op_name == "additive_noise":
            params["snr_db"] = round(float(rng.uniform(*bounds["snr_db_range"])), 1)
        elif op_name == "time_shift":
            params["shift_sec"] = round(float(rng.uniform(*bounds["shift_sec_range"])), 3)
        elif op_name == "time_stretch":
            params["rate"] = round(float(rng.uniform(*bounds["rate_range"])), 3)
        elif op_name == "pitch_shift":
            params["n_steps"] = round(float(rng.uniform(*bounds["n_steps_range"])), 2)
        elif op_name == "synthetic_reverb":
            params["decay_sec"] = round(float(rng.uniform(*bounds["decay_sec_range"])), 3)
            params["wet_mix"] = round(float(rng.uniform(*bounds["wet_mix_range"])), 3)
        elif op_name == "bandpass_filter":
            params["low_cut"] = round(float(rng.uniform(*bounds["low_cut_range"])), 1)
            params["high_cut"] = round(float(rng.uniform(*bounds["high_cut_range"])), 1)

        return op_name, params

    def evaluate_qc(self, audio: np.ndarray) -> Tuple[QualityCheckResult, float, float]:
        """Run 10-point QC and calculate signal RMS and peak."""
        qc_res = run_10_point_qc(audio, sample_rate=self.sample_rate)
        rms = float(np.sqrt(np.mean(audio**2))) if audio.size > 0 else 0.0
        peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
        return qc_res, rms, peak
