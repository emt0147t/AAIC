"""Augmentation Pipeline Orchestrator.

Implements Random K mode and Exact Plan mode with deterministic seeding,
per-sample QC gating, and metadata generation.
"""

from dataclasses import asdict, dataclass
import time
from typing import Any, Dict, List, Optional, Tuple, Union
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
class AugmentedAudioSample:
    aug_id: str
    aug_index: int
    audio_16k: np.ndarray
    sample_rate: int
    duration_sec: float
    transform_chain: List[Dict[str, Any]]
    qc_result: QualityCheckResult
    generation_latency_sec: float
    seed: int

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "aug_id": self.aug_id,
            "aug_index": self.aug_index,
            "sample_rate": self.sample_rate,
            "duration_sec": round(self.duration_sec, 3),
            "transform_chain": self.transform_chain,
            "qc_passed": self.qc_result.passed,
            "qc_status": self.qc_result.status,
            "sha256": self.qc_result.sha256_hash,
            "failed_checks": self.qc_result.failed_checks,
            "generation_latency_sec": round(self.generation_latency_sec, 4),
            "seed": self.seed,
        }


class AugmentationStudioEngine:
    """Core generator for waveform-space audio augmentation."""

    AVAILABLE_TRANSFORMS = [
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

    def generate_random_k(
        self,
        audio_16k: np.ndarray,
        k: int = 3,
        base_seed: int = 42,
        sample_idx: int = 0,
        enabled_transforms: Optional[List[str]] = None,
    ) -> List[AugmentedAudioSample]:
        """Generate K distinct augmented waveforms using deterministic seeded random sampling.

        Args:
            audio_16k: Input 16kHz mono audio.
            k: Number of augmentations to generate (1 <= k <= 20).
            base_seed: Master seed.
            sample_idx: Sample index for reproducible multi-sample processing.
            enabled_transforms: Subset of transform names to pick from.
        """
        k = max(1, min(20, int(k)))
        if enabled_transforms is None:
            enabled_transforms = list(self.AVAILABLE_TRANSFORMS)

        results: List[AugmentedAudioSample] = []

        for aug_idx in range(1, k + 1):
            aug_seed = base_seed + (sample_idx * 1000) + aug_idx
            rng = np.random.default_rng(aug_seed)

            t0 = time.perf_counter()
            current_audio = audio_16k.copy()
            chain_log: List[Dict[str, Any]] = []

            # Deterministically choose 1 to 3 distinct transforms for this augmentation
            num_ops = rng.integers(1, 4)
            chosen_ops = rng.choice(enabled_transforms, size=min(num_ops, len(enabled_transforms)), replace=False)

            for op in chosen_ops:
                if op == "gain":
                    gain_db = float(rng.uniform(-6.0, 6.0))
                    current_audio = apply_gain(current_audio, gain_db=gain_db)
                    chain_log.append({"op": "gain", "gain_db": round(gain_db, 2)})

                elif op == "additive_noise":
                    snr_db = float(rng.uniform(12.0, 30.0))
                    current_audio = apply_additive_noise(current_audio, snr_db=snr_db, rng=rng)
                    chain_log.append({"op": "additive_noise", "snr_db": round(snr_db, 2)})

                elif op == "time_shift":
                    shift_sec = float(rng.uniform(-0.3, 0.3))
                    current_audio = apply_time_shift(current_audio, shift_sec=shift_sec, sample_rate=self.sample_rate)
                    chain_log.append({"op": "time_shift", "shift_sec": round(shift_sec, 3)})

                elif op == "time_stretch":
                    rate = float(rng.uniform(0.92, 1.08))
                    current_audio = apply_time_stretch(current_audio, rate=rate)
                    chain_log.append({"op": "time_stretch", "rate": round(rate, 3)})

                elif op == "pitch_shift":
                    n_steps = float(rng.uniform(-1.5, 1.5))
                    current_audio = apply_pitch_shift(current_audio, n_steps=n_steps, sample_rate=self.sample_rate)
                    chain_log.append({"op": "pitch_shift", "n_steps": round(n_steps, 2)})

                elif op == "synthetic_reverb":
                    decay = float(rng.uniform(0.15, 0.35))
                    wet = float(rng.uniform(0.15, 0.30))
                    current_audio = apply_synthetic_reverb(
                        current_audio,
                        sample_rate=self.sample_rate,
                        decay_time_sec=decay,
                        wet_mix=wet,
                        rng=rng,
                    )
                    chain_log.append({"op": "synthetic_reverb", "decay_sec": round(decay, 2), "wet_mix": round(wet, 2)})

                elif op == "bandpass_filter":
                    low_cut = float(rng.uniform(200.0, 400.0))
                    high_cut = float(rng.uniform(3200.0, 3800.0))
                    current_audio = apply_frequency_filter(
                        current_audio,
                        sample_rate=self.sample_rate,
                        filter_type="bandpass",
                        low_cut=low_cut,
                        high_cut=high_cut,
                    )
                    chain_log.append({"op": "bandpass_filter", "low_cut": round(low_cut, 1), "high_cut": round(high_cut, 1)})

            # Finalize duration & run 10-point QC
            duration = float(len(current_audio)) / float(self.sample_rate)
            qc = run_10_point_qc(current_audio, sample_rate=self.sample_rate)
            latency = time.perf_counter() - t0

            results.append(
                AugmentedAudioSample(
                    aug_id=f"aug_{aug_idx:02d}",
                    aug_index=aug_idx,
                    audio_16k=current_audio,
                    sample_rate=self.sample_rate,
                    duration_sec=duration,
                    transform_chain=chain_log,
                    qc_result=qc,
                    generation_latency_sec=latency,
                    seed=aug_seed,
                )
            )

        return results

    def generate_exact_plan(
        self,
        audio_16k: np.ndarray,
        plan_ops: List[Dict[str, Any]],
        aug_index: int = 1,
        seed: int = 42,
    ) -> AugmentedAudioSample:
        """Execute an exact user-specified pipeline of transforms."""
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        current_audio = audio_16k.copy()
        chain_log: List[Dict[str, Any]] = []

        for item in plan_ops:
            op = item.get("op")
            if op == "gain":
                gain_db = float(item.get("gain_db", 0.0))
                current_audio = apply_gain(current_audio, gain_db=gain_db)
                chain_log.append({"op": "gain", "gain_db": gain_db})

            elif op == "additive_noise":
                snr_db = float(item.get("snr_db", 20.0))
                current_audio = apply_additive_noise(current_audio, snr_db=snr_db, rng=rng)
                chain_log.append({"op": "additive_noise", "snr_db": snr_db})

            elif op == "time_shift":
                shift_sec = float(item.get("shift_sec", 0.1))
                current_audio = apply_time_shift(current_audio, shift_sec=shift_sec, sample_rate=self.sample_rate)
                chain_log.append({"op": "time_shift", "shift_sec": shift_sec})

            elif op == "time_stretch":
                rate = float(item.get("rate", 1.0))
                current_audio = apply_time_stretch(current_audio, rate=rate)
                chain_log.append({"op": "time_stretch", "rate": rate})

            elif op == "pitch_shift":
                n_steps = float(item.get("n_steps", 0.0))
                current_audio = apply_pitch_shift(current_audio, n_steps=n_steps, sample_rate=self.sample_rate)
                chain_log.append({"op": "pitch_shift", "n_steps": n_steps})

            elif op == "synthetic_reverb":
                decay = float(item.get("decay_sec", 0.2))
                wet = float(item.get("wet_mix", 0.2))
                current_audio = apply_synthetic_reverb(
                    current_audio,
                    sample_rate=self.sample_rate,
                    decay_time_sec=decay,
                    wet_mix=wet,
                    rng=rng,
                )
                chain_log.append({"op": "synthetic_reverb", "decay_sec": decay, "wet_mix": wet})

            elif op == "bandpass_filter":
                low_cut = float(item.get("low_cut", 300.0))
                high_cut = float(item.get("high_cut", 3400.0))
                current_audio = apply_frequency_filter(
                    current_audio,
                    sample_rate=self.sample_rate,
                    filter_type="bandpass",
                    low_cut=low_cut,
                    high_cut=high_cut,
                )
                chain_log.append({"op": "bandpass_filter", "low_cut": low_cut, "high_cut": high_cut})

        duration = float(len(current_audio)) / float(self.sample_rate)
        qc = run_10_point_qc(current_audio, sample_rate=self.sample_rate)
        latency = time.perf_counter() - t0

        return AugmentedAudioSample(
            aug_id=f"plan_{aug_index:02d}",
            aug_index=aug_index,
            audio_16k=current_audio,
            sample_rate=self.sample_rate,
            duration_sec=duration,
            transform_chain=chain_log,
            qc_result=qc,
            generation_latency_sec=latency,
            seed=seed,
        )
