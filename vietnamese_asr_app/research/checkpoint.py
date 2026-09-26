"""Checkpoint and Experiment State Management for LoRA Models.

Provides:
- Adapter-only weight serialization (via PeftModel.save_pretrained)
- Trainer state persistence (optimizer, scheduler, epoch, step, full RNG states)
- Comprehensive machine-readable experiment_metadata.json
- Checkpoint restoration and state resume verification
"""

import json
import os
import random
import subprocess
import time
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from peft import PeftModel

try:
    from asr.model import detect_hardware
except (ImportError, ValueError):
    from ..asr.model import detect_hardware


def get_git_commit_hash() -> Optional[str]:
    """Retrieve current git commit hash if in a git repository."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out
    except Exception:
        return None


import hashlib
import sys
import accelerate
import peft
import transformers

def compute_file_sha256(path: str) -> Optional[str]:
    """Compute SHA-256 hash of a file if it exists."""
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return None
    return None

def save_lora_checkpoint(
    checkpoint_dir: str,
    peft_model: PeftModel,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    global_step: int = 0,
    base_model_id: str = "vinai/PhoWhisper-tiny",
    experiment_config: Optional[Dict[str, Any]] = None,
    parameter_stats: Optional[Any] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Save LoRA adapter weights, trainer state, and experiment metadata.

    Args:
        checkpoint_dir: Target directory path.
        peft_model: Instantiated PeftModel.
        optimizer: Optional optimizer with state.
        scheduler: Optional learning rate scheduler.
        epoch: Current training epoch index.
        global_step: Cumulative optimizer step count.
        base_model_id: Identifier of the frozen base model.
        experiment_config: Complete experiment configuration dictionary.
        parameter_stats: TrainableParameterStats instance.
        metrics: Optional metrics dictionary (loss, WER, etc.).

    Returns:
        Dictionary of created file paths.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    adapter_dir = os.path.join(checkpoint_dir, "adapter_model")
    os.makedirs(adapter_dir, exist_ok=True)

    # 1. Save lightweight PEFT adapter weights and adapter_config.json
    peft_model.save_pretrained(adapter_dir)
    peft_model.save_pretrained(checkpoint_dir)

    # 2. Capture complete RNG states
    rng_state = {
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

    # 3. Save trainer state (optimizer, scheduler, steps, RNG)
    trainer_state = {
        "epoch": epoch,
        "global_step": global_step,
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "rng_state": rng_state,
        "timestamp": time.time(),
    }
    trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.pt")
    torch.save(trainer_state, trainer_state_path)

    # 4. Extract measured CUDA memory statistics
    if torch.cuda.is_available():
        peak_alloc_bytes = int(torch.cuda.max_memory_allocated(0))
        peak_res_bytes = int(torch.cuda.max_memory_reserved(0))
        peak_alloc_mb = float(peak_alloc_bytes / (1024 * 1024))
        peak_res_mb = float(peak_res_bytes / (1024 * 1024))
        cuda_memory = {
            "peak_allocated_bytes": peak_alloc_bytes,
            "peak_reserved_bytes": peak_res_bytes,
            "peak_allocated_mb": round(peak_alloc_mb, 2),
            "peak_reserved_mb": round(peak_res_mb, 2),
            "measurement_source": "cuda_runtime_measured",
        }
    else:
        cuda_memory = {
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
            "peak_allocated_mb": None,
            "peak_reserved_mb": None,
            "measurement_source": "cpu_runtime_cuda_unavailable",
        }

    # 5. Extract manifest hashes
    manifest_hashes = {}
    if experiment_config and "data" in experiment_config:
        d_cfg = experiment_config["data"]
        for key in ["train_manifest", "val_manifest", "test_manifest", "synthetic_manifest"]:
            mpath = d_cfg.get(key)
            if mpath:
                manifest_hashes[key] = {
                    "path": mpath,
                    "sha256": compute_file_sha256(mpath),
                }

    # 6. Save comprehensive machine-readable experiment metadata
    hw = detect_hardware()
    meta = {
        "experiment_id": experiment_config.get("experiment_id", "unnamed") if experiment_config else "unnamed",
        "base_model_id": base_model_id,
        "git_commit": get_git_commit_hash(),
        "timestamp_utc": time.asctime(),
        "epoch": epoch,
        "global_step": global_step,
        "software_versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "accelerate": accelerate.__version__,
        },
        "hardware": {
            "device": hw.device,
            "cuda_available": hw.cuda_available,
            "gpu_name": hw.gpu_name,
            "vram_total_gb": hw.vram_total_gb,
            "torch_version": hw.torch_version,
            "recommended_precision": hw.recommended_precision,
        },
        "cuda_memory_statistics": cuda_memory,
        "manifest_hashes": manifest_hashes,
        "optimizer_info": {
            "optimizer_class": optimizer.__class__.__name__ if optimizer else None,
            "learning_rate": optimizer.param_groups[0]["lr"] if optimizer else None,
            "weight_decay": optimizer.param_groups[0].get("weight_decay") if optimizer else None,
            "scheduler_class": scheduler.__class__.__name__ if scheduler else None,
        },
        "trainable_parameters": {
            "total_params": parameter_stats.total_params if parameter_stats else None,
            "trainable_params": parameter_stats.trainable_params if parameter_stats else None,
            "frozen_params": parameter_stats.frozen_params if parameter_stats else None,
            "trainable_ratio_pct": parameter_stats.trainable_ratio_pct if parameter_stats else None,
            "target_modules": list(getattr(peft_model.peft_config.get("default", None), "target_modules", [])) if getattr(peft_model.peft_config.get("default", None), "target_modules", None) is not None else None,
            "r": getattr(peft_model.peft_config.get("default", None), "r", None),
            "lora_alpha": getattr(peft_model.peft_config.get("default", None), "lora_alpha", None),
            "lora_dropout": getattr(peft_model.peft_config.get("default", None), "lora_dropout", None),
        },
        "experiment_config": experiment_config or {},
        "metrics": metrics or {},
    }
    metadata_path = os.path.join(checkpoint_dir, "experiment_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return {
        "checkpoint_dir": checkpoint_dir,
        "adapter_dir": adapter_dir,
        "trainer_state_path": trainer_state_path,
        "metadata_path": metadata_path,
    }


def load_lora_checkpoint(
    checkpoint_dir: str,
    base_model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    is_trainable: bool = False,
) -> Tuple[PeftModel, Dict[str, Any]]:
    """Load adapter onto base model and optionally restore trainer state.

    Args:
        checkpoint_dir: Checkpoint directory path.
        base_model: Instantiated base WhisperForConditionalGeneration model.
        optimizer: Optional optimizer to restore state into.
        scheduler: Optional scheduler to restore state into.
        is_trainable: Whether the attached adapter should be trainable (default False for eval).

    Returns:
        (peft_model, trainer_state_dict)
    """
    adapter_dir = os.path.join(checkpoint_dir, "adapter_model")
    if not os.path.exists(adapter_dir):
        # Fallback if checkpoint_dir itself contains adapter config
        adapter_dir = checkpoint_dir

    peft_model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
        is_trainable=is_trainable,
    )

    trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.pt")
    trainer_state = {}
    if os.path.exists(trainer_state_path):
        trainer_state = torch.load(trainer_state_path, map_location="cpu", weights_only=False)

        if optimizer and trainer_state.get("optimizer_state_dict"):
            optimizer.load_state_dict(trainer_state["optimizer_state_dict"])

        if scheduler and trainer_state.get("scheduler_state_dict"):
            scheduler.load_state_dict(trainer_state["scheduler_state_dict"])

        if "rng_state" in trainer_state:
            rng = trainer_state["rng_state"]
            random.setstate(rng["python_rng"])
            np.random.set_state(rng["numpy_rng"])
            torch.set_rng_state(rng["torch_rng"])
            if torch.cuda.is_available() and rng.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all(rng["cuda_rng"])

    return peft_model, trainer_state
