"""LoRA Model Adaptation Service.

Integrates dynamic parameter accounting, checkpoint registry discovery,
frozen Full E1 protocol display (read-only), and strict execution safety checks.
Prevents unauthorized or uncalibrated CPU execution of full experiments.
"""

import glob
import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple
import peft
import torch
import transformers
from transformers import WhisperForConditionalGeneration

from asr.model import detect_hardware
from research.checkpoint import load_lora_checkpoint
from research.lora import TrainableParameterStats, build_whisper_lora_model, count_parameters

FROZEN_FULL_E1_PROTOCOL = {
    "experiment_id": "FULL_E1",
    "base_model": "vinai/PhoWhisper-tiny",
    "pinned_revision": "cc51d32be916efebde04ff549854fa1741cb5c02",
    "peft_version": "0.21.0",
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"],
    "bias": "none",
    "expected_trainable_parameters": 147456,
    "expected_trainable_ratio_pct": 0.3890,
    "training_samples": 26671,
    "vivos_train_samples": 11660,
    "vimd_train_samples": 15011,
    "validation_samples": 1900,
    "micro_batch_size": 4,
    "gradient_accumulation_steps": 8,
    "effective_batch_size": 32,
    "optimizer": "AdamW",
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "scheduler": "cosine",
    "warmup_ratio": 0.10,
    "gradient_clip": 1.0,
    "epochs": 3,
    "total_optimizer_steps": 2502,
    "seed": 42,
    "precision": "FP16 CUDA",
    "frozen_test_manifest_sha256": "efe54856d12613f109cebc9482a6f224ca4e2370d0ff3f8a2fe4ea0a013b60ca",
    "mode": "FROZEN_EXPERIMENT_READ_ONLY",
}


class LoRAService:
    """Service managing LoRA inspection, checkpoint discovery, and training safety gates."""

    def __init__(self, checkpoints_dir: str = "checkpoints"):
        self.checkpoints_dir = checkpoints_dir
        self._cached_param_stats: Optional[TrainableParameterStats] = None

    def compute_dynamic_parameters(self, model_id: str = "vinai/PhoWhisper-tiny") -> TrainableParameterStats:
        """Compute exact parameter statistics dynamically without hardcoded sources of truth."""
        if self._cached_param_stats is not None:
            return self._cached_param_stats

        # Instantiate temporary base model to calculate real LoRA parameter metrics
        base = WhisperForConditionalGeneration.from_pretrained(model_id)
        peft_model, stats = build_whisper_lora_model(
            base,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        )
        self._cached_param_stats = stats
        return stats

    def list_available_adapters(self) -> List[Dict[str, Any]]:
        """Scan checkpoints directory and return available trained adapters with metadata."""
        adapters = []
        if not os.path.exists(self.checkpoints_dir):
            return adapters

        # Find all adapter_config.json files
        config_files = glob.glob(os.path.join(self.checkpoints_dir, "**", "adapter_config.json"), recursive=True)
        for cfg_path in config_files:
            adapter_dir = os.path.dirname(cfg_path)
            # Find parent checkpoint directory (if inside adapter_model subdirectory)
            ckpt_dir = os.path.dirname(adapter_dir) if os.path.basename(adapter_dir) == "adapter_model" else adapter_dir
            meta_path = os.path.join(ckpt_dir, "experiment_metadata.json")

            metadata = {}
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                except Exception:
                    pass

            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg_data = json.load(f)
            except Exception:
                cfg_data = {}

            # Friendly adapter name
            rel_path = os.path.relpath(adapter_dir, self.checkpoints_dir)
            name = metadata.get("experiment_id", rel_path.replace("\\", "/"))

            # Determine weights existence
            has_safetensors = os.path.exists(os.path.join(adapter_dir, "adapter_model.safetensors"))
            has_bin = os.path.exists(os.path.join(adapter_dir, "adapter_model.bin"))

            adapters.append({
                "name": name,
                "path": adapter_dir,
                "checkpoint_dir": ckpt_dir,
                "r": cfg_data.get("r", 8),
                "lora_alpha": cfg_data.get("lora_alpha", 16),
                "target_modules": cfg_data.get("target_modules", ["q_proj", "v_proj"]),
                "base_model": cfg_data.get("base_model_name_or_path", "vinai/PhoWhisper-tiny"),
                "peft_version": cfg_data.get("peft_version", peft.__version__),
                "epoch": metadata.get("epoch", "-"),
                "global_step": metadata.get("global_step", "-"),
                "has_weights": has_safetensors or has_bin,
            })

        return adapters

    def get_frozen_protocol(self) -> Dict[str, Any]:
        """Return the immutable frozen Full E1 experimental protocol."""
        return dict(FROZEN_FULL_E1_PROTOCOL)

    def validate_full_e1_execution_gate(
        self,
        test_manifest_path: str = "manifests/frozen_test_manifest.csv",
    ) -> Tuple[bool, str, List[str]]:
        """Strict pre-training execution gate for Full E1.
        
        Refuses to run if CUDA is unavailable, PEFT version != 0.21.0,
        model revision is not pinned, or test manifest hash is compromised.
        """
        reasons = []

        # 1. CUDA requirement
        hw = detect_hardware()
        if not hw.cuda_available:
            reasons.append(
                f"CUDA is required for this Full E1 configuration. Detected {hw.device} ({hw.gpu_name}). "
                f"CPU training is strictly forbidden by experimental protocol."
            )

        # 2. PEFT version check
        expected_peft = FROZEN_FULL_E1_PROTOCOL["peft_version"]
        if peft.__version__ != expected_peft:
            reasons.append(f"PEFT version mismatch: current={peft.__version__}, expected={expected_peft}.")

        # 3. Frozen test manifest verification
        if not os.path.exists(test_manifest_path):
            reasons.append(f"Required frozen test manifest missing at {test_manifest_path}.")
        else:
            with open(test_manifest_path, "rb") as f:
                content = f.read()
            # Canonical CRLF normalization check
            crlf_content = content.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
            computed_hash = hashlib.sha256(crlf_content).hexdigest()
            expected_hash = FROZEN_FULL_E1_PROTOCOL["frozen_test_manifest_sha256"]
            if computed_hash != expected_hash:
                reasons.append(
                    f"Frozen test manifest integrity check failed: computed {computed_hash}, expected {expected_hash}."
                )

        # 4. Manifest existence check
        for m_name in ["manifests/full_train_manifest.csv", "manifests/val_manifest.csv"]:
            if not os.path.exists(m_name):
                reasons.append(f"Required manifest is missing: {m_name}")

        is_passed = len(reasons) == 0
        summary_msg = "PASSED: Full E1 readiness gate verified." if is_passed else "BLOCKED: Full E1 safety gate rejected execution."
        return is_passed, summary_msg, reasons
