"""LoRA Module for PhoWhisper / Whisper Models.

Provides:
- Dynamic target module inspection
- PEFT LoRA model creation
- Parameter accounting (trainable vs. frozen)
- Gradient and freeze integrity verification
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration
from peft import LoraConfig, PeftModel, get_peft_model


@dataclass
class TrainableParameterStats:
    total_params: int
    trainable_params: int
    frozen_params: int
    trainable_ratio_pct: float
    target_modules_matched: List[str]

    def summary(self) -> str:
        return (
            f"Trainable Parameters: {self.trainable_params:,} / {self.total_params:,} "
            f"({self.trainable_ratio_pct:.4f}%) | "
            f"Target Modules: {len(self.target_modules_matched)} matched"
        )


def inspect_model_modules(model: nn.Module) -> Dict[str, List[str]]:
    """Inspect model module hierarchy to identify projection layers.

    Returns:
        Dictionary mapping projection types ('q_proj', 'v_proj', 'k_proj', 'out_proj', 'fc1', 'fc2')
        to their full module names.
    """
    projections: Dict[str, List[str]] = {
        "q_proj": [],
        "v_proj": [],
        "k_proj": [],
        "out_proj": [],
        "fc1": [],
        "fc2": [],
    }
    for name, module in model.named_modules():
        for proj_key in projections:
            if name.endswith(proj_key):
                projections[proj_key].append(name)
    return projections


def count_parameters(model: nn.Module, target_modules: Optional[List[str]] = None) -> TrainableParameterStats:
    """Calculate exact parameter statistics without hardcoding.

    Args:
        model: PyTorch model or PeftModel.
        target_modules: Optional list of target module patterns.

    Returns:
        TrainableParameterStats instance.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    frozen = total - trainable
    ratio = (100.0 * trainable / total) if total > 0 else 0.0

    matched = []
    if target_modules:
        for name, _ in model.named_modules():
            if any(name.endswith(tm) for tm in target_modules):
                matched.append(name)

    return TrainableParameterStats(
        total_params=total,
        trainable_params=trainable,
        frozen_params=frozen,
        trainable_ratio_pct=round(ratio, 4),
        target_modules_matched=matched,
    )


def verify_freeze_integrity(peft_model: nn.Module) -> Tuple[bool, str]:
    """Verify that all base model parameters are frozen and only LoRA parameters have requires_grad=True.

    Returns:
        (is_valid, message)
    """
    unfrozen_base = []
    frozen_lora = []

    for name, param in peft_model.named_parameters():
        is_lora = "lora_" in name
        if is_lora:
            if not param.requires_grad:
                frozen_lora.append(name)
        else:
            if param.requires_grad:
                unfrozen_base.append(name)

    if unfrozen_base:
        return False, f"Found {len(unfrozen_base)} non-LoRA base parameters with requires_grad=True (e.g. {unfrozen_base[:3]})"
    if frozen_lora:
        return False, f"Found {len(frozen_lora)} LoRA parameters with requires_grad=False (e.g. {frozen_lora[:3]})"

    return True, "Freeze integrity verified: base parameters strictly frozen, LoRA parameters trainable."


def build_whisper_lora_model(
    base_model: WhisperForConditionalGeneration,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    bias: str = "none",
) -> Tuple[PeftModel, TrainableParameterStats]:
    """Attach LoRA adapters to PhoWhisper base model.

    Args:
        base_model: Instantiated WhisperForConditionalGeneration model.
        r: LoRA attention dimension.
        lora_alpha: LoRA alpha scaling factor.
        lora_dropout: LoRA dropout probability.
        target_modules: Modules to adapt (default: ['q_proj', 'v_proj']).
        bias: Bias type for LoRA ('none', 'all', 'lora_only').

    Returns:
        (peft_model, parameter_stats)
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    # Verify that target modules actually exist in the model
    module_inventory = inspect_model_modules(base_model)
    for tm in target_modules:
        if tm in module_inventory and len(module_inventory[tm]) == 0:
            raise ValueError(
                f"Target module '{tm}' was specified but does not exist in model. "
                f"Available projections: {[k for k, v in module_inventory.items() if len(v) > 0]}"
            )

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
    )

    peft_model = get_peft_model(base_model, lora_config)

    # Verify freeze integrity immediately
    is_valid, msg = verify_freeze_integrity(peft_model)
    if not is_valid:
        raise RuntimeError(f"LoRA initialization failed freeze integrity check: {msg}")

    stats = count_parameters(peft_model, target_modules=target_modules)
    return peft_model, stats
