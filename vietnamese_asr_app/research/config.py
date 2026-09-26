"""Experiment Configuration Schema for Vietnamese ASR Research.

Defines typed configurations for:
- E0: Pretrained baseline (zero training)
- E1: LoRA on real labeled speech
- E2: LoRA on real + synthetic speech
"""

from dataclasses import asdict, dataclass, field
import json
import os
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class LoRAHyperparameters:
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"


@dataclass
class TrainingHyperparameters:
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    epochs: int = 3
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    mixed_precision: str = "fp16"  # "fp16", "bf16", or "fp32"


@dataclass
class DataPilotConfig:
    train_manifest: str = "manifests/train_manifest.csv"
    val_manifest: Optional[str] = "manifests/val_manifest.csv"
    test_manifest: str = "manifests/test_manifest.csv"
    max_train_samples: Optional[int] = 100
    max_val_samples: Optional[int] = 20
    max_test_samples: Optional[int] = 50
    quarantine_manifests: List[str] = field(default_factory=list)
    synthetic_manifest: Optional[str] = None
    synthetic_ratio: float = 0.0  # 0.0 for E1, >0.0 for E2


@dataclass
class TrainingBudgetConfig:
    total_train_samples: int = 22
    real_sample_count: int = 22
    synthetic_sample_count: int = 0
    real_ratio: float = 1.0
    synthetic_ratio: float = 0.0
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    effective_batch_size: int = 2
    fixed_optimizer_steps_per_epoch: int = 11
    total_optimizer_steps: int = 33
    sample_exposures_per_epoch: int = 22
    total_sample_exposures: int = 66


@dataclass
class ExperimentConfig:
    experiment_id: str
    base_model_id: str = "vinai/PhoWhisper-tiny"
    mode: str = "train_and_eval"  # "eval_only" for E0, "train_and_eval" for E1/E2
    seed: int = 42
    output_dir: str = "checkpoints/lora_pilot"
    lora: LoRAHyperparameters = field(default_factory=LoRAHyperparameters)
    training: TrainingHyperparameters = field(default_factory=TrainingHyperparameters)
    data: DataPilotConfig = field(default_factory=DataPilotConfig)
    training_budget: TrainingBudgetConfig = field(default_factory=TrainingBudgetConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_yaml(self, file_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def load_yaml(cls, file_path: str, experiment_id: Optional[str] = None) -> "ExperimentConfig":
        """Load experiment configuration from YAML with explicit multi-experiment preset resolution."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)

        if not isinstance(d, dict):
            raise ValueError(f"Invalid YAML structure in {file_path}: root must be a dictionary.")

        # Detect multi-experiment structure (e.g., {"E0": {...}, "E1": {...}, "E2": {...}})
        is_multi_exp = any(k in d for k in ["E0", "E1", "E2"]) or (
            all(isinstance(v, dict) and "experiment_id" in v for v in d.values()) and len(d) > 1
        )

        if is_multi_exp:
            if not experiment_id:
                raise ValueError(
                    f"Configuration file '{file_path}' contains multiple experiments ({list(d.keys())}). "
                    f"You must specify an experiment_id (e.g. 'E1')."
                )
            if experiment_id not in d:
                raise KeyError(
                    f"Experiment '{experiment_id}' not found in configuration '{file_path}'. "
                    f"Available experiments: {list(d.keys())}"
                )
            d_exp = d[experiment_id]
        else:
            d_exp = d

        # Fail loudly if required sections are missing - do NOT silently use defaults
        mode = d_exp.get("mode", "train_and_eval")

        if "lora" not in d_exp:
            raise ValueError(
                f"Missing required 'lora' section in experiment configuration for '{experiment_id or d_exp.get('experiment_id')}'."
            )
        if mode != "eval_only" and "training" not in d_exp:
            raise ValueError(
                f"Missing required 'training' section in experiment configuration for '{experiment_id or d_exp.get('experiment_id')}'."
            )
        if "data" not in d_exp:
            raise ValueError(
                f"Missing required 'data' section in experiment configuration for '{experiment_id or d_exp.get('experiment_id')}'."
            )

        lora_cfg = LoRAHyperparameters(**d_exp["lora"])
        training_cfg = TrainingHyperparameters(**d_exp["training"]) if "training" in d_exp else TrainingHyperparameters()
        data_cfg = DataPilotConfig(**d_exp["data"])
        budget_cfg = TrainingBudgetConfig(**d_exp["training_budget"]) if "training_budget" in d_exp else TrainingBudgetConfig()

        return cls(
            experiment_id=d_exp.get("experiment_id", experiment_id or "unnamed"),
            base_model_id=d_exp.get("base_model_id", "vinai/PhoWhisper-tiny"),
            mode=mode,
            seed=int(d_exp.get("seed", 42)),
            output_dir=d_exp.get("output_dir", "checkpoints/lora_pilot"),
            lora=lora_cfg,
            training=training_cfg,
            data=data_cfg,
            training_budget=budget_cfg,
        )


def build_controlled_experiment_configs(base_dir: str = "configs") -> Dict[str, ExperimentConfig]:
    """Build controlled experiment configurations for E0, E1, and E2 guaranteeing invariance."""
    # E0: Pretrained baseline (zero training)
    e0 = ExperimentConfig(
        experiment_id="E0_baseline_tiny",
        base_model_id="vinai/PhoWhisper-tiny",
        mode="eval_only",
        data=DataPilotConfig(
            test_manifest="manifests/test_manifest.csv",
            max_test_samples=50,
        ),
        training_budget=TrainingBudgetConfig(
            total_train_samples=0,
            real_sample_count=0,
            synthetic_sample_count=0,
            real_ratio=0.0,
            synthetic_ratio=0.0,
            fixed_optimizer_steps_per_epoch=0,
            total_optimizer_steps=0,
        ),
    )

    # Common LoRA and training hyperparameters shared across E1 and E2
    shared_lora = LoRAHyperparameters(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    shared_training = TrainingHyperparameters(
        batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        epochs=3,
        lr_scheduler_type="cosine",
        mixed_precision="fp16",
    )

    # E1: LoRA on real labeled speech only
    e1 = ExperimentConfig(
        experiment_id="E1_lora_real_tiny",
        base_model_id="vinai/PhoWhisper-tiny",
        mode="train_and_eval",
        seed=42,
        output_dir="checkpoints/E1_lora_real",
        lora=shared_lora,
        training=shared_training,
        data=DataPilotConfig(
            train_manifest="manifests/train_manifest.csv",
            val_manifest="manifests/val_manifest.csv",
            test_manifest="manifests/test_manifest.csv",
            max_train_samples=22,
            max_val_samples=5,
            max_test_samples=9,
            quarantine_manifests=["manifests/test_manifest.csv"],
            synthetic_manifest=None,
            synthetic_ratio=0.0,
        ),
        training_budget=TrainingBudgetConfig(
            total_train_samples=22,
            real_sample_count=22,
            synthetic_sample_count=0,
            real_ratio=1.0,
            synthetic_ratio=0.0,
            micro_batch_size=2,
            gradient_accumulation_steps=1,
            effective_batch_size=2,
            fixed_optimizer_steps_per_epoch=11,
            total_optimizer_steps=33,
            sample_exposures_per_epoch=22,
            total_sample_exposures=66,
        ),
    )

    # E2: LoRA on real + synthetic speech (identically matched hyperparameters & budget)
    e2 = ExperimentConfig(
        experiment_id="E2_lora_real_synth_tiny",
        base_model_id="vinai/PhoWhisper-tiny",
        mode="train_and_eval",
        seed=42,  # Exactly matching seed
        output_dir="checkpoints/E2_lora_real_synth",
        lora=shared_lora,  # Exactly matching LoRA config
        training=shared_training,  # Exactly matching training config
        data=DataPilotConfig(
            train_manifest="manifests/train_manifest.csv",
            val_manifest="manifests/val_manifest.csv",
            test_manifest="manifests/test_manifest.csv",  # Exactly matching test set
            max_train_samples=22,
            max_val_samples=5,
            max_test_samples=9,
            quarantine_manifests=["manifests/test_manifest.csv"],
            synthetic_manifest="manifests/synthetic_qc_passed.csv",
            synthetic_ratio=0.3182,  # 7 / 22 = 31.82% synthetic data addition
        ),
        training_budget=TrainingBudgetConfig(
            total_train_samples=22,
            real_sample_count=15,
            synthetic_sample_count=7,
            real_ratio=0.6818,
            synthetic_ratio=0.3182,
            micro_batch_size=2,
            gradient_accumulation_steps=1,
            effective_batch_size=2,
            fixed_optimizer_steps_per_epoch=11,
            total_optimizer_steps=33,
            sample_exposures_per_epoch=22,
            total_sample_exposures=66,
        ),
    )

    return {"E0": e0, "E1": e1, "E2": e2}
