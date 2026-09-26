"""Test Suite for Phase 2 Research Layer: PEFT LoRA, Collator, Quarantine, and Checkpointing.

Executes all 12 verification stages mandated by the controlled experimental protocol:
1. CPU smoke test
2. Tiny dummy dataset creation
3. Load PhoWhisper-tiny
4. Attach LoRA
5. Verify trainable parameter accounting
6. Verify base parameters are frozen (requires_grad=False)
7. Run forward pass with loss
8. Run backward pass & verify gradients
9. Run optimizer step (AdamW)
10. Save adapter checkpoint
11. Reload adapter checkpoint
12. Verify inference still works
"""

import os
import shutil
import tempfile
import numpy as np
import pytest
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

from research.checkpoint import load_lora_checkpoint, save_lora_checkpoint
from research.collator import WhisperDataCollatorWithPadding
from research.config import ExperimentConfig, build_controlled_experiment_configs
from research.evaluator import evaluate_model_on_manifest
from research.lora import build_whisper_lora_model, count_parameters, verify_freeze_integrity
from research.quarantine import DataContaminationError, QuarantineEnforcer, hash_normalized_text


@pytest.fixture(scope="module")
def pho_tiny_artifacts():
    """Load PhoWhisper-tiny model and processor once for the test module."""
    model_id = "vinai/PhoWhisper-tiny"
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    return model, processor, model_id


def test_lora_attachment_and_freeze_integrity(pho_tiny_artifacts):
    """Stages 3, 4, 5, 6: Attach LoRA, verify parameter counts, verify freeze integrity."""
    model, processor, model_id = pho_tiny_artifacts

    # Build fresh copy for LoRA
    base_model = WhisperForConditionalGeneration.from_pretrained(model_id)
    peft_model, stats = build_whisper_lora_model(
        base_model,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )

    # 1. Verify trainable parameter accounting
    assert stats.total_params == 37_908_096
    assert stats.trainable_params == 147_456
    assert abs(stats.trainable_ratio_pct - 0.3890) < 1e-3
    assert len(stats.target_modules_matched) == 24  # 12 q_proj + 12 v_proj

    # 2. Verify freeze integrity
    is_valid, msg = verify_freeze_integrity(peft_model)
    assert is_valid is True, msg

    # Double check explicit parameter attributes
    for name, param in peft_model.named_parameters():
        if "lora_" in name:
            assert param.requires_grad is True, f"LoRA param {name} is unexpectedly frozen"
        else:
            assert param.requires_grad is False, f"Base param {name} is unexpectedly trainable"


def test_cpu_smoke_test_forward_backward_optimizer_step(pho_tiny_artifacts):
    """Stages 1, 2, 7, 8, 9: Forward, backward, gradient flow, and AdamW optimizer step on CPU."""
    _, processor, model_id = pho_tiny_artifacts
    base_model = WhisperForConditionalGeneration.from_pretrained(model_id)
    peft_model, _ = build_whisper_lora_model(base_model, r=8, lora_alpha=16)

    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=1e-4)

    # Stage 2: Create a tiny dummy dataset (1 dummy audio, 1 text transcript)
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    dummy_audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    inputs = processor(dummy_audio, sampling_rate=sr, return_tensors="pt")
    input_features = inputs.input_features

    labels = processor.tokenizer("xin chào bạn", return_tensors="pt").input_ids

    # Stage 7: Forward pass
    outputs = peft_model(input_features=input_features, labels=labels)
    loss = outputs.loss
    assert loss is not None
    assert torch.isfinite(loss).item()
    initial_loss_val = float(loss.detach().item())
    assert initial_loss_val > 0.0

    # Stage 8: Backward pass
    loss.backward()

    # Verify gradients: ONLY LoRA parameters should have gradients
    lora_grad_count = 0
    base_grad_count = 0
    for name, param in peft_model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"LoRA param {name} did not receive gradients"
            lora_grad_count += 1
        else:
            assert param.grad is None, f"Frozen param {name} received non-null gradients"
            base_grad_count += 1

    assert lora_grad_count > 0

    # Stage 9: Optimizer step
    optimizer.step()
    optimizer.zero_grad()


def test_adapter_checkpoint_save_reload_and_inference(pho_tiny_artifacts):
    """Stages 10, 11, 12: Save adapter checkpoint, reload onto fresh base model, verify inference."""
    _, processor, model_id = pho_tiny_artifacts
    base_model = WhisperForConditionalGeneration.from_pretrained(model_id)
    peft_model, stats = build_whisper_lora_model(base_model, r=8, lora_alpha=16)

    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=1e-4)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Stage 10: Save adapter checkpoint
        res = save_lora_checkpoint(
            checkpoint_dir=tmp_dir,
            peft_model=peft_model,
            optimizer=optimizer,
            epoch=1,
            global_step=10,
            base_model_id=model_id,
            parameter_stats=stats,
        )

        assert os.path.exists(res["adapter_dir"])
        assert os.path.exists(res["trainer_state_path"])
        assert os.path.exists(res["metadata_path"])

        # Stage 11: Reload adapter checkpoint onto a clean fresh base model
        clean_base = WhisperForConditionalGeneration.from_pretrained(model_id)
        reloaded_peft, trainer_state = load_lora_checkpoint(
            checkpoint_dir=tmp_dir,
            base_model=clean_base,
            is_trainable=True,
        )

        assert trainer_state["epoch"] == 1
        assert trainer_state["global_step"] == 10
        assert isinstance(reloaded_peft, PeftModel)

        # Verify optimizer state restoration on reloaded trainable model
        restored_opt = torch.optim.AdamW(reloaded_peft.parameters(), lr=1e-4)
        if trainer_state.get("optimizer_state_dict"):
            restored_opt.load_state_dict(trainer_state["optimizer_state_dict"])

        # Stage 12: Verify inference still works
        dummy_audio = 0.2 * np.sin(2 * np.pi * 300 * np.linspace(0, 1.0, 16000, dtype=np.float32))
        inputs = processor(dummy_audio, sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            pred_ids = reloaded_peft.generate(
                inputs.input_features,
                language="vi",
                task="transcribe",
                max_new_tokens=32,
            )
        assert pred_ids is not None
        assert pred_ids.shape[-1] > 0


def test_collator_padding_and_label_masking(pho_tiny_artifacts):
    """Verify WhisperDataCollatorWithPadding replaces padding tokens with -100."""
    _, processor, _ = pho_tiny_artifacts
    collator = WhisperDataCollatorWithPadding(processor=processor)

    # 2 samples of different lengths
    feature_1 = np.ones((80, 100), dtype=np.float32)
    feature_2 = np.ones((80, 200), dtype=np.float32)

    labels_1 = [50258, 200, 300]  # length 3
    labels_2 = [50258, 200, 300, 400, 500]  # length 5

    batch = collator([
        {"input_features": feature_1, "labels": labels_1},
        {"input_features": feature_2, "labels": labels_2},
    ])

    assert "input_features" in batch
    assert "labels" in batch
    labels = batch["labels"]
    assert labels.shape[0] == 2
    assert labels.shape[1] == 4  # Stripped leading 50258, padded to length 4

    # Padding tokens in the shorter sequence must be masked to -100
    assert (labels[0, 2:] == -100).all()
    # Longer sequence has valid token IDs
    assert (labels[1] != -100).all()


def test_quarantine_enforcer_detects_leakage():
    """Verify QuarantineEnforcer detects both sample ID and normalized transcript hash overlaps."""
    enforcer = QuarantineEnforcer()

    # Register quarantined test items
    enforcer.register_quarantined_sample(
        sample_id="vivos_test_spk01_001",
        transcript="Hôm nay trời nắng đẹp",
    )

    # 1. Safe sample must pass
    is_safe, reason = enforcer.check_sample("train_spk02_001", "Tôi đi học bài")
    assert is_safe is True
    assert reason is None

    # 2. Quarantined sample ID must fail
    is_safe, reason = enforcer.check_sample("vivos_test_spk01_001", "Nội dung bất kỳ")
    assert is_safe is False
    assert "vivos_test_spk01_001" in reason

    # 3. Quarantined text hash must fail even with different casing/punctuation
    is_safe, reason = enforcer.check_sample("train_spk99_001", "Hôm nay, trời nắng đẹp!!!")
    assert is_safe is False
    assert "Normalized transcript matches quarantined evaluation text" in reason

    # 4. assert_safe must raise DataContaminationError
    with pytest.raises(DataContaminationError):
        enforcer.assert_safe("vivos_test_spk01_001", "Tôi đi học")


def test_experiment_config_invariance():
    """Verify controlled configurations for E0, E1, and E2 maintain strict variable invariance."""
    cfgs = build_controlled_experiment_configs()
    e0, e1, e2 = cfgs["E0"], cfgs["E1"], cfgs["E2"]

    # E0 is eval only
    assert e0.mode == "eval_only"

    # E1 and E2 must have EXACTLY matching hyperparameters
    assert e1.seed == e2.seed
    assert e1.base_model_id == e2.base_model_id
    assert e1.lora.r == e2.lora.r
    assert e1.lora.lora_alpha == e2.lora.lora_alpha
    assert e1.lora.target_modules == e2.lora.target_modules
    assert e1.training.learning_rate == e2.training.learning_rate
    assert e1.training.epochs == e2.training.epochs
    assert e1.training.batch_size == e2.training.batch_size
    assert e1.training.gradient_accumulation_steps == e2.training.gradient_accumulation_steps
    assert e1.data.test_manifest == e2.data.test_manifest

    # Fixed training budget invariance
    assert e1.training_budget.total_train_samples == e2.training_budget.total_train_samples
    assert e1.training_budget.fixed_optimizer_steps_per_epoch == e2.training_budget.fixed_optimizer_steps_per_epoch
    assert e1.training_budget.total_optimizer_steps == e2.training_budget.total_optimizer_steps

    # The ONLY intended difference is synthetic data composition
    assert e1.data.synthetic_ratio == 0.0
    assert abs(e2.data.synthetic_ratio - 0.3182) < 1e-3
    assert e1.training_budget.real_ratio == 1.0
    assert e1.training_budget.synthetic_ratio == 0.0
    assert abs(e2.training_budget.real_ratio - 0.6818) < 1e-3
    assert abs(e2.training_budget.synthetic_ratio - 0.3182) < 1e-3
    assert e1.data.synthetic_manifest is None
    assert e2.data.synthetic_manifest is not None


def test_yaml_multi_experiment_resolution():
    """Test loading and resolving E0, E1, E2 from configs/experiments.yaml."""
    yaml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "experiments.yaml"))
    assert os.path.exists(yaml_path), f"YAML config missing: {yaml_path}"

    # 1. Resolve E0
    cfg_e0 = ExperimentConfig.load_yaml(yaml_path, experiment_id="E0")
    assert cfg_e0.experiment_id == "E0_baseline_tiny"
    assert cfg_e0.mode == "eval_only"

    # 2. Resolve E1
    cfg_e1 = ExperimentConfig.load_yaml(yaml_path, experiment_id="E1")
    assert cfg_e1.experiment_id == "E1_lora_real_tiny"
    assert cfg_e1.base_model_id == "vinai/PhoWhisper-tiny"
    assert cfg_e1.lora.r == 8
    assert cfg_e1.lora.lora_alpha == 16
    assert cfg_e1.lora.target_modules == ["q_proj", "v_proj"]
    assert cfg_e1.training.epochs == 3
    assert cfg_e1.training.learning_rate == 1e-4
    assert cfg_e1.training.batch_size == 2
    assert cfg_e1.training.gradient_accumulation_steps == 1
    assert cfg_e1.training_budget.total_train_samples == 22
    assert cfg_e1.training_budget.real_sample_count == 22
    assert cfg_e1.training_budget.synthetic_sample_count == 0
    assert cfg_e1.training_budget.real_ratio == 1.0
    assert cfg_e1.training_budget.synthetic_ratio == 0.0
    assert cfg_e1.training_budget.fixed_optimizer_steps_per_epoch == 11
    assert cfg_e1.training_budget.total_optimizer_steps == 33
    assert cfg_e1.training_budget.sample_exposures_per_epoch == 22
    assert cfg_e1.training_budget.total_sample_exposures == 66

    # Mathematical consistency equation check for E1
    assert cfg_e1.training_budget.sample_exposures_per_epoch == (
        cfg_e1.training_budget.fixed_optimizer_steps_per_epoch
        * cfg_e1.training.batch_size
        * cfg_e1.training.gradient_accumulation_steps
    )

    # 3. Resolve E2
    cfg_e2 = ExperimentConfig.load_yaml(yaml_path, experiment_id="E2")
    assert cfg_e2.experiment_id == "E2_lora_real_synth_tiny"
    assert cfg_e2.training_budget.total_train_samples == 22
    assert cfg_e2.training_budget.real_sample_count == 15
    assert cfg_e2.training_budget.synthetic_sample_count == 7
    assert abs(cfg_e2.training_budget.real_ratio - 0.6818) < 1e-3
    assert abs(cfg_e2.training_budget.synthetic_ratio - 0.3182) < 1e-3
    assert cfg_e2.training_budget.fixed_optimizer_steps_per_epoch == 11
    assert cfg_e2.training_budget.total_optimizer_steps == 33
    assert cfg_e2.training_budget.sample_exposures_per_epoch == 22
    assert cfg_e2.training_budget.total_sample_exposures == 66

    # Mathematical consistency equation check for E2
    assert cfg_e2.training_budget.sample_exposures_per_epoch == (
        cfg_e2.training_budget.fixed_optimizer_steps_per_epoch
        * cfg_e2.training.batch_size
        * cfg_e2.training.gradient_accumulation_steps
    )

    # 4. Strict budget & hyperparameter matching between E1 and E2
    assert cfg_e1.seed == cfg_e2.seed
    assert cfg_e1.lora.r == cfg_e2.lora.r
    assert cfg_e1.lora.lora_alpha == cfg_e2.lora.lora_alpha
    assert cfg_e1.training.learning_rate == cfg_e2.training.learning_rate
    assert cfg_e1.training.epochs == cfg_e2.training.epochs
    assert cfg_e1.training.batch_size == cfg_e2.training.batch_size
    assert cfg_e1.training.gradient_accumulation_steps == cfg_e2.training.gradient_accumulation_steps
    assert cfg_e1.training_budget.total_optimizer_steps == cfg_e2.training_budget.total_optimizer_steps
    assert cfg_e1.training_budget.sample_exposures_per_epoch == cfg_e2.training_budget.sample_exposures_per_epoch

    # 5. Fail loudly if unknown experiment requested
    with pytest.raises(KeyError, match="Experiment 'E99' not found"):
        ExperimentConfig.load_yaml(yaml_path, experiment_id="E99")

    # 6. Fail loudly if multi-experiment YAML loaded without experiment_id
    with pytest.raises(ValueError, match="contains multiple experiments"):
        ExperimentConfig.load_yaml(yaml_path, experiment_id=None)

    # 7. Fail loudly if required section is missing
    bad_yaml = os.path.join(tempfile.gettempdir(), "test_bad_exp.yaml")
    with open(bad_yaml, "w", encoding="utf-8") as f:
        f.write("experiment_id: bad\nmode: train_and_eval\n")
    try:
        with pytest.raises(ValueError, match="Missing required 'lora' section"):
            ExperimentConfig.load_yaml(bad_yaml)
    finally:
        if os.path.exists(bad_yaml):
            os.remove(bad_yaml)
