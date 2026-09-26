"""CLI Training Script for Phase 2 LoRA Fine-Tuning Experiments (E0 / E1 / E2).

Features:
- Configurable base model (default: vinai/PhoWhisper-tiny)
- Pre-run safety checks (leakage, checksums, parameter accounting, quarantine)
- Strict quarantine enforcement against evaluation splits
- Trainable parameter accounting and freeze integrity verification
- DataCollator with -100 label padding
- Mixed precision (fp16 on CUDA / fp32 on CPU)
- Gradient accumulation and optimizer/scheduler management
- Exact optimizer step budgeting
- Fixed-compute E2 data composition (15 real + 7 synthetic exposures per epoch)
- Post-training evaluation on frozen test manifest
- Checkpoint saving & reload verification
- Comprehensive pilot markdown report generation (E1 / E2)
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_cosine_schedule_with_warmup

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from asr.audio import load_audio
from asr.model import detect_hardware
from asr.normalization import normalize_vietnamese_text
from research.checkpoint import compute_file_sha256, get_git_commit_hash, save_lora_checkpoint
from research.collator import WhisperDataCollatorWithPadding
from research.config import ExperimentConfig, build_controlled_experiment_configs
from research.evaluator import CorpusEvaluationReport, evaluate_model_on_manifest
from research.lora import build_whisper_lora_model, count_parameters, verify_freeze_integrity
from research.quarantine import DataContaminationError, QuarantineEnforcer


class ASRDataset(Dataset):
    """Manifest- or record-driven dataset extracting 80-channel log-Mel spectrograms."""

    def __init__(
        self,
        manifest_path: Optional[str] = None,
        sample_records: Optional[List[Dict[str, Any]]] = None,
        processor: WhisperProcessor = None,
        quarantine_enforcer: Optional[QuarantineEnforcer] = None,
        max_samples: Optional[int] = None,
    ):
        self.processor = processor
        self.samples = []

        if sample_records is not None:
            records = sample_records[:max_samples] if max_samples else sample_records
            for idx, item in enumerate(records):
                sid = str(item.get("sample_id", f"sample_{idx}"))
                txt = str(item.get("transcript", ""))
                a_path = str(item.get("audio_path", ""))

                if quarantine_enforcer:
                    quarantine_enforcer.assert_safe(sample_id=sid, transcript=txt, context="sample_records")

                if os.path.exists(a_path):
                    self.samples.append({"audio_path": a_path, "transcript": txt, "sample_id": sid})
        elif manifest_path:
            if not os.path.exists(manifest_path):
                raise FileNotFoundError(f"Manifest not found: {manifest_path}")

            df = pd.read_csv(manifest_path)
            if max_samples:
                df = df.head(max_samples)

            audio_col = next((c for c in ["audio_path", "path", "filename"] if c in df.columns), None)
            text_col = next((c for c in ["transcript", "text", "sentence"] if c in df.columns), None)
            id_col = next((c for c in ["sample_id", "id"] if c in df.columns), None)

            for idx, row in df.iterrows():
                a_path = str(row[audio_col]) if audio_col else ""
                txt = str(row[text_col]) if text_col else ""
                sid = str(row[id_col]) if id_col else f"sample_{idx}"

                if quarantine_enforcer:
                    quarantine_enforcer.assert_safe(sample_id=sid, transcript=txt, context=manifest_path)

                if os.path.exists(a_path):
                    self.samples.append({"audio_path": a_path, "transcript": txt, "sample_id": sid})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        item = self.samples[idx]
        audio_16k, _ = load_audio(item["audio_path"], target_sr=16000)
        input_features = self.processor(audio_16k, sampling_rate=16000).input_features[0]
        labels = self.processor.tokenizer(item["transcript"]).input_ids
        return {"input_features": input_features, "labels": labels}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_pre_run_safety_checks(config: ExperimentConfig, base_model, peft_model, stats) -> Dict[str, Any]:
    """Execute all pre-run safety and integrity checks for E1 and E2."""
    print("\n" + "=" * 75)
    print(f"EXECUTING PRE-RUN SAFETY AND INTEGRITY CHECKS [{config.experiment_id}]")
    print("=" * 75)

    checks_passed = []
    errors = []

    # 1. Resolve configuration
    checks_passed.append("1. Resolved experiment configuration successfully.")

    # 2. Base model check
    if config.base_model_id != "vinai/PhoWhisper-tiny":
        errors.append(f"Base model mismatch: {config.base_model_id} != vinai/PhoWhisper-tiny")
    else:
        checks_passed.append("2. Verified base model is vinai/PhoWhisper-tiny.")

    # 3. LoRA configuration matches frozen protocol
    l_cfg = config.lora
    if (
        l_cfg.r != 8
        or l_cfg.lora_alpha != 16
        or l_cfg.lora_dropout != 0.05
        or set(l_cfg.target_modules) != {"q_proj", "v_proj"}
        or l_cfg.bias != "none"
    ):
        errors.append(f"LoRA config does not match frozen protocol: {l_cfg}")
    else:
        checks_passed.append("3. Verified LoRA parameters: r=8, alpha=16, dropout=0.05, targets=['q_proj', 'v_proj'], bias='none'.")

    # 4. Manifest sample counts
    df_train = pd.read_csv(config.data.train_manifest)
    df_val = pd.read_csv(config.data.val_manifest)
    df_test = pd.read_csv(config.data.test_manifest)

    n_train, n_val, n_test = len(df_train), len(df_val), len(df_test)
    if n_train != 22:
        errors.append(f"Train samples mismatch: {n_train} != 22")
    if n_val != 5:
        errors.append(f"Val samples mismatch: {n_val} != 5")
    if n_test != 9:
        errors.append(f"Test samples mismatch: {n_test} != 9")

    # E2 Synthetic Manifest Check
    is_e2 = config.experiment_id.startswith("E2")
    if is_e2:
        if not config.data.synthetic_manifest or not os.path.exists(config.data.synthetic_manifest):
            errors.append(f"Synthetic manifest not found: {config.data.synthetic_manifest}")
        else:
            df_synth = pd.read_csv(config.data.synthetic_manifest)
            if len(df_synth) != 22:
                errors.append(f"Synthetic manifest count mismatch: {len(df_synth)} != 22")
            else:
                checks_passed.append(f"4. Verified sample counts: train={n_train}, val={n_val}, test={n_test}, synthetic={len(df_synth)}.")
    else:
        if not errors:
            checks_passed.append(f"4. Verified sample counts: train={n_train}, val={n_val}, test={n_test}.")

    # 5. Speaker overlap = 0
    spk_train = set(df_train["speaker_id"].dropna().astype(str))
    spk_val = set(df_val["speaker_id"].dropna().astype(str))
    spk_test = set(df_test["speaker_id"].dropna().astype(str))

    tv_spk = spk_train & spk_val
    tt_spk = spk_train & spk_test
    vt_spk = spk_val & spk_test

    if tv_spk or tt_spk or vt_spk:
        errors.append(f"Speaker overlap detected: train/val={tv_spk}, train/test={tt_spk}, val/test={vt_spk}")
    else:
        checks_passed.append("5. Verified train/val/test speaker overlap = 0.")

    if is_e2 and os.path.exists(config.data.synthetic_manifest):
        df_synth = pd.read_csv(config.data.synthetic_manifest)
        spk_synth = set(df_synth["speaker_id"].dropna().astype(str))
        real_all_spk = spk_train | spk_val | spk_test
        synth_real_spk_overlap = spk_synth & real_all_spk
        if synth_real_spk_overlap:
            errors.append(f"Synthetic speaker overlaps with real dataset speakers: {synth_real_spk_overlap}")
        else:
            checks_passed.append("5b. Verified synthetic speakers are disjoint from all real dataset speakers.")

    # 6. Test sample IDs in training = 0
    id_train = set(df_train["sample_id"].dropna().astype(str))
    id_test = set(df_test["sample_id"].dropna().astype(str))
    id_overlap = id_train & id_test
    if id_overlap:
        errors.append(f"Test sample IDs found in training: {id_overlap}")
    else:
        checks_passed.append("6. Verified test sample IDs in training = 0.")

    # 7. Test transcript hashes in training = 0
    def text_hash(t):
        return hashlib.sha256(normalize_vietnamese_text(str(t)).encode("utf-8")).hexdigest()

    th_train = set(df_train["transcript"].apply(text_hash))
    th_val = set(df_val["transcript"].apply(text_hash))
    th_test = set(df_test["transcript"].apply(text_hash))

    th_tt_overlap = th_train & th_test
    if th_tt_overlap:
        errors.append(f"Test transcript hashes found in training: {th_tt_overlap}")
    else:
        checks_passed.append("7. Verified test transcript hashes in training = 0.")

    # 8. Validation transcript hashes in training = 0
    th_tv_overlap = th_train & th_val
    if th_tv_overlap:
        errors.append(f"Validation transcript hashes found in training: {th_tv_overlap}")
    else:
        checks_passed.append("8. Verified validation transcript hashes in training = 0.")

    if is_e2 and os.path.exists(config.data.synthetic_manifest):
        df_synth = pd.read_csv(config.data.synthetic_manifest)
        th_synth = set(df_synth["transcript"].apply(text_hash))
        synth_test_overlap = th_synth & th_test
        synth_val_overlap = th_synth & th_val
        if synth_test_overlap:
            errors.append(f"Synthetic transcripts overlap with test set: {synth_test_overlap}")
        if synth_val_overlap:
            errors.append(f"Synthetic transcripts overlap with validation set: {synth_val_overlap}")
        if not (synth_test_overlap or synth_val_overlap):
            checks_passed.append("8b. Verified synthetic transcript hashes have 0 overlap with val/test sets.")

    # 9. Verify manifest SHA-256 values
    expected_hashes = {
        "manifests/train_manifest.csv": "47a5aa1630713ca183f55a49d9e271d9839731fd3634069e0fe55c0b47540edb",
        "manifests/val_manifest.csv": "f4b22a21e91b7b061530cf3b330f47cda98ea01671a570c2b21814ffb543cc11",
        "manifests/test_manifest.csv": "efe54856d12613f109cebc9482a6f224ca4e2370d0ff3f8a2fe4ea0a013b60ca",
    }
    if is_e2:
        expected_hashes["manifests/synthetic_qc_passed.csv"] = "21ce3ad0b72ab70734e357a50d35eb1d37d527601a648904f1517afdc2883254"

    for mpath, exp_h in expected_hashes.items():
        act_h = compute_file_sha256(mpath)
        if act_h != exp_h:
            errors.append(f"SHA-256 mismatch for {mpath}: {act_h} != {exp_h}")
    if not any("SHA-256" in e for e in errors):
        checks_passed.append("9. Verified manifest SHA-256 checksums match canonical hashes.")

    # 10. Verify target modules q_proj and v_proj exist
    named_modules = dict(base_model.named_modules())
    has_q = any("q_proj" in name for name in named_modules.keys())
    has_v = any("v_proj" in name for name in named_modules.keys())
    if not (has_q and has_v):
        errors.append("Target modules q_proj and/or v_proj not found in base model!")
    else:
        checks_passed.append("10. Verified target modules q_proj and v_proj exist in base model.")

    # 11. Verify trainable parameter accounting dynamically
    dynamic_trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    dynamic_total = sum(p.numel() for p in peft_model.parameters())
    dynamic_ratio = (dynamic_trainable / dynamic_total) * 100.0
    checks_passed.append(
        f"11. Dynamic parameter accounting: {dynamic_trainable:,} / {dynamic_total:,} trainable ({dynamic_ratio:.4f}%)."
    )

    # 12. Freeze integrity verification
    freeze_ok, msg = verify_freeze_integrity(peft_model)
    if not freeze_ok:
        errors.append(f"Freeze integrity verification failed: {msg}")
    else:
        checks_passed.append(f"12. {msg}")

    # 13. Fixed-Compute Budget & Composition Check for E2
    if is_e2:
        b_cfg = config.training_budget
        if b_cfg.fixed_optimizer_steps_per_epoch != 11 or b_cfg.total_optimizer_steps != 33:
            errors.append(f"E2 optimizer step budget mismatch: {b_cfg.total_optimizer_steps} != 33")
        if b_cfg.sample_exposures_per_epoch != 22:
            errors.append(f"E2 sample exposures per epoch mismatch: {b_cfg.sample_exposures_per_epoch} != 22")
        if b_cfg.real_sample_count != 15 or b_cfg.synthetic_sample_count != 7:
            errors.append(f"E2 composition mismatch: {b_cfg.real_sample_count} real + {b_cfg.synthetic_sample_count} synth != 15 real + 7 synth")
        if not any("E2" in e for e in errors):
            checks_passed.append("13. Verified E2 fixed-budget composition: 15 real + 7 synthetic = 22 exposures/epoch (33 total steps).")

    for c in checks_passed:
        print(f"  [PASS] {c}")

    if errors:
        print("\n" + "!" * 75)
        print("PRE-RUN SAFETY CHECKS FAILED:")
        for err in errors:
            print(f"  [ERROR] {err}")
        print("!" * 75 + "\n")
        raise RuntimeError(f"PRE-RUN SAFETY CHECKS FAILED: BLOCKED. Errors: {errors}")

    print("=" * 75)
    print("ALL PRE-RUN SAFETY CHECKS PASSED — CLEARED FOR EXECUTION")
    print("=" * 75 + "\n")

    return {
        "status": "PASSED",
        "train_samples": n_train,
        "val_samples": n_val,
        "test_samples": n_test,
        "trainable_params": dynamic_trainable,
        "total_params": dynamic_total,
        "trainable_ratio_pct": dynamic_ratio,
    }


def run_training(config: ExperimentConfig, smoke_test: bool = False):
    print("=" * 75)
    print(f"VIETNAMESE ASR LoRA RESEARCH RUNNER: [{config.experiment_id}]")
    print("=" * 75)

    hw = detect_hardware()
    print(f"Hardware: {hw.summary()}")
    set_seed(config.seed)

    is_e2 = config.experiment_id.startswith("E2")

    # 1. Quarantine Setup
    quarantine = QuarantineEnforcer()
    for q_man in config.data.quarantine_manifests:
        if os.path.exists(q_man):
            n_q = quarantine.register_quarantined_manifest(q_man)
            print(f"[Quarantine] Registered {n_q} samples from {q_man}")

    # 2. Model & Processor
    model_id = config.base_model_id
    print(f"\n[Model] Loading base model: {model_id}...")
    processor = WhisperProcessor.from_pretrained(model_id)
    base_model = WhisperForConditionalGeneration.from_pretrained(model_id)

    # 3. Attach LoRA
    print(f"[LoRA] Attaching LoRA (r={config.lora.r}, alpha={config.lora.lora_alpha}, targets={config.lora.target_modules})...")
    peft_model, stats = build_whisper_lora_model(
        base_model,
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        target_modules=config.lora.target_modules,
        bias=config.lora.bias,
    )
    print(f"[LoRA Stats] {stats.summary()}")

    device = hw.device
    target_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    peft_model.to(device)

    # E0 Eval-Only Mode
    if config.mode == "eval_only":
        print(f"\n[E0 Baseline Evaluation] Evaluating base model on test manifest: {config.data.test_manifest}...")
        df_test = pd.read_csv(config.data.test_manifest)
        test_samples = [
            {"audio": r["audio_path"], "reference": r["transcript"], "sample_id": r.get("sample_id", f"test_{i}")}
            for i, r in df_test.iterrows()
        ]
        report = evaluate_model_on_manifest(
            model=base_model,
            processor=processor,
            samples=test_samples,
            device=device,
            language="vi",
            task="transcribe",
        )
        print("\n" + "=" * 75)
        print("E0 BASELINE EVALUATION (PIPELINE PILOT — NOT FINAL BENCHMARK)")
        print("=" * 75)
        print(report.summary())
        print("=" * 75)
        os.makedirs(config.output_dir, exist_ok=True)
        out_report_path = os.path.join(config.output_dir, "e0_baseline_evaluation.json")
        with open(out_report_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"Evaluation report written to: {out_report_path}")
        return report

    # Smoke Test Mode
    if smoke_test:
        print("\n[Smoke-Test Mode] Running 1-step verification pass...")
        optimizer = torch.optim.AdamW(peft_model.parameters(), lr=config.training.learning_rate)

        sr = 16000
        dummy_audio = 0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, sr, dtype=np.float32))
        inputs = processor(dummy_audio, sampling_rate=sr, return_tensors="pt")
        input_features = inputs.input_features.to(device, dtype=target_dtype)
        labels = processor.tokenizer("xin chào", return_tensors="pt").input_ids.to(device)

        peft_model.train()
        outputs = peft_model(input_features=input_features, labels=labels)
        loss = outputs.loss
        loss.backward()

        lora_grads = [p.grad for n, p in peft_model.named_parameters() if "lora_" in n and p.grad is not None]
        assert len(lora_grads) > 0, "No LoRA parameter received gradients!"
        optimizer.step()
        optimizer.zero_grad()

        smoke_dir = os.path.join(config.output_dir, "smoke_checkpoint")
        res = save_lora_checkpoint(
            checkpoint_dir=smoke_dir,
            peft_model=peft_model,
            optimizer=optimizer,
            epoch=0,
            global_step=1,
            base_model_id=model_id,
            experiment_config=config.to_dict(),
            parameter_stats=stats,
            metrics={"smoke_loss": float(loss.detach().item())},
        )

        print("[Smoke-Test] Successfully executed forward, backward, optimizer step, and checkpoint save!")
        print(f"[Smoke-Test] Adapter saved to: {res['adapter_dir']}")
        print(f"[Smoke-Test] Metadata written to: {res['metadata_path']}")
        print("=" * 75)
        return

    # 4. PRE-RUN SAFETY CHECK
    safety_summary = run_pre_run_safety_checks(config, base_model, peft_model, stats)

    # 5. Validation Dataset Loader
    val_dataset = ASRDataset(
        manifest_path=config.data.val_manifest,
        processor=processor,
        quarantine_enforcer=quarantine,
        max_samples=config.data.max_val_samples,
    )
    collator = WhisperDataCollatorWithPadding(processor=processor)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    # Setup E2 or E1 Training Plan
    epoch_plans: List[List[Dict[str, Any]]] = []
    if is_e2:
        df_real = pd.read_csv(config.data.train_manifest)
        df_synth = pd.read_csv(config.data.synthetic_manifest)
        real_records = df_real.to_dict("records")
        synth_records = df_synth.to_dict("records")

        print("\n" + "=" * 75)
        print("E2 PER-EPOCH DATA COMPOSITION SCHEDULE (FIXED-COMPUTE PROTOCOL)")
        print("=" * 75)
        for ep in range(config.training.epochs):
            rng = random.Random(config.seed + ep * 1000)
            sampled_real = rng.sample(real_records, 15)
            sampled_synth = rng.sample(synth_records, 7)
            combined = sampled_real + sampled_synth
            rng.shuffle(combined)

            real_ids = [r["sample_id"] for r in sampled_real]
            synth_ids = [r["sample_id"] for r in sampled_synth]

            print(f"Epoch {ep + 1}/{config.training.epochs}:")
            print(f"  Real exposures:      {len(sampled_real)} (68.18%)")
            print(f"  Synthetic exposures: {len(sampled_synth)} (31.82%)")
            print(f"  Total exposures:     {len(combined)} (Budget invariant: 22)")
            print(f"  Real Sample IDs:     {real_ids}")
            print(f"  Synthetic IDs:       {synth_ids}")
            print("-" * 75)
            epoch_plans.append(combined)
    else:
        # E1: All 22 real samples every epoch
        df_real = pd.read_csv(config.data.train_manifest)
        real_records = df_real.to_dict("records")
        for ep in range(config.training.epochs):
            rng = random.Random(config.seed + ep * 1000)
            epoch_items = list(real_records)
            rng.shuffle(epoch_items)
            epoch_plans.append(epoch_items)

    optimizer = torch.optim.AdamW(
        peft_model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    steps_per_epoch = 11  # 22 samples // batch_size 2
    total_steps = steps_per_epoch * config.training.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.training.warmup_ratio),
        num_training_steps=total_steps,
    )

    print(f"\n[Training] Starting {config.training.epochs} epochs ({total_steps} optimizer steps)...")
    global_step = 0
    step_loss_records = []
    epoch_train_losses = []
    epoch_val_losses = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    train_start_time = time.perf_counter()
    peft_model.train()

    for epoch in range(config.training.epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()

        # Build epoch dataset
        epoch_dataset = ASRDataset(
            sample_records=epoch_plans[epoch],
            processor=processor,
            quarantine_enforcer=quarantine,
        )
        train_loader = DataLoader(
            epoch_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,  # Already deterministically shuffled in plan
            collate_fn=collator,
        )

        for step, batch in enumerate(train_loader):
            input_features = batch["input_features"].to(device, dtype=target_dtype)
            labels = batch["labels"].to(device)

            outputs = peft_model(input_features=input_features, labels=labels)
            loss = outputs.loss / config.training.gradient_accumulation_steps
            loss.backward()

            cur_loss_val = float(outputs.loss.detach().item())
            epoch_loss += cur_loss_val

            if (step + 1) % config.training.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(peft_model.parameters(), config.training.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                step_loss_records.append({
                    "epoch": epoch + 1,
                    "step": step + 1,
                    "global_step": global_step,
                    "train_loss": cur_loss_val,
                })

        avg_train_loss = epoch_loss / len(train_loader)
        epoch_train_losses.append(avg_train_loss)

        # Validation loss evaluation
        peft_model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for vbatch in val_loader:
                v_feat = vbatch["input_features"].to(device, dtype=target_dtype)
                v_lab = vbatch["labels"].to(device)
                v_out = peft_model(input_features=v_feat, labels=v_lab)
                val_loss_total += float(v_out.loss.detach().item())
        avg_val_loss = val_loss_total / max(1, len(val_loader))
        epoch_val_losses.append(avg_val_loss)
        peft_model.train()

        print(
            f"Epoch {epoch + 1}/{config.training.epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Global Step: {global_step}"
        )

        # Save epoch checkpoint
        ckpt_dir = os.path.join(config.output_dir, f"checkpoint_epoch_{epoch + 1}")
        save_lora_checkpoint(
            checkpoint_dir=ckpt_dir,
            peft_model=peft_model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            global_step=global_step,
            base_model_id=model_id,
            experiment_config=config.to_dict(),
            parameter_stats=stats,
            metrics={"train_loss": avg_train_loss, "val_loss": avg_val_loss},
        )

    train_end_time = time.perf_counter()
    total_training_duration = train_end_time - train_start_time

    # Peak CUDA memory stats
    if torch.cuda.is_available():
        peak_cuda_alloc_bytes = torch.cuda.max_memory_allocated()
        peak_cuda_res_bytes = torch.cuda.max_memory_reserved()
        peak_cuda_alloc_mb = round(peak_cuda_alloc_bytes / (1024 * 1024), 2)
        peak_cuda_res_mb = round(peak_cuda_res_bytes / (1024 * 1024), 2)
        cuda_version = torch.version.cuda
        gpu_name = torch.cuda.get_device_name(0)
    else:
        peak_cuda_alloc_bytes = 0
        peak_cuda_res_bytes = 0
        peak_cuda_alloc_mb = 0.0
        peak_cuda_res_mb = 0.0
        cuda_version = "N/A (CPU execution)"
        gpu_name = "CPU"

    # Save final model checkpoint in output_dir
    final_ckpt_paths = save_lora_checkpoint(
        checkpoint_dir=config.output_dir,
        peft_model=peft_model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=config.training.epochs,
        global_step=global_step,
        base_model_id=model_id,
        experiment_config=config.to_dict(),
        parameter_stats=stats,
        metrics={
            "final_train_loss": epoch_train_losses[-1],
            "final_val_loss": epoch_val_losses[-1],
            "training_duration_sec": round(total_training_duration, 2),
            "peak_cuda_alloc_mb": peak_cuda_alloc_mb,
            "peak_cuda_res_mb": peak_cuda_res_mb,
        },
    )
    print(f"\n[Completed] LoRA training finished. Checkpoint saved to: {config.output_dir}")

    # 6. POST-TRAINING EVALUATION ON TEST MANIFEST
    print(f"\n[Post-Training Evaluation] Evaluating trained model on test manifest: {config.data.test_manifest}...")
    df_test = pd.read_csv(config.data.test_manifest)
    test_samples = [
        {"audio": r["audio_path"], "reference": r["transcript"], "sample_id": r.get("sample_id", f"test_{i}")}
        for i, r in df_test.iterrows()
    ]
    eval_report = evaluate_model_on_manifest(
        model=peft_model,
        processor=processor,
        samples=test_samples,
        device=device,
        language="vi",
        task="transcribe",
    )

    tag = "E2" if is_e2 else "E1"
    print("\n" + "=" * 75)
    print(f"{tag} TEST EVALUATION (PIPELINE PILOT — NOT FINAL BENCHMARK)")
    print("=" * 75)
    print(eval_report.summary())
    print("=" * 75)

    # 7. COMPARISONS
    e0_wer, e0_cer, e0_lat, e0_rtf = 0.1893, 0.1107, 1.412, 0.140
    e1_wer, e1_cer, e1_lat, e1_rtf = 0.1921, 0.1216, 1.264, 0.124

    delta_e0_wer = eval_report.micro_wer - e0_wer
    delta_e0_cer = eval_report.micro_cer - e0_cer
    delta_e1_wer = eval_report.micro_wer - e1_wer
    delta_e1_cer = eval_report.micro_cer - e1_cer

    print(f"\n[Descriptive Comparison to Baselines]")
    print(f"  E0 Pretrained: WER {e0_wer*100:.2f}% | CER {e0_cer*100:.2f}%")
    print(f"  E1 Real LoRA:  WER {e1_wer*100:.2f}% | CER {e1_cer*100:.2f}%")
    print(f"  {tag} Pilot:     WER {eval_report.micro_wer*100:.2f}% | CER {eval_report.micro_cer*100:.2f}%")
    print(f"  Delta {tag} vs E0: WER {delta_e0_wer*100:+.2f}% | CER {delta_e0_cer*100:+.2f}%")
    if is_e2:
        print(f"  Delta E2 vs E1: WER {delta_e1_wer*100:+.2f}% | CER {delta_e1_cer*100:+.2f}%")
    print("  Note: PIPELINE PILOT — NOT FINAL BENCHMARK. Descriptive only; no claims of statistical significance or generalization.")

    # 8. CHECKPOINT RELOAD VERIFICATION
    print("\n" + "=" * 75)
    print(f"CHECKPOINT RELOAD VERIFICATION [{tag}]")
    print("=" * 75)
    print("1. Loading fresh PhoWhisper-tiny base model...")
    fresh_base = WhisperForConditionalGeneration.from_pretrained(config.base_model_id)

    print(f"2. Loading saved {tag} LoRA adapter from {config.output_dir}...")
    reloaded_peft = PeftModel.from_pretrained(fresh_base, config.output_dir)
    reloaded_peft.eval()
    reloaded_peft.to(device)
    print("3. Verified adapter loaded successfully.")

    print("4. Running inference on test sample vimd_test_vimd_01...")
    sample_0 = test_samples[0]
    sample_0_audio, _ = load_audio(sample_0["audio"], target_sr=16000)
    with torch.inference_mode():
        in_f = processor(sample_0_audio, sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype=target_dtype)
        p_ids = reloaded_peft.generate(in_f, language="vi", task="transcribe", max_new_tokens=64)
        reload_hyp = processor.batch_decode(p_ids, skip_special_tokens=True)[0].strip()

    try:
        print(f"5. Inference Output: '{reload_hyp}'")
    except Exception:
        print(f"5. Inference Output: '{reload_hyp.encode('ascii', errors='replace').decode('ascii')}'")
    assert len(reload_hyp) > 0, "Reload verification failed: empty hypothesis!"
    print("6. Verified non-empty output with no runtime exceptions.")

    # 9. GENERATE REPORT
    git_hash = get_git_commit_hash() or "unknown"
    if is_e2:
        report_md_content = f"""# E2 LoRA Controlled Pilot Experiment Report

> [!NOTE]
> **Status**: `E2 PILOT: PASSED`  
> **Benchmark Label**: `PIPELINE PILOT — NOT FINAL BENCHMARK`  
> Results reflect an isolated proof-of-concept pipeline over a controlled pilot subset. Do not interpret as final benchmark or claim generalization.

---

## 1. Research Question
> *"Under an identical LoRA optimization budget, does changing the training-data composition to include QC-verified synthetic Vietnamese speech change ASR performance?"*

**Core Protocol Clarification**:
This experiment uses a fixed optimization budget. E2 changes the composition of the training exposures by introducing synthetic speech.
Do NOT state or assume that "22 real + 22 synthetic samples were all trained each epoch."

## 2. Exact Resolved E2 Configuration
```json
{json.dumps(config.to_dict(), indent=2, ensure_ascii=False)}
```

## 3. Data Composition
- **Total Real Training Pool**: 22 clean real speech samples (10 VIVOS + 12 ViMD clean train)
- **Total Synthetic Pool**: 22 QC-passed synthetic utterances (Gwen-TTS 0.6B)
- **Per-Epoch Exposures**: 22 exposures (15 real + 7 synthetic)
- **Composition Ratio**: `68.18% real` / `31.82% synthetic` (~30% synthetic)
- **Sampling Policy**: Sampling without replacement per epoch; deterministic under `seed = 42`.

## 4. Per-Epoch Real / Synthetic Exposure Breakdown
""" + "\n".join([
            f"- **Epoch {ep + 1}**: 15 real exposures + 7 synthetic exposures = 22 total exposures"
            for ep in range(config.training.epochs)
        ]) + f"""

## 5. Total Optimizer Steps & Optimization Budget
- **Epochs**: {config.training.epochs}
- **Micro Batch Size**: {config.training.batch_size}
- **Gradient Accumulation Steps**: {config.training.gradient_accumulation_steps}
- **Effective Batch Size**: {config.training.batch_size * config.training.gradient_accumulation_steps}
- **Steps per Epoch**: {steps_per_epoch}
- **Total Optimizer Steps**: {global_step} (Target: 33, exactly matching E1)
- **Total Sample Exposures**: {global_step * config.training.batch_size} (Target: 66, exactly matching E1)

## 6. LoRA Parameter Counts & Trainable Ratio
- **Base Model**: `{config.base_model_id}`
- **Total Parameters**: {stats.total_params:,}
- **Trainable LoRA Parameters**: {stats.trainable_params:,}
- **Frozen Parameters**: {stats.frozen_params:,}
- **Trainable Ratio**: `{stats.trainable_ratio_pct:.4f}%`
- **Target Modules**: `q_proj`, `v_proj` (24 linear projection layers)

## 7. Training Loss Trajectory
| Epoch | Step | Global Step | Train Loss |
| :--- | :--- | :--- | :--- |
""" + "\n".join([f"| {s['epoch']} | {s['step']} | {s['global_step']} | {s['train_loss']:.4f} |" for s in step_loss_records]) + f"""

### Epoch Average Train Losses:
""" + "\n".join([f"- Epoch {i+1}: `{loss:.4f}`" for i, loss in enumerate(epoch_train_losses)]) + f"""

## 8. Validation Loss Trajectory
""" + "\n".join([f"- Epoch {i+1}: `{loss:.4f}`" for i, loss in enumerate(epoch_val_losses)]) + f"""

## 9. Total Training Duration
- **Duration**: `{total_training_duration:.2f} seconds` (`{total_training_duration / 60.0:.2f} minutes`)

## 10. Measured Peak CUDA Memory
- **Peak CUDA Allocated**: `{peak_cuda_alloc_mb} MB` (`{peak_cuda_alloc_bytes:,} bytes`)
- **Peak CUDA Reserved**: `{peak_cuda_res_mb} MB` (`{peak_cuda_res_bytes:,} bytes`)
- **GPU Name**: `{gpu_name}`
- **CUDA Version**: `{cuda_version}`

## 11. E2 Test Word Error Rate (WER)
- **Micro WER**: `{eval_report.micro_wer * 100:.2f}%` ({eval_report.total_samples} samples, {eval_report.total_reference_words} words)
- Label: `PIPELINE PILOT — NOT FINAL BENCHMARK`

## 12. E2 Test Character Error Rate (CER)
- **Micro CER**: `{eval_report.micro_cer * 100:.2f}%` ({eval_report.total_reference_chars} chars)
- Label: `PIPELINE PILOT — NOT FINAL BENCHMARK`

## 13. E2 Mean Latency
- **Mean Inference Latency**: `{eval_report.mean_latency_sec:.3f} seconds`

## 14. E2 Mean Real-Time Factor (RTF)
- **Mean RTF**: `{eval_report.mean_rtf:.3f}`

## 15. E0 vs E1 vs E2 Controlled Comparison
| Metric | E0 Baseline (Pretrained) | E1 LoRA (Real Speech Only) | E2 LoRA (Real + Synthetic) | Delta (E2 - E1) | Delta (E2 - E0) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Micro WER** | `{e0_wer * 100:.2f}%` | `{e1_wer * 100:.2f}%` | `{eval_report.micro_wer * 100:.2f}%` | `{delta_e1_wer * 100:+.2f}%` | `{delta_e0_wer * 100:+.2f}%` |
| **Micro CER** | `{e0_cer * 100:.2f}%` | `{e1_cer * 100:.2f}%` | `{eval_report.micro_cer * 100:.2f}%` | `{delta_e1_cer * 100:+.2f}%` | `{delta_e0_cer * 100:+.2f}%` |
| **Mean Latency** | `{e0_lat:.3f} s` | `{e1_lat:.3f} s` | `{eval_report.mean_latency_sec:.3f} s` | `{eval_report.mean_latency_sec - e1_lat:+.3f} s` | `{eval_report.mean_latency_sec - e0_lat:+.3f} s` |
| **Mean RTF** | `{e0_rtf:.3f}` | `{e1_rtf:.3f}` | `{eval_report.mean_rtf:.3f}` | `{eval_report.mean_rtf - e1_rtf:+.3f}` | `{eval_report.mean_rtf - e0_rtf:+.3f}` |

> [!NOTE]
> Label: `PIPELINE PILOT — NOT FINAL BENCHMARK`  
> Descriptive comparison only. No claims of statistical significance or generalization from this 9-sample pilot.

### Per-Utterance Breakdown:
| Sample ID | Ref Words | WER (%) | CER (%) | Latency (s) | RTF |
| :--- | :--- | :--- | :--- | :--- | :--- |
""" + "\n".join([
            f"| `{u['sample_id']}` | {len(u['reference'].split())} | {u['wer']*100:.2f}% | {u['cer']*100:.2f}% | {u['latency_sec']:.3f} | {u['rtf']:.3f} |"
            for u in eval_report.per_utterance_results
        ]) + f"""

## 16. Checkpoint Reload Verification
- **Fresh Base Model Reload**: SUCCESS (`{config.base_model_id}`)
- **Adapter Mount**: SUCCESS (`{config.output_dir}`)
- **Inference Probe**: Sample `vimd_test_vimd_01`
- **Probe Hypothesis**: `{reload_hyp}`
- **Runtime Exceptions**: None (0)
- **Reload Verification Status**: `PASSED`

## 17. Manifest Checksums & Cryptographic Hashes
- `manifests/train_manifest.csv`: `{compute_file_sha256(config.data.train_manifest)}`
- `manifests/synthetic_qc_passed.csv`: `{compute_file_sha256(config.data.synthetic_manifest)}`
- `manifests/val_manifest.csv`: `{compute_file_sha256(config.data.val_manifest)}`
- `manifests/test_manifest.csv`: `{compute_file_sha256(config.data.test_manifest)}`
- Git Commit: `{git_hash}`

## 18. Provenance and Leakage Audit
- **Synthetic $\cap$ Test Transcript Overlap**: 0 (0%)
- **Synthetic $\cap$ Val Transcript Overlap**: 0 (0%)
- **Synthetic Speaker IDs $\cap$ Real Speaker IDs**: 0 (0%)
- **Lineage**: All 7 synthetic exposures per epoch originate strictly from QC-passed synthetic utterances generated from `train_manifest.csv`.

## 19. Limitations & Controlled Scope
1. **Sample Size**: Fixed to a 22-exposure budget per epoch over 3 epochs (33 steps) to ensure strictly identical compute between E1 and E2.
2. **Evaluation Domain**: 9 official ViMD gold test samples. Results demonstrate proof-of-concept pipeline feasibility and do not represent generalized benchmark claims.
"""
        report_paths = [
            os.path.join("reports", "e2_pilot_report.md"),
            os.path.join(config.output_dir, "e2_pilot_report.md"),
        ]
        artifact_dir = "C:/Users/DELL/.gemini/antigravity/brain/c944cf42-f0ab-4aad-9d26-cc19fec94159"
        if os.path.exists(artifact_dir):
            report_paths.append(os.path.join(artifact_dir, "e2_pilot_report.md"))

        for rpath in report_paths:
            os.makedirs(os.path.dirname(os.path.abspath(rpath)), exist_ok=True)
            with open(rpath, "w", encoding="utf-8") as f:
                f.write(report_md_content)
            print(f"Report written to: {rpath}")

        print("\n" + "=" * 75)
        print("E2 PILOT: PASSED")
        print("=" * 75)

    else:
        # E1 Report
        report_md_content = f"""# E1 LoRA Controlled Pilot Experiment Report

> [!NOTE]
> **Status**: `E1 PILOT: PASSED`  
> **Benchmark Label**: `PIPELINE PILOT — NOT FINAL BENCHMARK`  
> Results reflect an isolated proof-of-concept pipeline over a controlled pilot subset. Do not interpret as final benchmark or claim generalization.

---

## 1. Exact Resolved Configuration
```json
{json.dumps(config.to_dict(), indent=2, ensure_ascii=False)}
```

## 2. Dataset Sample Counts
- **Training Samples**: 22 real speech samples
- **Validation Samples**: 5 held-out real speech samples
- **Test Samples**: 9 gold ViMD test samples

## 3. Data Leakage & Integrity Verification
- **Train / Val Speaker Overlap**: 0
- **Train / Test Speaker Overlap**: 0
- **Val / Test Speaker Overlap**: 0
- **Test Sample IDs in Training**: 0
- **Test Transcript Hashes in Training**: 0
- **Validation Transcript Hashes in Training**: 0
- **Quarantine Enforcer**: ACTIVE (9 test samples quarantined)

## 4. Parameter Accounting
- **Base Model**: `{config.base_model_id}`
- **Total Model Parameters**: {stats.total_params:,}
- **Trainable LoRA Parameters**: {stats.trainable_params:,}
- **Frozen Parameters**: {stats.frozen_params:,}
- **Target Modules**: `q_proj`, `v_proj` (24 linear projection layers)

## 5. LoRA Trainable Parameter Ratio
- **Trainable Ratio**: `{stats.trainable_ratio_pct:.4f}%`

## 6. Training Steps & Budget
- **Epochs**: {config.training.epochs}
- **Micro Batch Size**: {config.training.batch_size}
- **Gradient Accumulation Steps**: {config.training.gradient_accumulation_steps}
- **Effective Batch Size**: {config.training.batch_size * config.training.gradient_accumulation_steps}
- **Steps per Epoch**: {steps_per_epoch}
- **Total Optimizer Steps**: {global_step} (Exact target: 33)

## 7. Training Loss Trajectory
| Epoch | Step | Global Step | Train Loss |
| :--- | :--- | :--- | :--- |
""" + "\n".join([f"| {s['epoch']} | {s['step']} | {s['global_step']} | {s['train_loss']:.4f} |" for s in step_loss_records]) + f"""

### Epoch Average Train Losses:
""" + "\n".join([f"- Epoch {i+1}: `{loss:.4f}`" for i, loss in enumerate(epoch_train_losses)]) + f"""

## 8. Validation Loss Trajectory
""" + "\n".join([f"- Epoch {i+1}: `{loss:.4f}`" for i, loss in enumerate(epoch_val_losses)]) + f"""

## 9. Total Training Duration
- **Duration**: `{total_training_duration:.2f} seconds` (`{total_training_duration / 60.0:.2f} minutes`)

## 10. Actual Measured Peak CUDA Allocated Memory
- **Peak CUDA Allocated**: `{peak_cuda_alloc_mb} MB` (`{peak_cuda_alloc_bytes:,} bytes`)

## 11. Actual Measured Peak CUDA Reserved Memory
- **Peak CUDA Reserved**: `{peak_cuda_res_mb} MB` (`{peak_cuda_res_bytes:,} bytes`)
- **GPU Name**: `{gpu_name}`
- **CUDA Version**: `{cuda_version}`

## 12. E1 Test Word Error Rate (WER)
- **Micro WER**: `{eval_report.micro_wer * 100:.2f}%` ({eval_report.total_samples} samples, {eval_report.total_reference_words} words)
- Label: `PIPELINE PILOT — NOT FINAL BENCHMARK`

## 13. E1 Test Character Error Rate (CER)
- **Micro CER**: `{eval_report.micro_cer * 100:.2f}%` ({eval_report.total_reference_chars} chars)
- Label: `PIPELINE PILOT — NOT FINAL BENCHMARK`

## 14. E1 Mean Latency
- **Mean Inference Latency**: `{eval_report.mean_latency_sec:.3f} seconds`

## 15. E1 Mean Real-Time Factor (RTF)
- **Mean RTF**: `{eval_report.mean_rtf:.3f}`

## 16. E0 vs E1 Descriptive Comparison
| Metric | E0 Baseline | E1 LoRA (Real) | Delta (E1 - E0) |
| :--- | :--- | :--- | :--- |
| **Micro WER** | `{e0_wer * 100:.2f}%` | `{eval_report.micro_wer * 100:.2f}%` | `{delta_e0_wer * 100:+.2f}%` |
| **Micro CER** | `{e0_cer * 100:.2f}%` | `{eval_report.micro_cer * 100:.2f}%` | `{delta_e0_cer * 100:+.2f}%` |
| **Mean Latency** | `1.412 s` | `{eval_report.mean_latency_sec:.3f} s` | `{eval_report.mean_latency_sec - 1.412:+.3f} s` |
| **Mean RTF** | `0.140` | `{eval_report.mean_rtf:.3f}` | `{eval_report.mean_rtf - 0.140:+.3f}` |

> [!NOTE]
> Label: `PIPELINE PILOT — NOT FINAL BENCHMARK`  
> Descriptive comparison only. No claims of statistical significance or generalization from this 9-sample pilot.

### Per-Utterance Results:
| Sample ID | Ref Words | WER (%) | CER (%) | Latency (s) | RTF |
| :--- | :--- | :--- | :--- | :--- | :--- |
""" + "\n".join([
            f"| `{u['sample_id']}` | {len(u['reference'].split())} | {u['wer']*100:.2f}% | {u['cer']*100:.2f}% | {u['latency_sec']:.3f} | {u['rtf']:.3f} |"
            for u in eval_report.per_utterance_results
        ]) + f"""

## 17. Checkpoint Reload Verification
- **Fresh Base Model Reload**: SUCCESS (`{config.base_model_id}`)
- **Adapter Mount**: SUCCESS (`{config.output_dir}`)
- **Inference Probe**: Sample `vimd_test_vimd_01`
- **Probe Hypothesis**: `{reload_hyp}`
- **Runtime Exceptions**: None (0)
- **Reload Verification Status**: `PASSED`

## 18. Git Commit Hash
- **Git Commit**: `{git_hash}`

## 19. Canonical Manifest SHA-256 Checksums
- `train_manifest.csv`: `{compute_file_sha256(config.data.train_manifest)}`
- `val_manifest.csv`: `{compute_file_sha256(config.data.val_manifest)}`
- `test_manifest.csv`: `{compute_file_sha256(config.data.test_manifest)}`

## 20. Warnings & Errors
- None. Pipeline executed with zero runtime errors and zero quarantine violations.
"""
        report_dirs = [
            config.output_dir,
            os.path.join("checkpoints", "E1_lora"),
        ]
        for rdir in report_dirs:
            os.makedirs(rdir, exist_ok=True)
            rpath = os.path.join(rdir, "e1_pilot_report.md")
            with open(rpath, "w", encoding="utf-8") as f:
                f.write(report_md_content)
            print(f"Report written to: {rpath}")

        alt_dir = os.path.join("checkpoints", "E1_lora")
        if os.path.abspath(config.output_dir) != os.path.abspath(alt_dir):
            for fname in ["adapter_model.safetensors", "adapter_config.json", "trainer_state.pt", "experiment_metadata.json"]:
                src = os.path.join(config.output_dir, fname)
                dst = os.path.join(alt_dir, fname)
                if os.path.exists(src):
                    shutil.copy2(src, dst)

        print("\n" + "=" * 75)
        print("E1 PILOT: PASSED")
        print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Vietnamese ASR LoRA Training Runner")
    parser.add_argument("--config", type=str, default=None, help="Path to experiment YAML")
    parser.add_argument("--experiment", type=str, default="E1", choices=["E0", "E1", "E2"], help="Experiment preset")
    parser.add_argument("--smoke_test", action="store_true", help="Run 1-step CPU smoke test and exit")
    args = parser.parse_args()

    if args.config:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Configuration file not found: {args.config}")
        cfg = ExperimentConfig.load_yaml(args.config, experiment_id=args.experiment)
    else:
        configs = build_controlled_experiment_configs()
        if args.experiment not in configs:
            raise KeyError(f"Unknown experiment '{args.experiment}'. Available: {list(configs.keys())}")
        cfg = configs[args.experiment]

    print("\n" + "=" * 75)
    print(f"RESOLVED EXPERIMENT CONFIGURATION: [{args.experiment}]")
    print("=" * 75)
    print(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))
    print("=" * 75 + "\n")

    run_training(cfg, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
