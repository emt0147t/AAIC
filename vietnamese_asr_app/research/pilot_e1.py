"""Phase 2 Controlled E1 Pilot Runner: Steps 1 through 6.

Executes:
- STEP 1: GPU Preflight & Parameter Accounting
- STEP 2: One-Batch Smoke Test (real batch forward, backward, optimizer step)
- STEP 3: Data Integrity Check (manifests, files, speaker disjointness, zero contamination)
- STEP 4: Controlled E1 Pilot (1 epoch on pilot data, loss, steps, checkpointing)
- STEP 5: Evaluation Sanity (held-out validation subset inference, micro WER/CER)
- STEP 6: Checkpoint Save / Resume Verification

Usage:
    python research/pilot_e1.py
"""

import gc
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_cosine_schedule_with_warmup
from peft import PeftModel

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from asr.audio import load_audio
from asr.model import detect_hardware
from asr.normalization import normalize_vietnamese_text
from evaluation.metrics import compute_levenshtein_cer, compute_levenshtein_wer
from research.checkpoint import load_lora_checkpoint, save_lora_checkpoint
from research.collator import WhisperDataCollatorWithPadding
from research.lora import build_whisper_lora_model, count_parameters, verify_freeze_integrity
from research.quarantine import DataContaminationError, QuarantineEnforcer


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PilotAudioDataset(Dataset):
    """Dataset reading from verified manifest CSV."""

    def __init__(self, manifest_csv: str, processor: WhisperProcessor):
        self.processor = processor
        self.df = pd.read_csv(manifest_csv)
        self.samples = []

        for idx, row in self.df.iterrows():
            a_path = str(row["audio_path"])
            if not os.path.isabs(a_path) and not os.path.exists(a_path):
                # Try relative to repo root
                alt_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), a_path)
                if os.path.exists(alt_path):
                    a_path = alt_path

            self.samples.append({
                "audio_path": a_path,
                "transcript": str(row["transcript"]).strip(),
                "sample_id": str(row.get("sample_id", f"sample_{idx}")),
                "speaker_id": str(row.get("speaker_id", "unknown")),
                "duration_sec": float(row.get("duration_sec", 0.0)),
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        item = self.samples[idx]
        audio_16k, _ = load_audio(item["audio_path"], target_sr=16000)
        input_features = self.processor(audio_16k, sampling_rate=16000).input_features[0]
        labels = self.processor.tokenizer(item["transcript"]).input_ids
        return {
            "input_features": input_features,
            "labels": labels,
            "sample_id": item["sample_id"],
            "raw_text": item["transcript"],
            "audio_16k": audio_16k,
        }


def step1_gpu_preflight(model_id: str = "vinai/PhoWhisper-tiny") -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("STEP 1 — GPU PREFLIGHT & HARDWARE DIAGNOSTICS")
    print("=" * 70)

    import transformers
    import peft
    import accelerate

    cuda_avail = torch.cuda.is_available()
    cuda_ver = torch.version.cuda if cuda_avail else None
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "N/A"
    gpu_vram = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB" if cuda_avail else "0.00 GB"

    info = {
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "cuda_available": cuda_avail,
        "cuda_version": cuda_ver,
        "gpu_name": gpu_name,
        "gpu_vram": gpu_vram,
        "transformers_version": transformers.__version__,
        "peft_version": peft.__version__,
        "accelerate_version": accelerate.__version__,
    }

    for k, v in info.items():
        print(f"  {k:22s}: {v}")

    print(f"\nLoading base model: {model_id}...")
    processor = WhisperProcessor.from_pretrained(model_id)
    base_model = WhisperForConditionalGeneration.from_pretrained(model_id)

    # Attach verified LoRA configuration
    peft_model, stats = build_whisper_lora_model(
        base_model,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )

    info["total_params"] = stats.total_params
    info["trainable_params"] = stats.trainable_params
    info["trainable_ratio_pct"] = stats.trainable_ratio_pct
    info["matched_modules_count"] = len(stats.target_modules_matched)

    print("\nLoRA Parameter Verification:")
    print(f"  Total Parameters:       {stats.total_params:,}")
    print(f"  Trainable Parameters:   {stats.trainable_params:,}")
    print(f"  Trainable Ratio:        {stats.trainable_ratio_pct:.4f}%")
    print(f"  Matched Target Modules: {len(stats.target_modules_matched)} layers")

    # Assert exact match with CPU verification
    assert stats.total_params == 37_908_096, f"Mismatch in total params: {stats.total_params}"
    assert stats.trainable_params == 147_456, f"Mismatch in trainable params: {stats.trainable_params}"
    assert abs(stats.trainable_ratio_pct - 0.3890) < 1e-3, f"Mismatch in ratio: {stats.trainable_ratio_pct}"

    return info, peft_model, processor


def step2_one_batch_smoke_test(
    peft_model: nn.Module,
    processor: WhisperProcessor,
    train_dataset: Dataset,
    device: str,
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("STEP 2 — ONE-BATCH SMOKE TEST")
    print("=" * 70)

    collator = WhisperDataCollatorWithPadding(processor=processor)
    loader = DataLoader(train_dataset, batch_size=2, shuffle=False, collate_fn=collator)
    batch = next(iter(loader))

    target_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    peft_model.to(device)
    peft_model.train()

    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=1e-4)

    input_features = batch["input_features"].to(device, dtype=target_dtype)
    labels = batch["labels"].to(device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()

    # Forward
    outputs = peft_model(input_features=input_features, labels=labels)
    loss = outputs.loss
    loss_val = float(loss.detach().item())
    assert torch.isfinite(loss).item(), "Smoke test loss is non-finite!"

    # Backward
    loss.backward()

    # Verify gradients
    lora_grads = [p.grad for n, p in peft_model.named_parameters() if "lora_" in n and p.grad is not None]
    assert len(lora_grads) > 0, "No LoRA parameter received gradients!"

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    t1 = time.perf_counter()
    batch_time = t1 - t0

    peak_vram = None
    if torch.cuda.is_available():
        peak_vram = f"{torch.cuda.max_memory_allocated(0) / (1024**2):.2f} MB"

    res = {
        "batch_time_sec": round(batch_time, 3),
        "smoke_loss": round(loss_val, 4),
        "peak_vram": peak_vram or "N/A (CPU)",
        "gradients_received": len(lora_grads),
    }

    print(f"  Batch Processing Time:  {res['batch_time_sec']} s")
    print(f"  Smoke Loss:             {res['smoke_loss']}")
    print(f"  Peak VRAM:              {res['peak_vram']}")
    print(f"  LoRA Gradients Checked: {res['gradients_received']} tensors")
    return res


def step3_data_integrity_check(
    train_manifest: str,
    val_manifest: str,
    test_manifest: str,
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("STEP 3 — DATA INTEGRITY & LEAKAGE CHECK")
    print("=" * 70)

    for p in [train_manifest, val_manifest, test_manifest]:
        assert os.path.exists(p), f"Manifest file missing: {p}"

    df_train = pd.read_csv(train_manifest)
    df_val = pd.read_csv(val_manifest)
    df_test = pd.read_csv(test_manifest)

    # 1. Check all audio files exist
    for split_name, df_s in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        for idx, row in df_s.iterrows():
            fpath = str(row["audio_path"])
            assert os.path.exists(fpath), f"{split_name} audio missing: {fpath}"

    # 2. Check no duplicate sample IDs across splits
    train_ids = set(df_train["sample_id"].astype(str))
    val_ids = set(df_val["sample_id"].astype(str))
    test_ids = set(df_test["sample_id"].astype(str))

    assert len(train_ids.intersection(val_ids)) == 0, "Duplicate ID between Train and Val!"
    assert len(train_ids.intersection(test_ids)) == 0, "Duplicate ID between Train and Test!"
    assert len(val_ids.intersection(test_ids)) == 0, "Duplicate ID between Val and Test!"

    # 3. Check speaker disjointness
    train_spks = set(df_train["speaker_id"].astype(str))
    val_spks = set(df_val["speaker_id"].astype(str))
    test_spks = set(df_test["speaker_id"].astype(str))

    assert len(train_spks.intersection(val_spks)) == 0, "Speaker overlap between Train and Val!"
    assert len(train_spks.intersection(test_spks)) == 0, "Speaker overlap between Train and Test!"
    assert len(val_spks.intersection(test_spks)) == 0, "Speaker overlap between Val and Test!"

    # 4. Quarantine test transcript leakage check
    enforcer = QuarantineEnforcer()
    for _, r in df_test.iterrows():
        enforcer.register_quarantined_sample(sample_id=str(r["sample_id"]), transcript=str(r["transcript"]))

    for _, r in df_train.iterrows():
        enforcer.assert_safe(sample_id=str(r["sample_id"]), transcript=str(r["transcript"]), context="Train Split")

    # 5. Check transcripts are non-empty
    for split_name, df_s in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        for _, row in df_s.iterrows():
            norm_t = normalize_vietnamese_text(str(row["transcript"]))
            assert len(norm_t) > 0, f"Empty normalized transcript in {split_name} split!"

    counts = {
        "train_samples": len(df_train),
        "val_samples": len(df_val),
        "test_samples": len(df_test),
        "train_speakers": len(train_spks),
        "val_speakers": len(val_spks),
        "test_speakers": len(test_spks),
        "zero_contamination_verified": True,
        "speaker_disjoint_verified": True,
    }

    print(f"  Train Manifest:  {counts['train_samples']} samples ({counts['train_speakers']} speakers)")
    print(f"  Val Manifest:    {counts['val_samples']} samples ({counts['val_speakers']} speakers)")
    print(f"  Test Manifest:   {counts['test_samples']} samples ({counts['test_speakers']} speakers)")
    print(f"  Audio Files:     All {counts['train_samples'] + counts['val_samples'] + counts['test_samples']} files exist physically on disk.")
    print("  Zero Leakage:    VERIFIED (0 ID overlap, 0 Speaker overlap, 0 Transcript contamination).")

    return counts


def step4_controlled_e1_pilot(
    peft_model: nn.Module,
    processor: WhisperProcessor,
    train_dataset: Dataset,
    val_dataset: Dataset,
    device: str,
    output_dir: str = "checkpoints/E1_pilot",
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("STEP 4 — CONTROLLED E1 PILOT (1 EPOCH)")
    print("=" * 70)
    print("  PIPELINE PILOT — NOT FINAL BENCHMARK")

    os.makedirs(output_dir, exist_ok=True)
    target_dtype = torch.float16 if device.startswith("cuda") else torch.float32

    collator = WhisperDataCollatorWithPadding(processor=processor)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collator)

    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=1e-4)
    total_steps = len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1, num_training_steps=total_steps)

    peft_model.to(device)
    peft_model.train()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t_start = time.perf_counter()
    train_losses = []

    for step, batch in enumerate(train_loader):
        input_features = batch["input_features"].to(device, dtype=target_dtype)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = peft_model(input_features=input_features, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_losses.append(float(loss.detach().item()))

    train_time = time.perf_counter() - t_start
    mean_train_loss = float(np.mean(train_losses))
    initial_train_loss = float(train_losses[0])
    final_train_loss = float(train_losses[-1])
    loss_change = float(final_train_loss - initial_train_loss)

    # Evaluate validation loss
    peft_model.eval()
    val_losses = []
    with torch.inference_mode():
        for batch in val_loader:
            input_features = batch["input_features"].to(device, dtype=target_dtype)
            labels = batch["labels"].to(device)
            outputs = peft_model(input_features=input_features, labels=labels)
            val_losses.append(float(outputs.loss.detach().item()))

    mean_val_loss = float(np.mean(val_losses))

    peak_mem = None
    if torch.cuda.is_available():
        peak_mem = f"{torch.cuda.max_memory_allocated(0) / (1024**2):.2f} MB"

    ckpt_res = save_lora_checkpoint(
        checkpoint_dir=output_dir,
        peft_model=peft_model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        global_step=total_steps,
        base_model_id="vinai/PhoWhisper-tiny",
        metrics={"train_loss": mean_train_loss, "val_loss": mean_val_loss},
    )

    res = {
        "epochs_completed": 1,
        "optimizer_steps": total_steps,
        "training_time_sec": round(train_time, 2),
        "initial_train_loss": round(initial_train_loss, 4),
        "final_train_loss": round(final_train_loss, 4),
        "loss_change": round(loss_change, 4),
        "mean_train_loss": round(mean_train_loss, 4),
        "mean_val_loss": round(mean_val_loss, 4),
        "peak_memory": peak_mem or "N/A (CPU)",
        "checkpoint_dir": output_dir,
    }

    print(f"  Training Time:        {res['training_time_sec']} s")
    print(f"  Optimizer Steps:      {res['optimizer_steps']}")
    print(f"  Initial Train Loss:   {res['initial_train_loss']}")
    print(f"  Final Train Loss:     {res['final_train_loss']}")
    print(f"  Train Loss Change:    {res['loss_change']}")
    print(f"  Mean Train Loss:      {res['mean_train_loss']}")
    print(f"  Mean Val Loss:        {res['mean_val_loss']}")
    print(f"  Peak Memory:          {res['peak_memory']}")
    print(f"  Checkpoint Saved:     {output_dir}")

    return res


def step5_evaluation_sanity(
    peft_model: nn.Module,
    processor: WhisperProcessor,
    eval_dataset: Dataset,
    device: str,
    split_name: str = "Validation",
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print(f"STEP 5 — EVALUATION SANITY (HELD-OUT {split_name.upper()} SUBSET)")
    print("=" * 70)
    print("  PIPELINE PILOT — NOT FINAL BENCHMARK")

    peft_model.eval()
    peft_model.to(device)
    target_dtype = torch.float16 if device.startswith("cuda") else torch.float32

    total_words = 0
    total_word_errors = 0
    total_chars = 0
    total_char_errors = 0
    predictions = []

    with torch.inference_mode():
        for i in range(len(eval_dataset)):
            item = eval_dataset[i]
            audio_16k = item["audio_16k"]
            ref_text = item["raw_text"]

            inputs = processor(audio_16k, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(device, dtype=target_dtype)

            pred_ids = peft_model.generate(
                input_features,
                language="vi",
                task="transcribe",
                max_new_tokens=128,
            )
            hyp_text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()

            wer_res = compute_levenshtein_wer(ref_text, hyp_text)
            cer_res = compute_levenshtein_cer(ref_text, hyp_text)

            total_words += max(1, wer_res.total_reference_tokens)
            total_word_errors += (wer_res.substitutions + wer_res.deletions + wer_res.insertions)
            total_chars += max(1, cer_res.total_reference_tokens)
            total_char_errors += (cer_res.substitutions + cer_res.deletions + cer_res.insertions)

            predictions.append({
                "sample_id": item["sample_id"],
                "reference": ref_text,
                "hypothesis": hyp_text,
                "wer": wer_res.error_rate,
                "cer": cer_res.error_rate,
            })

    micro_wer = total_word_errors / max(1, total_words)
    micro_cer = total_char_errors / max(1, total_chars)

    res = {
        "split": split_name,
        "evaluated_samples": len(predictions),
        "pilot_micro_wer": round(micro_wer, 4),
        "pilot_micro_cer": round(micro_cer, 4),
        "transcripts_non_empty": all(len(p["hypothesis"]) > 0 for p in predictions),
    }

    print(f"  Split:                {res['split']}")
    print(f"  Evaluated Samples:    {res['evaluated_samples']}")
    print(f"  Pilot Micro WER:      {res['pilot_micro_wer'] * 100:.2f}%")
    print(f"  Pilot Micro CER:      {res['pilot_micro_cer'] * 100:.2f}%")
    print(f"  Non-empty Output:     {res['transcripts_non_empty']}")

    return res


def step6_checkpoint_resume_verification(
    checkpoint_dir: str,
    processor: WhisperProcessor,
    model_id: str,
    train_dataset: Dataset,
    device: str,
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("STEP 6 — CHECKPOINT RESUME & STATE RESTORATION")
    print("=" * 70)

    # 1. Instantiate clean base model in fresh state
    clean_base = WhisperForConditionalGeneration.from_pretrained(model_id)

    # 2. Reload adapter onto clean base model
    reloaded_model, state = load_lora_checkpoint(
        checkpoint_dir=checkpoint_dir,
        base_model=clean_base,
        is_trainable=True,
    )
    assert isinstance(reloaded_model, PeftModel), "Reloaded model is not a PeftModel!"
    assert state.get("epoch") == 1, "Failed to restore epoch count!"

    # 3. Restore optimizer state
    restored_opt = torch.optim.AdamW(reloaded_model.parameters(), lr=1e-4)
    if state.get("optimizer_state_dict"):
        restored_opt.load_state_dict(state["optimizer_state_dict"])

    # 4. Perform 1 additional training step
    reloaded_model.to(device)
    reloaded_model.train()
    target_dtype = torch.float16 if device.startswith("cuda") else torch.float32

    collator = WhisperDataCollatorWithPadding(processor=processor)
    loader = DataLoader(train_dataset, batch_size=2, shuffle=False, collate_fn=collator)
    batch = next(iter(loader))

    input_features = batch["input_features"].to(device, dtype=target_dtype)
    labels = batch["labels"].to(device)

    restored_opt.zero_grad()
    outputs = reloaded_model(input_features=input_features, labels=labels)
    loss = outputs.loss
    loss.backward()
    restored_opt.step()

    loss_val = float(loss.detach().item())
    assert torch.isfinite(loss).item(), "Resumed step loss is non-finite!"

    res = {
        "restored_epoch": state.get("epoch"),
        "restored_step": state.get("global_step"),
        "post_resume_step_loss": round(loss_val, 4),
        "resume_successful": True,
    }

    print(f"  Restored Epoch:       {res['restored_epoch']}")
    print(f"  Restored Step:        {res['restored_step']}")
    print(f"  Post-resume Step Loss:{res['post_resume_step_loss']}")
    print("  Resume Verification:  PASSED (Zero runtime exceptions, gradients applied).")

    return res


def main():
    set_seed(42)

    model_id = "vinai/PhoWhisper-tiny"
    manifest_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "manifests"))
    train_m = os.path.join(manifest_dir, "pilot_train_manifest.csv")
    val_m = os.path.join(manifest_dir, "pilot_val_manifest.csv")
    test_m = os.path.join(manifest_dir, "pilot_test_manifest.csv")

    # Step 1: Preflight
    preflight_info, peft_model, processor = step1_gpu_preflight(model_id=model_id)

    # Step 3: Data Integrity Check
    data_counts = step3_data_integrity_check(train_m, val_m, test_m)

    # Prepare datasets
    train_ds = PilotAudioDataset(train_m, processor=processor)
    val_ds = PilotAudioDataset(val_m, processor=processor)
    test_ds = PilotAudioDataset(test_m, processor=processor)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Step 2: One-Batch Smoke Test
    smoke_res = step2_one_batch_smoke_test(peft_model, processor, train_ds, device=device)

    # Step 4: Controlled Pilot (1 Epoch)
    pilot_res = step4_controlled_e1_pilot(
        peft_model, processor, train_ds, val_ds, device=device, output_dir="checkpoints/E1_pilot"
    )

    # Step 5: Evaluation Sanity (Validation & Test)
    eval_val_res = step5_evaluation_sanity(peft_model, processor, val_ds, device=device, split_name="Validation")
    eval_test_res = step5_evaluation_sanity(peft_model, processor, test_ds, device=device, split_name="Test")

    # Step 6: Checkpoint / Resume Verification
    resume_res = step6_checkpoint_resume_verification(
        checkpoint_dir="checkpoints/E1_pilot",
        processor=processor,
        model_id=model_id,
        train_dataset=train_ds,
        device=device,
    )

    print("\n" + "=" * 70)
    print("CONTROLLED E1 PILOT PIPELINE COMPLETED WITH 100% SUCCESS")
    print("=" * 70)


if __name__ == "__main__":
    main()
