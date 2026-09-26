"""Audited Full E1 Training Script for Vietnamese ASR.

Frozen Protocol:
- Base Model: vinai/PhoWhisper-tiny (cc51d32be916efebde04ff549854fa1741cb5c02)
- PEFT: 0.21.0
- Transformers: 4.49.0
- LoRA: r=8, alpha=16, dropout=0.05, target_modules=['q_proj', 'v_proj'], bias='none'
- Trainable Params: 147,456 (0.389%)
- Micro Batch: 4, Grad Accum: 8, Effective Batch: 32
- Optimizer: AdamW (lr=1e-4, weight_decay=0.01, grad_clip=1.0)
- Scheduler: Cosine with 10% warmup (250 warmup steps, 2502 total steps)
- Epochs: 3 (834 steps/epoch)
- Precision: FP16 mixed precision on CUDA
- Population: 26,671 training-ready utterances (VIVOS: 11,660, ViMD: 15,011)
- Validation: manifests/full_val_manifest.csv (1,900 utterances)
- Test: manifests/test_manifest.csv (9 gold ViMD test samples, strictly evaluated post-training)
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    get_cosine_schedule_with_warmup,
)

# Ensure UTF-8 on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add application root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from asr.audio import load_audio
from asr.normalization import normalize_vietnamese_text
from evaluation.metrics import compute_levenshtein_cer, compute_levenshtein_wer
from research.checkpoint import save_lora_checkpoint
from research.collator import WhisperDataCollatorWithPadding
from research.evaluator import evaluate_model_on_manifest


# --- HARD FROZEN CONSTANTS ---
BASE_MODEL_ID = "vinai/PhoWhisper-tiny"
PINNED_MODEL_REVISION = "cc51d32be916efebde04ff549854fa1741cb5c02"
PEFT_REQUIRED_VERSION = "0.21.0"
TRANSFORMERS_REQUIRED_VERSION = "4.49.0"

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
LORA_BIAS = "none"

MICRO_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 8
EFFECTIVE_BATCH_SIZE = 32

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
GRADIENT_CLIP = 1.0
WARMUP_RATIO = 0.10
EPOCHS = 3
SEED = 42

TOTAL_TRAINING_READY = 26671
STEPS_PER_EPOCH = 834
TOTAL_OPTIMIZER_STEPS = 2502
EXPOSURES_PER_EPOCH = 26688
EXTRA_EXPOSURES_PER_EPOCH = 17

AUDITED_REPEATED_INDICES = {
    1: [186, 2394, 3596, 4922, 5311, 9527, 9922, 11904, 12434, 13590, 14261, 15356, 15589, 16068, 19817, 19918, 20082],
    2: [791, 2720, 3793, 4460, 4869, 6765, 7657, 8088, 10060, 10616, 12573, 15738, 17952, 22257, 23086, 23991, 26608],
    3: [329, 854, 868, 2852, 2949, 7126, 9800, 12278, 12298, 17346, 17498, 20781, 21165, 21287, 22885, 24648, 26633],
}

FROZEN_TEST_SHA256 = "efe54856d12613f109cebc9482a6f224ca4e2370d0ff3f8a2fe4ea0a013b60ca"
OUTPUT_DIR = "checkpoints/FULL_E1"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_canonical_crlf_sha256(data: bytes) -> str:
    """Canonicalize text CSV bytes to CRLF line endings and compute SHA-256.

    CSV data is textual. Across platforms and Git configurations (e.g. Windows
    vs Linux/Colab checkouts), Git may store or check out files with LF newlines
    instead of CRLF newlines. The frozen integrity criterion is based on
    canonicalized text content bytes. Changing line endings alone (LF vs CRLF)
    does not represent content modification or data corruption.
    Note: This canonicalization applies strictly to textual CSV manifests,
    specifically manifests/test_manifest.csv.
    """
    canonical = (
        data
        .replace(b"\r\n", b"\n")
        .replace(b"\r", b"\n")
        .replace(b"\n", b"\r\n")
    )
    return hashlib.sha256(canonical).hexdigest()


def verify_exposure_policy(epoch: int, n_samples: int = 26671, base_seed: int = 42) -> List[int]:
    """Computes and asserts the deterministic exposure order and padding indices."""
    epoch_seed = base_seed + (epoch - 1)
    pad_rng = np.random.default_rng(epoch_seed + 1000)
    padding_indices = sorted(pad_rng.choice(n_samples, size=EXTRA_EXPOSURES_PER_EPOCH, replace=False).tolist())
    expected = AUDITED_REPEATED_INDICES[epoch]
    if padding_indices != expected:
        raise ValueError(
            f"FULL E1: ABORTED — EXPOSURE POLICY MISMATCH for Epoch {epoch}!\n"
            f"Expected: {expected}\n"
            f"Computed: {padding_indices}"
        )
    return padding_indices


def build_epoch_exposure_indices(epoch: int, n_samples: int = 26671, base_seed: int = 42) -> List[int]:
    """Constructs the exact 26,688 deterministic sample exposure order for a training epoch."""
    epoch_seed = base_seed + (epoch - 1)
    rng = np.random.default_rng(epoch_seed)
    permutation = rng.permutation(n_samples).tolist()
    
    padding_indices = verify_exposure_policy(epoch, n_samples, base_seed)
    epoch_sequence = permutation + padding_indices
    assert len(epoch_sequence) == EXPOSURES_PER_EPOCH, (
        f"Expected {EXPOSURES_PER_EPOCH} exposures, got {len(epoch_sequence)}"
    )
    return epoch_sequence


def preflight_check(test_manifest_path: str = "manifests/test_manifest.csv") -> Dict[str, Any]:
    print("=" * 70)
    print("FULL E1 PRE-TRAINING INTEGRITY PRECHECK")
    print("=" * 70)

    # 1. Check test manifest hash (Platform-independent canonical CRLF normalization)
    if not os.path.exists(test_manifest_path):
        raise FileNotFoundError(f"Test manifest missing: {test_manifest_path}")
    raw_bytes = open(test_manifest_path, "rb").read()
    raw_hash = hashlib.sha256(raw_bytes).hexdigest()
    canonical_hash = compute_canonical_crlf_sha256(raw_bytes)

    print(f"Frozen test manifest RAW FILE SHA-256:         {raw_hash}")
    print(f"Frozen test manifest CANONICAL FROZEN SHA-256: {canonical_hash}")

    if canonical_hash != FROZEN_TEST_SHA256:
        raise ValueError(
            f"CRITICAL: Test manifest content altered!\n"
            f"Expected Canonical SHA-256: {FROZEN_TEST_SHA256}\n"
            f"Got Canonical SHA-256:      {canonical_hash}\n"
            f"Raw File SHA-256:           {raw_hash}"
        )
    print("Frozen test manifest integrity: PASS")

    # 2. Check Hardware & Versions
    cuda_available = torch.cuda.is_available()

    import peft
    import transformers
    print(f"PEFT Version:         {peft.__version__}")
    print(f"Transformers Version: {transformers.__version__}")
    if peft.__version__ != PEFT_REQUIRED_VERSION:
        raise ValueError(f"PEFT version mismatch! Expected {PEFT_REQUIRED_VERSION}, got {peft.__version__}")
    if transformers.__version__ != TRANSFORMERS_REQUIRED_VERSION:
        if cuda_available:
            raise ValueError(f"Transformers version mismatch! Expected {TRANSFORMERS_REQUIRED_VERSION}, got {transformers.__version__}")
        else:
            print(f"Notice: Local host Transformers is {transformers.__version__} (cloud GPU runtime pinned to {TRANSFORMERS_REQUIRED_VERSION}).")

    # 3. Check CUDA requirement for FP16
    print(f"CUDA Available:       {cuda_available}")
    if cuda_available:
        print(f"CUDA Device:          {torch.cuda.get_device_name(0)}")
        total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        print(f"VRAM Capacity:        {total_vram_mb:.1f} MB")
    else:
        print("\n" + "!" * 70)
        print("CRITICAL HARDWARE FAILURE:")
        print("Protocol requires FP16 mixed precision on CUDA.")
        print("Local host runs PyTorch CPU build. CUDA device is unavailable.")
        print("!" * 70 + "\n")

    # 4. Verify exposure policy for all 3 epochs
    for ep in [1, 2, 3]:
        indices = verify_exposure_policy(ep)
        print(f"Epoch {ep} exposure policy verified: 17 repeated indices match audited schedule.")

    return {
        "raw_test_hash": raw_hash,
        "canonical_test_hash": canonical_hash,
        "peft_version": peft.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": cuda_available,
    }


class FullE1InMemoryDataset(Dataset):
    """Dataset serving pre-extracted Mel spectrogram features and tokenized labels."""

    def __init__(self, sample_records: List[Dict[str, Any]], processor: WhisperProcessor):
        self.samples = sample_records
        self.processor = processor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        waveform = item["waveform"]
        text = item["transcript"]

        inputs = self.processor.feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")
        labels = self.processor.tokenizer(text, return_tensors="pt").input_ids[0]
        return {
            "input_features": inputs.input_features[0],
            "labels": labels,
            "sample_id": item.get("sample_id", f"sample_{idx}"),
        }


def load_full_corpus_records(processor: WhisperProcessor) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Loads the audited full-scale Vietnamese ASR training corpus and validation set.

    In cloud environments (Colab/GPU), streams directly from pinned Hugging Face datasets:
    - VIVOS train: thanhduycao/vivos_ng_only (rev b2fbc10431b721dc9b0409b716d56a759d1cf332)
    - ViMD train: nguyendv02/ViMD_Dataset (rev 3a5b30157034e7eadd5c75fae1a820c6f9383398)
    Applying duration filtering (0.5s <= duration <= 30.0s), yielding exactly 26,671 training utterances.
    """
    print("Loading training corpus records...")
    # Check if local manifests and audio exist
    local_manifest = "manifests/full_train_manifest.csv"
    val_manifest = "manifests/full_val_manifest.csv"

    train_records: List[Dict[str, Any]] = []
    val_records: List[Dict[str, Any]] = []

    try:
        from datasets import Audio, load_dataset

        print("Streaming VIVOS train (pinned revision b2fbc10431b721dc9b0409b716d56a759d1cf332)...")
        ds_vivos = load_dataset(
            "thanhduycao/vivos_ng_only",
            split="train",
            revision="b2fbc10431b721dc9b0409b716d56a759d1cf332",
        )
        ds_vivos = ds_vivos.cast_column("audio", Audio(sampling_rate=16000))
        for row in ds_vivos:
            arr = row["audio"]["array"].astype(np.float32)
            dur = len(arr) / 16000.0
            if 0.5 <= dur <= 30.0:
                train_records.append({
                    "sample_id": f"vivos_{row.get('speaker_id', 'spk')}_{len(train_records)}",
                    "waveform": arr,
                    "transcript": normalize_vietnamese_text(row.get("sentence", row.get("transcript", ""))),
                    "duration_sec": dur,
                })

        print(f"Loaded {len(train_records)} VIVOS train utterances.")

        print("Streaming ViMD train (pinned revision 3a5b30157034e7eadd5c75fae1a820c6f9383398)...")
        ds_vimd = load_dataset(
            "nguyendv02/ViMD_Dataset",
            split="train",
            revision="3a5b30157034e7eadd5c75fae1a820c6f9383398",
        )
        ds_vimd = ds_vimd.cast_column("audio", Audio(sampling_rate=16000))
        vimd_count = 0
        for row in ds_vimd:
            arr = row["audio"]["array"].astype(np.float32)
            dur = len(arr) / 16000.0
            if 0.5 <= dur <= 30.0:
                train_records.append({
                    "sample_id": f"vimd_{row.get('speaker_id', 'spk')}_{vimd_count}",
                    "waveform": arr,
                    "transcript": normalize_vietnamese_text(row.get("transcript", row.get("sentence", ""))),
                    "duration_sec": dur,
                })
                vimd_count += 1

        print(f"Loaded {vimd_count} ViMD train utterances (<=30.0s).")

        # Load ViMD valid
        print("Loading ViMD valid split for validation...")
        ds_val = load_dataset(
            "nguyendv02/ViMD_Dataset",
            split="valid",
            revision="3a5b30157034e7eadd5c75fae1a820c6f9383398",
        )
        ds_val = ds_val.cast_column("audio", Audio(sampling_rate=16000))
        for row in ds_val:
            arr = row["audio"]["array"].astype(np.float32)
            val_records.append({
                "sample_id": f"vimd_val_{len(val_records)}",
                "waveform": arr,
                "transcript": normalize_vietnamese_text(row.get("transcript", row.get("sentence", ""))),
                "duration_sec": len(arr) / 16000.0,
            })
        print(f"Loaded {len(val_records)} ViMD valid utterances.")

    except Exception as e:
        print(f"Cloud dataset streaming failed or offline: {e}")
        # Fallback to local files if available
        if os.path.exists(local_manifest):
            df_train = pd.read_csv(local_manifest)
            for _, r in df_train.iterrows():
                ap = r.get("audio_path", "")
                if os.path.exists(ap):
                    wf, sr = load_audio(ap, target_sr=16000)
                    train_records.append({
                        "sample_id": r.get("sample_id", f"sample_{len(train_records)}"),
                        "waveform": wf,
                        "transcript": normalize_vietnamese_text(r.get("transcript", "")),
                        "duration_sec": len(wf) / 16000.0,
                    })
        if os.path.exists(val_manifest):
            df_v = pd.read_csv(val_manifest)
            for _, r in df_v.iterrows():
                ap = r.get("audio_path", "")
                if os.path.exists(ap):
                    wf, sr = load_audio(ap, target_sr=16000)
                    val_records.append({
                        "sample_id": r.get("sample_id", f"val_{len(val_records)}"),
                        "waveform": wf,
                        "transcript": normalize_vietnamese_text(r.get("transcript", "")),
                        "duration_sec": len(wf) / 16000.0,
                    })

    print(f"Total training-ready records available: {len(train_records)}")
    return train_records, val_records


def run_full_e1_training(
    epochs: int = EPOCHS,
    micro_batch_size: int = MICRO_BATCH_SIZE,
    grad_accum_steps: int = GRAD_ACCUM_STEPS,
    output_dir: str = OUTPUT_DIR,
    val_manifest_path: str = "manifests/full_val_manifest.csv",
    test_manifest_path: str = "manifests/test_manifest.csv",
):
    """Executes the complete audited Full E1 LoRA training run on CUDA."""
    start_time = time.time()
    set_seed(SEED)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # 1. Run Preflight Check
    precheck = preflight_check(test_manifest_path)
    if not precheck["cuda_available"]:
        print("STATUS: FULL E1: ABORTED")
        print("Reason: CUDA execution environment unavailable on local workstation.")
        print("Execution must take place in the audited Google Colab / cloud Linux GPU runtime.")
        sys.exit(1)

    device = torch.device("cuda")

    # 2. Model & Processor Setup
    print("\nInitializing WhisperProcessor and WhisperForConditionalGeneration...")
    processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID, revision=PINNED_MODEL_REVISION)
    base_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_ID, revision=PINNED_MODEL_REVISION)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias=LORA_BIAS,
    )
    model = get_peft_model(base_model, lora_config)
    model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable Parameters: {trainable_params:,} ({100 * trainable_params / all_params:.4f}%)")
    assert trainable_params == 147456, f"Expected 147,456 trainable parameters, got {trainable_params}"

    # 3. Load Data Records
    train_records, val_records = load_full_corpus_records(processor)
    if len(train_records) == 0:
        raise RuntimeError("No training records available to train Full E1!")

    n_corpus_samples = len(train_records)
    print(f"Loaded {n_corpus_samples} training samples.")

    # 4. Setup Optimizer, Scheduler, and GradScaler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    total_micro_steps = TOTAL_OPTIMIZER_STEPS * grad_accum_steps
    warmup_steps = int(TOTAL_OPTIMIZER_STEPS * WARMUP_RATIO) * grad_accum_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_micro_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    collator = WhisperDataCollatorWithPadding(processor=processor)

    epoch_metrics: List[Dict[str, Any]] = []
    global_step = 0
    best_val_loss = float("inf")
    best_checkpoint_path = ""

    print("\n" + "=" * 70)
    print(f"STARTING FULL E1 TRAINING: {epochs} EPOCHS | {TOTAL_OPTIMIZER_STEPS} TOTAL STEPS")
    print("=" * 70)

    # 5. Epoch Loop
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        model.train()

        # Build audited deterministic exposure sequence for this epoch
        exposure_indices = build_epoch_exposure_indices(epoch, n_samples=n_corpus_samples, base_seed=SEED)
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        print(f"Deterministic exposures: {len(exposure_indices)} (834 optimizer steps of effective batch 32)")

        # Create epoch subset from exposure sequence
        epoch_samples = [train_records[idx] for idx in exposure_indices]
        epoch_dataset = FullE1InMemoryDataset(epoch_samples, processor)
        epoch_loader = DataLoader(
            epoch_dataset,
            batch_size=micro_batch_size,
            shuffle=False,  # Already deterministically sequenced
            collate_fn=collator,
        )

        running_loss = 0.0
        optimizer_step_loss = 0.0
        epoch_steps = 0

        for micro_step, batch in enumerate(epoch_loader):
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(input_features=input_features, labels=labels)
                loss = outputs.loss / grad_accum_steps

            scaler.scale(loss).backward()
            running_loss += outputs.loss.item()
            optimizer_step_loss += outputs.loss.item()

            if (micro_step + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                epoch_steps += 1

                if epoch_steps % 50 == 0 or epoch_steps == STEPS_PER_EPOCH:
                    avg_step_loss = optimizer_step_loss / grad_accum_steps
                    lr_curr = scheduler.get_last_lr()[0]
                    print(f"Epoch {epoch} | Step {epoch_steps}/{STEPS_PER_EPOCH} (Global {global_step}) | Loss: {avg_step_loss:.4f} | LR: {lr_curr:.6f}")
                    optimizer_step_loss = 0.0

        avg_epoch_train_loss = running_loss / len(epoch_loader)
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch} complete | Mean Train Loss: {avg_epoch_train_loss:.4f} | Time: {epoch_duration:.1f}s")

        # 6. Epoch Validation
        val_start_time = time.time()
        val_loss = 0.0
        val_samples_evaluated = 0

        if len(val_records) > 0:
            model.eval()
            val_dataset = FullE1InMemoryDataset(val_records, processor)
            val_loader = DataLoader(val_dataset, batch_size=micro_batch_size, shuffle=False, collate_fn=collator)

            with torch.no_grad():
                for v_batch in val_loader:
                    v_feats = v_batch["input_features"].to(device)
                    v_lbls = v_batch["labels"].to(device)
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        v_out = model(input_features=v_feats, labels=v_lbls)
                    val_loss += v_out.loss.item()
                    val_samples_evaluated += len(v_batch["labels"])

            val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
            val_duration = time.time() - val_start_time
            print(f"Epoch {epoch} Validation | Mean Val Loss: {val_loss:.4f} | Evaluated {val_samples_evaluated} samples ({val_duration:.1f}s)")
        else:
            val_loss = 0.0
            val_duration = 0.0

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_epoch_train_loss,
            "val_loss": val_loss,
            "train_duration_sec": epoch_duration,
            "val_duration_sec": val_duration,
            "optimizer_steps": epoch_steps,
            "global_step": global_step,
        }
        epoch_metrics.append(epoch_record)

        # 7. Checkpoint Serialization
        epoch_ckpt_dir = os.path.join(output_dir, f"checkpoint_epoch_{epoch}")
        save_lora_checkpoint(
            checkpoint_dir=epoch_ckpt_dir,
            peft_model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            base_model_id=BASE_MODEL_ID,
            metrics=epoch_record,
        )
        print(f"Saved checkpoint for Epoch {epoch} to {epoch_ckpt_dir}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = epoch_ckpt_dir

    # 8. Final Checkpoint Selection & Post-Training Frozen Test Evaluation
    final_dir = os.path.join(output_dir, "final")
    save_lora_checkpoint(
        checkpoint_dir=final_dir,
        peft_model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epochs,
        global_step=global_step,
        base_model_id=BASE_MODEL_ID,
        metrics={"best_val_loss": best_val_loss},
    )
    print(f"\nFinal Full E1 checkpoint saved to {final_dir}")

    # 9. Final Frozen Test Set Evaluation
    print("\n" + "=" * 70)
    print("EVALUATING FINAL CHECKPOINT ON FROZEN GOLD TEST SET")
    print("=" * 70)

    # Re-verify frozen test manifest canonical hash before evaluating
    assert os.path.exists(test_manifest_path), f"Test manifest missing: {test_manifest_path}"
    test_raw_bytes = open(test_manifest_path, "rb").read()
    test_canonical_hash = compute_canonical_crlf_sha256(test_raw_bytes)
    assert test_canonical_hash == FROZEN_TEST_SHA256, "CRITICAL: Test manifest altered prior to evaluation!"

    df_test = pd.read_csv(test_manifest_path)
    test_sample_dicts = []
    for _, r in df_test.iterrows():
        test_sample_dicts.append({
            "audio_path": r.get("audio_path", ""),
            "transcript": str(r.get("transcript", "")),
            "sample_id": str(r.get("sample_id", "")),
        })

    model.eval()
    test_report = evaluate_model_on_manifest(
        model=model,
        processor=processor,
        samples=test_sample_dicts,
        device="cuda",
    )
    print(test_report.summary())

    total_wall_clock = time.time() - start_time
    final_metrics = {
        "status": "COMPLETED",
        "model": BASE_MODEL_ID,
        "revision": PINNED_MODEL_REVISION,
        "peft_version": PEFT_REQUIRED_VERSION,
        "total_optimizer_steps": global_step,
        "epochs_completed": epochs,
        "total_wall_clock_sec": total_wall_clock,
        "epoch_metrics": epoch_metrics,
        "best_val_loss": best_val_loss,
        "final_test_wer": test_report.micro_wer,
        "final_test_cer": test_report.micro_cer,
        "final_test_mean_latency_sec": test_report.mean_latency_sec,
        "final_test_mean_rtf": test_report.mean_rtf,
    }

    metrics_path = "reports/full_e1_training_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)
    print(f"Training metrics logged to {metrics_path}")

    print("\n" + "=" * 70)
    print("FULL E1 TRAINING COMPLETED SUCCESSFULLY")
    print(f"Final Test WER: {test_report.micro_wer * 100:.2f}% | CER: {test_report.micro_cer * 100:.2f}%")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Full E1 Vietnamese ASR Training")
    parser.add_argument("--device_check_only", action="store_true", help="Perform pre-run integrity check and exit")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--micro_batch_size", type=int, default=MICRO_BATCH_SIZE, help="Micro batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=GRAD_ACCUM_STEPS, help="Gradient accumulation steps")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Checkpoint output directory")
    args = parser.parse_args()

    check_res = preflight_check()

    if args.device_check_only:
        if not check_res["cuda_available"]:
            print("STATUS: FULL E1: ABORTED")
            print("Reason: CUDA execution environment unavailable on local workstation.")
            print("Execution must take place in the audited Google Colab / cloud Linux GPU runtime.")
            sys.exit(0)
        else:
            print("\nFULL E1 DEVICE CHECK: PASS. READY FOR TRAINING.")
            sys.exit(0)

    if not check_res["cuda_available"]:
        print("STATUS: FULL E1: ABORTED")
        print("Reason: CUDA execution environment unavailable on local workstation.")
        print("Execution must take place in the audited Google Colab / cloud Linux GPU runtime.")
        sys.exit(1)

    print("\nCUDA available. Launching Full E1 training execution...")
    run_full_e1_training(
        epochs=args.epochs,
        micro_batch_size=args.micro_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
