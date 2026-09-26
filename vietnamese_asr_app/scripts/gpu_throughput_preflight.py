"""GPU Throughput & Parameter-Efficient LoRA Preflight Benchmark.

Executes a short (50-100 optimizer step) throughput and memory benchmark
under the frozen E1 training hyperparameters (micro_batch=4, grad_accum=8,
effective_batch=32, fp16 on CUDA) using representative real Vietnamese speech.

Measures:
- Device Model & CUDA Version
- Samples / Second & Seconds / Optimizer Step
- Actual Peak Allocated and Reserved VRAM (torch.cuda)
- Average Training Loss & Convergence
- Estimated Full E1 Multi-Epoch Duration
"""

import argparse
import gc
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from asr.audio import load_audio
from asr.normalization import normalize_vietnamese_text
from research.collator import WhisperDataCollatorWithPadding
from research.lora import build_whisper_lora_model, count_parameters


class BenchmarkDataset(Dataset):
    def __init__(self, manifest_csv: str, processor: WhisperProcessor):
        self.processor = processor
        self.df = pd.read_csv(manifest_csv)
        self.samples = []
        for idx, row in self.df.iterrows():
            self.samples.append({
                "sample_id": str(row["sample_id"]),
                "audio_path": str(row.get("audio_path_or_source_ref") or row.get("audio_path")),
                "transcript": str(row["transcript"]),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        sig, sr = load_audio(item["audio_path"], target_sr=16000)
        input_features = self.processor.feature_extractor(
            sig, sampling_rate=16000, return_tensors="pt"
        ).input_features[0]

        norm_text = normalize_vietnamese_text(item["transcript"])
        labels = self.processor.tokenizer(norm_text).input_ids

        return {
            "input_features": input_features,
            "labels": labels,
            "sample_id": item["sample_id"],
        }


def run_benchmark(
    manifest_csv: str = "manifests/full_train_manifest.csv",
    base_model_id: str = "vinai/PhoWhisper-tiny",
    max_optimizer_steps: int = 50,
    micro_batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    seed: int = 42,
    output_json: str = "reports/gpu_throughput_preflight.json",
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    effective_batch_size = micro_batch_size * gradient_accumulation_steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "CPU (Fallback)"
    cuda_version = torch.version.cuda if cuda_available else "N/A"

    print("=" * 75)
    print("PHASE 3.5 GPU THROUGHPUT PREFLIGHT BENCHMARK")
    print(f"Device:           {gpu_name}")
    print(f"CUDA Available:   {cuda_available} (Version: {cuda_version})")
    print(f"PyTorch Version:  {torch.__version__}")
    print(f"Base Model:       {base_model_id}")
    print(f"Micro Batch:      {micro_batch_size}")
    print(f"Grad Accum:       {gradient_accumulation_steps}")
    print(f"Effective Batch:  {effective_batch_size}")
    print(f"Target Opt Steps: {max_optimizer_steps}")
    print("=" * 75)

    processor = WhisperProcessor.from_pretrained(base_model_id)
    base_model = WhisperForConditionalGeneration.from_pretrained(base_model_id)
    model, param_counts = build_whisper_lora_model(
        base_model=base_model,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    model.to(device)

    print(f"Total Params:     {param_counts.total_params:,}")
    print(f"Trainable Params: {param_counts.trainable_params:,} ({param_counts.trainable_ratio_pct:.4f}%)")

    dataset = BenchmarkDataset(manifest_csv, processor)
    collator = WhisperDataCollatorWithPadding(processor=processor)

    # Use cycling sampler or DataLoader with repeat to sustain 50-100 optimizer steps
    # Total samples needed = max_optimizer_steps * effective_batch_size
    total_samples_needed = max_optimizer_steps * effective_batch_size
    indices = [i % len(dataset) for i in range(total_samples_needed)]
    subset = torch.utils.data.Subset(dataset, indices)
    dataloader = DataLoader(
        subset,
        batch_size=micro_batch_size,
        shuffle=False,
        collate_fn=collator,
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    total_micro_steps = max_optimizer_steps * gradient_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_micro_steps * 0.1),
        num_training_steps=total_micro_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=cuda_available)

    if cuda_available:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    model.train()
    step_losses = []
    optimizer_step_times = []
    start_bench_time = time.time()
    micro_step_count = 0
    opt_step_count = 0
    current_step_loss = 0.0
    step_start_time = time.time()

    for batch in dataloader:
        input_features = batch["input_features"].to(device)
        labels = batch["labels"].to(device)

        with torch.cuda.amp.autocast(enabled=cuda_available):
            outputs = model(input_features=input_features, labels=labels)
            loss = outputs.loss / gradient_accumulation_steps

        scaler.scale(loss).backward()
        current_step_loss += loss.item() * gradient_accumulation_steps
        micro_step_count += 1

        if micro_step_count % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            opt_step_count += 1
            step_duration = time.time() - step_start_time
            optimizer_step_times.append(step_duration)
            step_losses.append(current_step_loss)

            if opt_step_count % 10 == 0 or opt_step_count == 1:
                print(f"Step [{opt_step_count:3d}/{max_optimizer_steps:3d}] | Loss: {current_step_loss:.4f} | Latency: {step_duration:.3f}s")

            current_step_loss = 0.0
            step_start_time = time.time()

            if opt_step_count >= max_optimizer_steps:
                break

    total_bench_duration = time.time() - start_bench_time
    total_samples_processed = opt_step_count * effective_batch_size

    samples_per_sec = total_samples_processed / total_bench_duration if total_bench_duration > 0 else 0
    seconds_per_opt_step = np.mean(optimizer_step_times) if optimizer_step_times else 0

    if cuda_available:
        peak_allocated_bytes = torch.cuda.max_memory_allocated()
        peak_reserved_bytes = torch.cuda.max_memory_reserved()
        peak_allocated_mb = peak_allocated_bytes / (1024 * 1024)
        peak_reserved_mb = peak_reserved_bytes / (1024 * 1024)
    else:
        peak_allocated_bytes = 0
        peak_reserved_bytes = 0
        peak_allocated_mb = 0.0
        peak_reserved_mb = 0.0

    # Total full training duration estimate: 2,502 optimizer steps
    full_e1_steps = 2502
    estimated_full_duration_sec = full_e1_steps * seconds_per_opt_step
    estimated_full_duration_hours = estimated_full_duration_sec / 3600.0

    results = {
        "device": str(device),
        "gpu_name": gpu_name,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "pytorch_version": torch.__version__,
        "successful_optimizer_steps": opt_step_count,
        "total_samples_processed": total_samples_processed,
        "effective_batch_size": effective_batch_size,
        "total_benchmark_duration_sec": round(total_bench_duration, 3),
        "samples_per_sec": round(samples_per_sec, 3),
        "mean_seconds_per_optimizer_step": round(float(seconds_per_opt_step), 3),
        "peak_vram_allocated_mb": round(peak_allocated_mb, 2),
        "peak_vram_reserved_mb": round(peak_reserved_mb, 2),
        "mean_training_loss": round(float(np.mean(step_losses)), 4) if step_losses else 0.0,
        "loss_progression": [round(l, 4) for l in step_losses],
        "estimated_full_e1_steps": full_e1_steps,
        "estimated_full_duration_sec": round(estimated_full_duration_sec, 2),
        "estimated_full_duration_hours": round(estimated_full_duration_hours, 2),
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 75)
    print("BENCHMARK EXECUTION SUMMARY")
    print(f"Successful Steps:       {opt_step_count} / {max_optimizer_steps}")
    print(f"Throughput:             {samples_per_sec:.2f} samples/sec")
    print(f"Step Latency:           {seconds_per_opt_step:.3f} s / optimizer step")
    print(f"Peak VRAM Allocated:    {peak_allocated_mb:.1f} MB (Actual CUDA)")
    print(f"Peak VRAM Reserved:     {peak_reserved_mb:.1f} MB (Actual CUDA)")
    print(f"Average Loss:           {results['mean_training_loss']:.4f}")
    print(f"Estimated Full E1 Time: {estimated_full_duration_hours:.2f} hours (ESTIMATE)")
    print("=" * 75)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Throughput Preflight Benchmark")
    parser.add_argument("--steps", type=int, default=50, help="Optimizer steps to run (50-100)")
    parser.add_argument("--manifest", type=str, default="manifests/full_train_manifest.csv")
    parser.add_argument("--output", type=str, default="reports/gpu_throughput_preflight.json")
    args = parser.parse_args()

    run_benchmark(
        manifest_csv=args.manifest,
        max_optimizer_steps=args.steps,
        output_json=args.output,
    )
