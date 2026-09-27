"""
ViMD Full Test Set Evaluation Protocol (2,026 Samples)
======================================================
Protocol designed for cloud GPU evaluation (e.g., Google Colab / Kaggle T4 / A100).
Evaluates PhoWhisper base model and fine-tuned LoRA adapters on the full 2,026-sample
test partition of nguyendv02/ViMD_Dataset with bootstrap confidence interval calculation.

Status on Local CPU Workstation: NOT_EXECUTED (9.3h estimated CPU latency and multi-GB
parquet sharding; queued for turnkey Cloud GPU execution).
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import soundfile as sf
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

from vietnamese_asr_app.evaluation.metrics import (
    compute_levenshtein_wer,
    compute_levenshtein_cer,
    normalize_vietnamese_text,
)

def evaluate_full_vimd_test(
    model_name_or_path="vinai/PhoWhisper-tiny",
    adapter_path=None,
    split="test",
    output_csv="reports/vimd_full_test_results.csv",
    device="cuda" if torch.cuda.is_available() else "cpu",
    batch_size=8,
    n_bootstrap=1000
):
    print(f"Device: {device}")
    print(f"Base model: {model_name_or_path}")
    print(f"Adapter: {adapter_path if adapter_path else 'None (Zero-shot base)'}")
    
    processor = WhisperProcessor.from_pretrained(model_name_or_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path)
    
    if adapter_path and os.path.exists(adapter_path):
        print(f"Loading LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        
    model = model.to(device)
    model.eval()
    
    print("Loading nguyendv02/ViMD_Dataset (streaming=False for full test split)...")
    dataset = load_dataset("nguyendv02/ViMD_Dataset", split=split)
    n_total = len(dataset)
    print(f"Total test utterances loaded: {n_total}")
    
    results = []
    start_time = time.time()
    
    for idx, item in enumerate(dataset):
        audio_array = item["audio"]["array"]
        sr = item["audio"]["sampling_rate"]
        ref_text = item.get("transcription", item.get("transcript", ""))
        ref_norm = normalize_vietnamese_text(ref_text)
        
        inputs = processor(audio_array, sampling_rate=sr, return_tensors="pt")
        input_features = inputs.input_features.to(device)
        
        with torch.no_grad():
            gen_ids = model.generate(
                input_features,
                max_new_tokens=225,
                language="vi",
                task="transcribe"
            )
            
        pred_raw = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        pred_norm = normalize_vietnamese_text(pred_raw)
        
        wer_res = compute_levenshtein_wer(ref_norm, pred_norm)
        cer_res = compute_levenshtein_cer(ref_norm, pred_norm)
        
        n_words = len(ref_norm.split())
        n_chars = len(ref_norm)
        
        results.append({
            "index": idx,
            "ref_words": n_words,
            "wer_errors": wer_res.substitutions + wer_res.deletions + wer_res.insertions,
            "ref_chars": n_chars,
            "cer_errors": cer_res.substitutions + cer_res.deletions + cer_res.insertions,
        })
        
        if (idx + 1) % 100 == 0 or (idx + 1) == n_total:
            elapsed = time.time() - start_time
            print(f"[{idx+1}/{n_total}] Processed in {elapsed:.1f}s ({(idx+1)/elapsed:.2f} utt/s)...")
            
    df = pd.DataFrame(results)
    tot_words = df["ref_words"].sum()
    tot_wer_errors = df["wer_errors"].sum()
    tot_chars = df["ref_chars"].sum()
    tot_cer_errors = df["cer_errors"].sum()
    
    micro_wer = (tot_wer_errors / tot_words) * 100
    micro_cer = (tot_cer_errors / tot_chars) * 100
    
    print("\n================ EVALUATION SUMMARY ================")
    print(f"Total Test Utterances: {n_total}")
    print(f"Total Words: {tot_words:,} | Total Chars: {tot_chars:,}")
    print(f"Micro WER: {micro_wer:.2f}% | Micro CER: {micro_cer:.2f}%")
    
    # Bootstrap CI
    print(f"\nComputing {n_bootstrap} bootstrap resamples for 95% Confidence Interval...")
    np.random.seed(42)
    boot_wers, boot_cers = [], []
    for _ in range(n_bootstrap):
        sampled = df.sample(n=len(df), replace=True)
        boot_wers.append(sampled["wer_errors"].sum() / sampled["ref_words"].sum() * 100)
        boot_cers.append(sampled["cer_errors"].sum() / sampled["ref_chars"].sum() * 100)
        
    wer_ci = (np.percentile(boot_wers, 2.5), np.percentile(boot_wers, 97.5))
    cer_ci = (np.percentile(boot_cers, 2.5), np.percentile(boot_cers, 97.5))
    
    print(f"WER 95% CI: [{wer_ci[0]:.2f}%, {wer_ci[1]:.2f}%]")
    print(f"CER 95% CI: [{cer_ci[0]:.2f}%, {cer_ci[1]:.2f}%]")
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Per-utterance evaluation saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, default=None, help="Path to LoRA adapter")
    args = parser.parse_args()
    evaluate_full_vimd_test(adapter_path=args.adapter)
