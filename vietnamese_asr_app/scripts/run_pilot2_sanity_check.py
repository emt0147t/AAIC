"""Pilot-2 Sanity Check: 200-Step LoRA Verification Script.

Verifies:
1. Loss trajectory decreases substantially across 200 optimizer steps.
2. Norm ratios ||Delta W|| / ||W_base|| after 200 steps.
3. Test predictions on the 9 test samples vs. E0 baseline to confirm adapter modulates decoding.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import torch
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model, PeftModel

sys.stdout.reconfigure(encoding="utf-8")

def main():
    print("=" * 70)
    print("STARTING PILOT-2 SANITY CHECK (200 OPTIMIZER STEPS)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model_id = "vinai/PhoWhisper-tiny"
    processor = WhisperProcessor.from_pretrained(model_id)
    base_model = WhisperForConditionalGeneration.from_pretrained(model_id)

    # 1. Evaluate E0 Baseline on 9 test samples first
    test_manifest_path = "manifests/test_manifest.csv"
    test_df = pd.read_csv(test_manifest_path)
    print(f"Loaded {len(test_df)} test samples from {test_manifest_path}")

    base_model.to(device)
    base_model.eval()

    e0_predictions = []
    print("\n--- Evaluating E0 Baseline on Test Samples ---")
    for idx, row in test_df.iterrows():
        audio_path = row["audio_path"]
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        feats = processor(y, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        prompt_ids = processor.get_decoder_prompt_ids(language="vi", task="transcribe")
        with torch.no_grad():
            gen_ids = base_model.generate(
                feats,
                forced_decoder_ids=prompt_ids,
                max_new_tokens=192,
                num_beams=1,
                do_sample=False,
            )
        pred = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        e0_predictions.append(pred)
        print(f"[{row['sample_id']}] E0: {pred[:60]}...")

    # 2. Setup LoRA Model for Pilot-2 Training
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    lora_model = get_peft_model(base_model, lora_config)
    lora_model.print_trainable_parameters()
    lora_model.train()

    optimizer = torch.optim.AdamW(lora_model.parameters(), lr=1e-3, weight_decay=0.01)

    # Load 22 clean real training samples
    train_manifest_path = "manifests/train_manifest.csv"
    train_df = pd.read_csv(train_manifest_path)
    print(f"\nLoaded {len(train_df)} training samples from {train_manifest_path}")

    # Pre-cache audio features and labels
    print("Pre-caching training tensors...")
    cached_data = []
    for idx, row in train_df.iterrows():
        y, _ = librosa.load(row["audio_path"], sr=16000, mono=True)
        feats = processor(y, sampling_rate=16000, return_tensors="pt").input_features[0]
        # Tokenize transcript
        labels = processor.tokenizer(row["transcript"]).input_ids
        cached_data.append((feats, labels))
    print(f"Successfully cached {len(cached_data)} training items.")

    # 3. Execute 200 Optimizer Steps
    total_steps = 200
    batch_size = 2
    log_interval = 20
    step = 0
    epoch = 0
    loss_history = []

    t_start = time.time()
    rng = np.random.default_rng(42)

    print("\n--- Training Loop (200 Steps) ---")
    while step < total_steps:
        epoch += 1
        indices = rng.permutation(len(cached_data))
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            b_feats = torch.stack([cached_data[idx][0] for idx in batch_indices]).to(device)
            # Pad labels
            max_len = max(len(cached_data[idx][1]) for idx in batch_indices)
            b_labels = torch.full((len(batch_indices), max_len), -100, dtype=torch.long)
            for b_idx, idx in enumerate(batch_indices):
                lbl = cached_data[idx][1]
                b_labels[b_idx, :len(lbl)] = torch.tensor(lbl, dtype=torch.long)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            out = lora_model(b_feats, labels=b_labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_model.parameters(), 1.0)
            optimizer.step()

            step += 1
            loss_val = loss.item()
            loss_history.append((step, loss_val))

            if step % log_interval == 0 or step == 1 or step == total_steps:
                elapsed = time.time() - t_start
                rate = elapsed / step
                print(f"Step {step:3d}/{total_steps} | Loss: {loss_val:.4f} | Elapsed: {elapsed:.1f}s ({rate:.3f} s/step)")

            if step >= total_steps:
                break

    train_time = time.time() - t_start
    print(f"\nTraining completed in {train_time:.2f} seconds ({train_time/60:.2f} minutes).")

    # 4. Measure LoRA delta weight norms
    print("\n--- Pilot-2 LoRA Delta Weight Norm Analysis ---")
    scaling = 16.0 / 8.0
    base_dict = dict(lora_model.base_model.named_parameters())
    ratios = []

    for name, module in lora_model.named_modules():
        if hasattr(module, "lora_A") and "default" in module.lora_A:
            wa = module.lora_A["default"].weight.data
            wb = module.lora_B["default"].weight.data
            dw = scaling * (wb @ wa)
            norm_dw = torch.linalg.norm(dw).item()

            base_w = module.base_layer.weight.data
            norm_w = torch.linalg.norm(base_w).item()
            ratio = norm_dw / norm_w
            ratios.append(ratio)
            print(f"{name}: ||dW||={norm_dw:.4f}, ||W||={norm_w:.4f}, ratio={ratio:.4e}")

    mean_ratio = float(np.mean(ratios))
    max_ratio = float(np.max(ratios))
    min_ratio = float(np.min(ratios))
    print(f"\nPilot-2 Delta Ratios: Mean={mean_ratio:.4e}, Max={max_ratio:.4e}, Min={min_ratio:.4e}")

    # 5. Evaluate Pilot-2 on 9 test samples
    print("\n--- Evaluating Pilot-2 Adapter on Test Samples ---")
    lora_model.eval()

    pilot2_predictions = []
    diff_count = 0

    for idx, row in test_df.iterrows():
        audio_path = row["audio_path"]
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        feats = processor(y, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        prompt_ids = processor.get_decoder_prompt_ids(language="vi", task="transcribe")
        with torch.no_grad():
            gen_ids = lora_model.generate(
                feats,
                forced_decoder_ids=prompt_ids,
                max_new_tokens=192,
                num_beams=1,
                do_sample=False,
            )
        pred = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        pilot2_predictions.append(pred)

        is_diff = pred != e0_predictions[idx]
        if is_diff:
            diff_count += 1
        status_str = "DIFFERENT" if is_diff else "IDENTICAL"
        print(f"[{row['sample_id']}] {status_str}:")
        print(f"   E0:     {e0_predictions[idx]}")
        print(f"   Pilot2: {pred}")

    print("\n" + "=" * 70)
    print(f"PILOT-2 SANITY CHECK RESULTS:")
    print(f"  Total Steps: {total_steps}")
    print(f"  Initial Loss: {loss_history[0][1]:.4f}")
    print(f"  Final Loss:   {loss_history[-1][1]:.4f}")
    print(f"  Loss Reduction: {loss_history[0][1] - loss_history[-1][1]:.4f} ({(1 - loss_history[-1][1]/loss_history[0][1])*100:.1f}%)")
    print(f"  Mean Delta Norm Ratio: {mean_ratio:.4e}")
    print(f"  Test Predictions Different from E0: {diff_count} / {len(test_df)}")
    print("=" * 70)

    # Save checkpoint to scratch / pilot2 directory for evidence
    os.makedirs("scratch/pilot2_checkpoint", exist_ok=True)
    lora_model.save_pretrained("scratch/pilot2_checkpoint")
    print("Saved Pilot-2 adapter to scratch/pilot2_checkpoint")

if __name__ == "__main__":
    main()
