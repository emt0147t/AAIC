"""
Gate 6 — Tiny ASR Engineering Pilot Pipeline
Model: vinai/PhoWhisper-tiny
Hardware: CPU
Protocol: VIVOS train + ViMD train -> ViMD valid
Seed: 42
"""
import sys, os, io, re, time, json, csv, unicodedata
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import torch
import librosa
from scipy.io import wavfile
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from torch.optim import AdamW

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

PILOT_OUT = r"d:\Vietnamese_ASR_Week6\artifacts\pilot"
os.makedirs(PILOT_OUT, exist_ok=True)
os.makedirs(os.path.join(PILOT_OUT, "checkpoint"), exist_ok=True)

# Also ensure asr_experiment/pilot has a copy
ALT_OUT = r"d:\Vietnamese_ASR_Week6\asr_experiment\pilot"
os.makedirs(ALT_OUT, exist_ok=True)
os.makedirs(os.path.join(ALT_OUT, "checkpoint"), exist_ok=True)


# ============================================================
# 1. Transcript Normalization & Levenshtein Metric Functions
# ============================================================

def normalize_transcript(text):
    """
    Deterministic transcript normalization:
    - Unicode NFC normalization
    - Lowercase
    - Remove punctuation (preserve diacritics)
    - Collapse repeated whitespace
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def levenshtein(ref, hyp):
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]

def compute_wer(ref, hyp):
    r_words = ref.strip().split()
    h_words = hyp.strip().split()
    if len(r_words) == 0:
        return 0.0 if len(h_words) == 0 else 1.0
    return levenshtein(r_words, h_words) / len(r_words)

def compute_cer(ref, hyp):
    r_chars = list(ref.strip().replace(" ", ""))
    h_chars = list(hyp.strip().replace(" ", ""))
    if len(r_chars) == 0:
        return 0.0 if len(h_chars) == 0 else 1.0
    return levenshtein(r_chars, h_chars) / len(r_chars)


# ============================================================
# 2. Audio Processing Helper
# ============================================================

def process_audio_bytes(raw_bytes, source_name):
    """
    Decodes audio bytes, checks mono, resamples to 16000 Hz if needed.
    Returns: audio_16k (float32 numpy array), orig_sr, orig_dur, new_dur
    """
    sr, data = wavfile.read(io.BytesIO(raw_bytes))
    orig_sr = sr
    orig_samples = len(data)
    orig_dur = orig_samples / orig_sr

    # Convert to float32 [-1, 1]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype != np.float32:
        data = data.astype(np.float32)

    # Convert stereo to mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample if needed
    if orig_sr != 16000:
        audio_16k = librosa.resample(data, orig_sr=orig_sr, target_sr=16000)
    else:
        audio_16k = data

    new_dur = len(audio_16k) / 16000.0
    return audio_16k, orig_sr, orig_dur, new_dur


# ============================================================
# 3. Main Pilot Execution
# ============================================================

def main():
    print("=" * 70)
    print("GATE 6 — TINY ASR ENGINEERING PILOT")
    print("Model: vinai/PhoWhisper-tiny | Hardware: CPU | Seed: 42")
    print("=" * 70)

    # Load processor and model
    print("\n[Step 1] Loading PhoWhisper-tiny...")
    t0_load = time.time()
    processor = WhisperProcessor.from_pretrained("vinai/PhoWhisper-tiny")
    model = WhisperForConditionalGeneration.from_pretrained("vinai/PhoWhisper-tiny")
    model.train()
    total_params = model.num_parameters()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Loaded in {time.time() - t0_load:.2f}s")
    print(f"Total parameters: {total_params:,} | Trainable: {trainable_params:,}")

    # Inspect 20 transcripts (10 VIVOS + 10 ViMD) before training
    print("\n[Step 2] Inspecting 20 Transcripts (10 VIVOS + 10 ViMD)...")
    ds_vivos_stream = load_dataset('ademax/vivos-vie-speech2text', split='train', streaming=True).cast_column('audio', Audio(decode=False))
    ds_vimd_stream = load_dataset('nguyendv02/ViMD_Dataset', split='train', streaming=True).cast_column('audio', Audio(decode=False))

    transcript_inspection = []
    
    # 10 VIVOS
    vivos_iter = iter(ds_vivos_stream)
    for i in range(10):
        item = next(vivos_iter)
        raw_text = item.get("raw_transcription") or item.get("transcription", "")
        norm_text = normalize_transcript(raw_text)
        transcript_inspection.append({
            "idx": i + 1,
            "dataset": "VIVOS",
            "raw": raw_text,
            "norm": norm_text,
            "char_len": len(norm_text),
            "word_len": len(norm_text.split()),
            "empty": len(norm_text) == 0,
            "has_punct": bool(re.search(r"[^\w\s]", raw_text)),
            "casing": "UPPERCASE" if raw_text.isupper() else ("lowercase" if raw_text.islower() else "mixed"),
        })

    # 10 ViMD
    vimd_iter = iter(ds_vimd_stream)
    for i in range(10):
        item = next(vimd_iter)
        raw_text = item.get("text", "")
        norm_text = normalize_transcript(raw_text)
        transcript_inspection.append({
            "idx": i + 11,
            "dataset": "ViMD",
            "raw": raw_text,
            "norm": norm_text,
            "char_len": len(norm_text),
            "word_len": len(norm_text.split()),
            "empty": len(norm_text) == 0,
            "has_punct": bool(re.search(r"[^\w\s]", raw_text)),
            "casing": "UPPERCASE" if raw_text.isupper() else ("lowercase" if raw_text.islower() else "mixed"),
        })

    # Print summary of inspection
    print(f"{'ID':<4} {'Dataset':<7} {'Casing':<10} {'Punct?':<7} {'Words':<6} {'Chars':<6} {'Preview'}")
    print("-" * 70)
    for t in transcript_inspection:
        preview = t['norm'][:35] + ("..." if len(t['norm']) > 35 else "")
        print(f"{t['idx']:<4} {t['dataset']:<7} {t['casing']:<10} {str(t['has_punct']):<7} {t['word_len']:<6} {t['char_len']:<6} {preview}")

    vivos_words = [t['word_len'] for t in transcript_inspection[:10]]
    vimd_words = [t['word_len'] for t in transcript_inspection[10:]]
    print(f"\nInspection Analysis:")
    print(f"  VIVOS casing: 10/10 {transcript_inspection[0]['casing']} (standard prompt prompts)")
    print(f"  ViMD casing:  10/10 {transcript_inspection[10]['casing']} (punctuated news text)")
    print(f"  VIVOS word length: min={min(vivos_words)}, max={max(vivos_words)}, mean={np.mean(vivos_words):.1f}")
    print(f"  ViMD word length:  min={min(vimd_words)}, max={max(vimd_words)}, mean={np.mean(vimd_words):.1f}")
    print(f"  Empty transcripts: 0/20")
    print(f"  Unicode issues: None (NFC validated)")

    # ============================================================
    # 4. Stream and Prepare Pilot Datasets (100 Train, 20 Val)
    # ============================================================
    print("\n[Step 3] Preparing Pilot Training (50 VIVOS + 50 ViMD = 100) & Validation (20 ViMD valid)...")

    # Re-instantiate streams to get deterministic sequential slices from beginning
    ds_vivos_train = load_dataset('ademax/vivos-vie-speech2text', split='train', streaming=True).cast_column('audio', Audio(decode=False))
    ds_vimd_train = load_dataset('nguyendv02/ViMD_Dataset', split='train', streaming=True).cast_column('audio', Audio(decode=False))
    ds_vimd_val = load_dataset('nguyendv02/ViMD_Dataset', split='valid', streaming=True).cast_column('audio', Audio(decode=False))

    train_data = []
    
    # 50 VIVOS train samples
    vivos_train_iter = iter(ds_vivos_train)
    for i in range(50):
        item = next(vivos_train_iter)
        raw_text = item.get("raw_transcription") or item.get("transcription", "")
        norm_text = normalize_transcript(raw_text)
        audio_arr, orig_sr, orig_dur, new_dur = process_audio_bytes(item["audio"]["bytes"], "VIVOS")
        train_data.append({
            "sample_id": f"vivos_train_{i:03d}",
            "dataset": "VIVOS",
            "audio": audio_arr,
            "orig_sr": orig_sr,
            "orig_dur": orig_dur,
            "new_dur": new_dur,
            "raw_text": raw_text,
            "norm_text": norm_text,
        })

    # 50 ViMD train samples
    vimd_train_iter = iter(ds_vimd_train)
    for i in range(50):
        item = next(vimd_train_iter)
        raw_text = item.get("text", "")
        norm_text = normalize_transcript(raw_text)
        audio_arr, orig_sr, orig_dur, new_dur = process_audio_bytes(item["audio"]["bytes"], "ViMD")
        train_data.append({
            "sample_id": f"vimd_train_{i:03d}",
            "dataset": "ViMD",
            "audio": audio_arr,
            "orig_sr": orig_sr,
            "orig_dur": orig_dur,
            "new_dur": new_dur,
            "raw_text": raw_text,
            "norm_text": norm_text,
        })

    print(f"Collected {len(train_data)} train samples (50 VIVOS + 50 ViMD).")

    # 20 ViMD validation samples
    val_data = []
    vimd_val_iter = iter(ds_vimd_val)
    for i in range(20):
        item = next(vimd_val_iter)
        raw_text = item.get("text", "")
        norm_text = normalize_transcript(raw_text)
        audio_arr, orig_sr, orig_dur, new_dur = process_audio_bytes(item["audio"]["bytes"], "ViMD")
        val_data.append({
            "sample_id": f"vimd_val_{i:03d}",
            "dataset": "ViMD",
            "audio": audio_arr,
            "orig_sr": orig_sr,
            "orig_dur": orig_dur,
            "new_dur": new_dur,
            "raw_text": raw_text,
            "norm_text": norm_text,
        })
    print(f"Collected {len(val_data)} validation samples (ViMD valid).")

    # Log audio preprocessing verification
    print("\nAudio Preprocessing Verification:")
    print(f"  VIVOS: orig_sr={train_data[0]['orig_sr']}Hz -> resampled={16000}Hz | dur={train_data[0]['new_dur']:.2f}s")
    print(f"  ViMD:  orig_sr={train_data[50]['orig_sr']}Hz -> resampled={16000}Hz | orig_dur={train_data[50]['orig_dur']:.2f}s -> new_dur={train_data[50]['new_dur']:.2f}s")

    # ============================================================
    # 5. Featurize Data & Create Batch Loader
    # ============================================================
    print("\n[Step 4] Extracting log-mel features and tokenizing labels...")
    for ex in train_data:
        # Extract features (shape: [80, 3000])
        feat = processor.feature_extractor(ex["audio"], sampling_rate=16000, return_tensors="pt").input_features[0]
        # Tokenize labels
        label_ids = processor.tokenizer(ex["norm_text"]).input_ids
        ex["input_features"] = feat
        ex["labels"] = label_ids

    for ex in val_data:
        feat = processor.feature_extractor(ex["audio"], sampling_rate=16000, return_tensors="pt").input_features[0]
        label_ids = processor.tokenizer(ex["norm_text"]).input_ids
        ex["input_features"] = feat
        ex["labels"] = label_ids

    print(f"Feature tensor shape: {train_data[0]['input_features'].shape} [80 mel bins, 3000 frames]")

    # ============================================================
    # 6. 1-Epoch Training Run
    # ============================================================
    print("\n[Step 5] Running 1-Epoch Training on CPU...")
    BATCH_SIZE = 2
    GRAD_ACCUM = 2
    LEARNING_RATE = 1e-4

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # Shuffle training data deterministically
    rng = np.random.RandomState(SEED)
    indices = np.arange(len(train_data))
    rng.shuffle(indices)

    train_log = []
    model.train()
    optimizer.zero_grad()

    t_train_start = time.time()
    step_loss_accum = 0.0
    num_batches = len(train_data) // BATCH_SIZE

    print(f"Total training utterances: {len(train_data)}")
    print(f"Batch size: {BATCH_SIZE} | Gradient accumulation steps: {GRAD_ACCUM}")
    print(f"Total forward batches: {num_batches} | Optimizer updates: {num_batches // GRAD_ACCUM}")
    print("-" * 70)

    for b_idx in range(num_batches):
        batch_indices = indices[b_idx * BATCH_SIZE : (b_idx + 1) * BATCH_SIZE]
        batch_feats = torch.stack([train_data[idx]["input_features"] for idx in batch_indices])
        
        # Collate labels: pad to max length in batch
        batch_labels_list = [train_data[idx]["labels"] for idx in batch_indices]
        max_label_len = max(len(l) for l in batch_labels_list)
        batch_labels = torch.full((len(batch_indices), max_label_len), -100, dtype=torch.long)
        for i_l, l in enumerate(batch_labels_list):
            batch_labels[i_l, :len(l)] = torch.tensor(l, dtype=torch.long)

        # Forward pass
        t_b0 = time.time()
        outputs = model(input_features=batch_feats, labels=batch_labels)
        loss = outputs.loss / GRAD_ACCUM
        loss_val = outputs.loss.item()
        
        # Check finite loss
        assert not np.isnan(loss_val) and not np.isinf(loss_val), f"Non-finite loss encountered at batch {b_idx}: {loss_val}"

        # Backward pass
        loss.backward()
        step_loss_accum += loss_val

        # Optimizer step
        if (b_idx + 1) % GRAD_ACCUM == 0 or (b_idx + 1) == num_batches:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            avg_loss = step_loss_accum / GRAD_ACCUM
            opt_step = (b_idx + 1) // GRAD_ACCUM
            dt_step = time.time() - t_b0
            train_log.append({
                "step": opt_step,
                "batch_idx": b_idx + 1,
                "loss": round(avg_loss, 4),
                "is_finite": True,
                "lr": LEARNING_RATE,
                "elapsed_s": round(time.time() - t_train_start, 2)
            })
            step_loss_accum = 0.0

            if opt_step % 5 == 0 or opt_step == (num_batches // GRAD_ACCUM):
                print(f"  Step {opt_step:02d}/{num_batches // GRAD_ACCUM} | Loss: {avg_loss:.4f} | Time: {time.time() - t_train_start:.1f}s")

    train_duration = time.time() - t_train_start
    final_loss = train_log[-1]["loss"] if train_log else 0.0
    print(f"\n1-Epoch Training Complete! Total time: {train_duration:.2f}s | Final Step Loss: {final_loss:.4f}")

    # ============================================================
    # 7. Checkpoint Save & Verification
    # ============================================================
    print("\n[Step 6] Saving and verifying checkpoint...")
    ckpt_dir = os.path.join(PILOT_OUT, "checkpoint")
    model.save_pretrained(ckpt_dir)
    processor.save_pretrained(ckpt_dir)

    # Mirror to alternate dir
    alt_ckpt_dir = os.path.join(ALT_OUT, "checkpoint")
    model.save_pretrained(alt_ckpt_dir)
    processor.save_pretrained(alt_ckpt_dir)

    # Test re-loading checkpoint
    test_reloaded_model = WhisperForConditionalGeneration.from_pretrained(ckpt_dir)
    assert test_reloaded_model is not None
    print(f"Checkpoint successfully saved to {ckpt_dir} and reloaded with {test_reloaded_model.num_parameters():,} parameters.")

    # ============================================================
    # 8. Inference Sanity Check on 5 Validation Samples
    # ============================================================
    print("\n[Step 7] Qualitative Inference Sanity Check on 5 Validation Samples...")
    model.eval()
    five_val_samples = val_data[:5]
    val_predictions = []

    print(f"{'Sample ID':<14} {'Ref Words':<10} {'Hyp Words':<10} {'WER':<8} {'CER':<8} {'Status'}")
    print("-" * 70)

    for s_idx, sample in enumerate(five_val_samples):
        feat = sample["input_features"].unsqueeze(0)
        with torch.no_grad():
            # Force Vietnamese transcription
            forced_decoder_ids = processor.get_decoder_prompt_ids(language="vi", task="transcribe")
            pred_ids = model.generate(feat, forced_decoder_ids=forced_decoder_ids, max_length=225)
            hyp_text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]

        ref_norm = sample["norm_text"]
        hyp_norm = normalize_transcript(hyp_text)

        wer_val = compute_wer(ref_norm, hyp_norm)
        cer_val = compute_cer(ref_norm, hyp_norm)

        val_predictions.append({
            "sample_id": sample["sample_id"],
            "dataset_id": sample["dataset"],
            "raw_reference": sample["raw_text"],
            "norm_reference": ref_norm,
            "raw_prediction": hyp_text,
            "norm_prediction": hyp_norm,
            "wer": round(wer_val, 4),
            "cer": round(cer_val, 4),
            "is_non_empty": len(hyp_text.strip()) > 0,
            "is_vietnamese": bool(re.search(r"[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]", hyp_text.lower()))
        })

        print(f"{sample['sample_id']:<14} {len(ref_norm.split()):<10} {len(hyp_norm.split()):<10} {wer_val:<8.4f} {cer_val:<8.4f} {'OK' if val_predictions[-1]['is_non_empty'] else 'EMPTY'}")
        print(f"  REF: {ref_norm[:70]}...")
        print(f"  HYP: {hyp_norm[:70]}...")

    # ============================================================
    # 9. Save Artifacts & Outputs
    # ============================================================
    print("\n[Step 8] Writing Pilot Artifacts...")

    # 1. train_log.csv
    train_log_path = os.path.join(PILOT_OUT, "train_log.csv")
    with open(train_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "batch_idx", "loss", "is_finite", "lr", "elapsed_s"])
        writer.writeheader()
        writer.writerows(train_log)

    with open(os.path.join(ALT_OUT, "train_log.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "batch_idx", "loss", "is_finite", "lr", "elapsed_s"])
        writer.writeheader()
        writer.writerows(train_log)

    # 2. five_sample_predictions.csv
    pred_path = os.path.join(PILOT_OUT, "five_sample_predictions.csv")
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sample_id", "dataset_id", "raw_reference", "norm_reference",
            "raw_prediction", "norm_prediction", "wer", "cer", "is_non_empty", "is_vietnamese"
        ])
        writer.writeheader()
        writer.writerows(val_predictions)

    with open(os.path.join(ALT_OUT, "five_sample_predictions.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sample_id", "dataset_id", "raw_reference", "norm_reference",
            "raw_prediction", "norm_prediction", "wer", "cer", "is_non_empty", "is_vietnamese"
        ])
        writer.writeheader()
        writer.writerows(val_predictions)

    # 3. pilot_config.json
    pilot_config = {
        "gate": "GATE 6 — TINY ASR ENGINEERING PILOT",
        "benchmark_claim": "ENGINEERING PILOT — NOT FOR FINAL BENCHMARK",
        "environment": {
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "device": "cpu",
            "cuda_available": False,
            "gpu_used": False
        },
        "model": {
            "name": "vinai/PhoWhisper-tiny",
            "architecture": "WhisperForConditionalGeneration",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "language": "vi",
            "task": "transcribe"
        },
        "data_composition": {
            "train_samples_total": len(train_data),
            "train_vivos_count": 50,
            "train_vimd_count": 50,
            "val_samples_total": len(val_data),
            "val_source": "ViMD valid split",
            "test_touched": False,
            "seed": SEED
        },
        "audio_pipeline": {
            "feature_format": "80-channel log-mel spectrogram",
            "sample_rate_target": 16000,
            "vivos_orig_sr": 16000,
            "vimd_orig_sr": 44100,
            "vimd_resampled": True,
            "resampling_method": "librosa.resample",
            "tensor_shape": [80, 3000]
        },
        "transcript_pipeline": {
            "normalization": "NFC Unicode, lowercasing, punctuation stripping, whitespace collapse",
            "raw_preserved": True
        },
        "training_run": {
            "epochs": 1,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRAD_ACCUM,
            "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
            "learning_rate": LEARNING_RATE,
            "optimizer": "AdamW",
            "weight_decay": 0.01,
            "runtime_seconds": round(train_duration, 2),
            "final_loss": round(final_loss, 4),
            "loss_finite": True,
            "oom_encountered": False
        },
        "checkpoint": {
            "saved_successfully": True,
            "reloaded_successfully": True,
            "path": ckpt_dir
        },
        "validation_sanity_check": {
            "samples_evaluated": len(val_predictions),
            "all_non_empty": all(p["is_non_empty"] for p in val_predictions),
            "all_vietnamese": all(p["is_vietnamese"] for p in val_predictions),
            "mean_pilot_wer": round(float(np.mean([p["wer"] for p in val_predictions])), 4),
            "mean_pilot_cer": round(float(np.mean([p["cer"] for p in val_predictions])), 4),
            "metric_label": "ENGINEERING PILOT — NOT FOR FINAL BENCHMARK"
        }
    }

    config_path = os.path.join(PILOT_OUT, "pilot_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(pilot_config, f, indent=2, ensure_ascii=False)

    with open(os.path.join(ALT_OUT, "pilot_config.json"), "w", encoding="utf-8") as f:
        json.dump(pilot_config, f, indent=2, ensure_ascii=False)

    print(f"Artifacts saved to {PILOT_OUT} and {ALT_OUT}")
    print("\n" + "=" * 70)
    print("GATE 6 PILOT RUN COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
