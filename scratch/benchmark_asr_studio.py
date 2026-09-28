"""Benchmark ASR Inference Pipeline on the 9 ViMD test samples.
Measures latency breakdown, RTF, WER, CER, and RSS memory for both greedy and beam search.
"""

import os
import sys
import time
import json
import psutil
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("vietnamese_asr_app"))

from asr.audio import load_audio, validate_audio
from asr.inference import ASRInferenceEngine
from asr.normalization import normalize_vietnamese_text
from evaluation.metrics import compute_levenshtein_wer, compute_levenshtein_cer


def get_current_rss_mb():
    p = psutil.Process()
    return p.memory_info().rss / (1024 * 1024)


def run_benchmark():
    manifest_path = "vietnamese_asr_app/manifests/test_manifest.csv"
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest not found: {manifest_path}")
        return

    df = pd.read_csv(manifest_path)
    engine = ASRInferenceEngine(model_id="vinai/PhoWhisper-tiny", preferred_device="cpu")
    engine._ensure_loaded()

    # Hardware info
    hw = engine.hardware_info
    print(f"Hardware: {hw.summary()}")
    print(f"CPU count: {psutil.cpu_count(logical=True)} logical, {psutil.cpu_count(logical=False)} physical")
    print(f"Total RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")

    strategies = ["greedy", "beam_search"]
    results = {}

    for strat in strategies:
        print(f"\n================ Running Strategy: {strat} ================")
        num_beams = 4 if strat == "beam_search" else 1
        strat_results = []
        
        total_audio_sec = 0.0
        total_latency_sec = 0.0
        total_words_ref = 0
        total_word_errors = 0
        total_chars_ref = 0
        total_char_errors = 0

        stage_timings = {
            "load_audio": [],
            "validate_audio": [],
            "feature_extraction": [],
            "model_generate": [],
            "batch_decode": [],
            "normalization": [],
            "metrics": [],
        }

        for idx, row in df.iterrows():
            audio_path = row["audio_path"]
            ground_truth = str(row["transcript"])
            
            rss_before = get_current_rss_mb()
            
            # Stage 1: Load audio
            t0 = time.perf_counter()
            audio_data, sr = load_audio(audio_path, target_sr=16000)
            t_load = time.perf_counter() - t0
            
            # Stage 2: Validate audio
            t0 = time.perf_counter()
            val_res = validate_audio(audio_data, sample_rate=sr)
            t_val = time.perf_counter() - t0
            
            audio_dur = len(audio_data) / sr
            
            # Stage 3: Feature extraction
            t0 = time.perf_counter()
            features = engine._processor.feature_extractor(
                audio_data,
                sampling_rate=16000,
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_features = features.input_features.to(engine._hw.device, dtype=torch.float32)
            attention_mask = features.attention_mask.to(engine._hw.device)
            t_feat = time.perf_counter() - t0

            # Decoder prompt
            start_token = getattr(engine._model.config, "decoder_start_token_id", 50258)
            prompt_ids = engine._processor.get_decoder_prompt_ids(language="vi", task="transcribe")
            token_list = [start_token] + [token_id for _, token_id in prompt_ids]
            decoder_input_ids = torch.tensor([token_list], device=engine._hw.device, dtype=torch.long)

            gen_kwargs = {
                "language": "vi",
                "task": "transcribe",
                "decoder_input_ids": decoder_input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": 192,
                "num_beams": num_beams,
                "do_sample": False,
            }

            # Stage 4: Model Generate
            t0 = time.perf_counter()
            with torch.inference_mode():
                predicted_ids = engine._model.generate(input_features, **gen_kwargs)
            t_gen = time.perf_counter() - t0

            # Stage 5: Batch Decode
            t0 = time.perf_counter()
            raw_text = engine._processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
            t_dec = time.perf_counter() - t0

            # Stage 6: Normalization
            t0 = time.perf_counter()
            norm_hyp = normalize_vietnamese_text(raw_text)
            norm_ref = normalize_vietnamese_text(ground_truth)
            t_norm = time.perf_counter() - t0

            # Stage 7: Metric
            t0 = time.perf_counter()
            wer_res = compute_levenshtein_wer(norm_ref, norm_hyp)
            cer_res = compute_levenshtein_cer(norm_ref, norm_hyp)
            t_metric = time.perf_counter() - t0

            rss_after = get_current_rss_mb()
            
            sample_total_latency = t_load + t_val + t_feat + t_gen + t_dec + t_norm + t_metric
            rtf = sample_total_latency / audio_dur

            stage_timings["load_audio"].append(t_load)
            stage_timings["validate_audio"].append(t_val)
            stage_timings["feature_extraction"].append(t_feat)
            stage_timings["model_generate"].append(t_gen)
            stage_timings["batch_decode"].append(t_dec)
            stage_timings["normalization"].append(t_norm)
            stage_timings["metrics"].append(t_metric)

            total_audio_sec += audio_dur
            total_latency_sec += sample_total_latency
            
            # Accumulate Levenshtein counts
            total_words_ref += wer_res.total_reference_tokens
            total_word_errors += wer_res.substitutions + wer_res.deletions + wer_res.insertions
            total_chars_ref += cer_res.total_reference_tokens
            total_char_errors += cer_res.substitutions + cer_res.deletions + cer_res.insertions

            item = {
                "sample_id": row["sample_id"],
                "duration_sec": round(audio_dur, 3),
                "total_latency_sec": round(sample_total_latency, 3),
                "rtf": round(rtf, 3),
                "t_gen": round(t_gen, 3),
                "wer": round(wer_res.error_rate, 4),
                "cer": round(cer_res.error_rate, 4),
                "rss_peak_mb": round(rss_after, 1),
                "ref_norm": norm_ref,
                "hyp_norm": norm_hyp,
            }
            strat_results.append(item)
            print(f"Sample {row['sample_id']} ({audio_dur:.2f}s) -> Latency: {sample_total_latency:.2f}s (RTF: {rtf:.2f}), WER: {wer_res.error_rate*100:.2f}%, CER: {cer_res.error_rate*100:.2f}%")

        corpus_wer = total_word_errors / max(total_words_ref, 1)
        corpus_cer = total_char_errors / max(total_chars_ref, 1)
        corpus_rtf = total_latency_sec / max(total_audio_sec, 1e-4)

        avg_stages = {k: float(np.mean(v)) for k, v in stage_timings.items()}

        results[strat] = {
            "samples": strat_results,
            "corpus_wer": round(corpus_wer, 4),
            "corpus_cer": round(corpus_cer, 4),
            "corpus_rtf": round(corpus_rtf, 4),
            "total_audio_sec": round(total_audio_sec, 3),
            "total_latency_sec": round(total_latency_sec, 3),
            "average_stage_latency_sec": {k: round(v, 4) for k, v in avg_stages.items()},
        }

        print(f"\n--- {strat.upper()} SUMMARY ---")
        print(f"Total Audio Duration: {total_audio_sec:.2f}s")
        print(f"Total Pipeline Latency: {total_latency_sec:.2f}s")
        print(f"Corpus RTF: {corpus_rtf:.3f}")
        print(f"Corpus WER: {corpus_wer*100:.2f}% ({total_word_errors}/{total_words_ref} words)")
        print(f"Corpus CER: {corpus_cer*100:.2f}% ({total_char_errors}/{total_chars_ref} chars)")
        print("Average Stage Breakdown (seconds per sample):")
        for k, v in avg_stages.items():
            print(f"  {k:20s}: {v*1000:7.2f} ms ({v/np.mean([s['total_latency_sec'] for s in strat_results])*100:.1f}%)")

    # Save benchmark report to JSON
    out_file = "vietnamese_asr_app/exports/asr_inference_benchmark.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWrote benchmark data to {out_file}")


if __name__ == "__main__":
    run_benchmark()
