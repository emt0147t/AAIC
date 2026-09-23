"""ASR Evaluation CLI.

Evaluates a model or fine-tuned checkpoint on a benchmark manifest CSV.

Usage:
    python scripts/evaluate.py --manifest manifests/test_manifest.csv --model_id vinai/PhoWhisper-small
"""

import argparse
import json
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from asr.audio import load_audio
from asr.inference import ASRInferenceEngine
from asr.normalization import normalize_vietnamese_text
from evaluation.metrics import compute_levenshtein_cer, compute_levenshtein_wer


def main():
    parser = argparse.ArgumentParser(description="Evaluate Vietnamese ASR Model on Manifest")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest CSV")
    parser.add_argument("--model_id", type=str, default="vinai/PhoWhisper-small", help="HF Model ID or local path")
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test samples")
    parser.add_argument("--output_csv", type=str, default="eval_results.csv", help="Where to save predictions")
    args = parser.parse_args()

    if not os.path.exists(args.manifest):
        print(f"Error: Manifest file '{args.manifest}' does not exist!")
        sys.exit(1)

    df = pd.read_csv(args.manifest)
    if args.limit:
        df = df.head(args.limit)

    print(f"Loaded {len(df)} samples from {args.manifest}")
    print(f"Initializing ASR engine with model: {args.model_id}...")
    engine = ASRInferenceEngine(model_id=args.model_id, preferred_device=args.device)
    print(f"Hardware: {engine.hardware_info.summary()}")

    predictions = []
    total_words = 0
    total_chars = 0
    wer_errors = 0
    cer_errors = 0

    for idx, row in df.iterrows():
        audio_path = row.get("audio_path") or row.get("path")
        ref_text = str(row.get("transcript") or row.get("sentence") or row.get("text") or "")
        
        if not audio_path or not os.path.exists(str(audio_path)):
            print(f"Skipping row {idx}: audio path not found: {audio_path}")
            continue

        audio_16k, _ = load_audio(str(audio_path))
        res = engine.transcribe(audio_16k)

        wer_res = compute_levenshtein_wer(ref_text, res.raw_text)
        cer_res = compute_levenshtein_cer(ref_text, res.raw_text)

        total_words += max(1, wer_res.total_reference_tokens)
        wer_errors += (wer_res.substitutions + wer_res.deletions + wer_res.insertions)

        total_chars += max(1, cer_res.total_reference_tokens)
        cer_errors += (cer_res.substitutions + cer_res.deletions + cer_res.insertions)

        predictions.append({
            "sample_id": row.get("sample_id", idx),
            "reference": ref_text,
            "hypothesis": res.raw_text,
            "wer": wer_res.error_rate,
            "cer": cer_res.error_rate,
            "latency_sec": res.latency_sec,
            "rtf": res.rtf,
        })

    overall_wer = wer_errors / max(1, total_words)
    overall_cer = cer_errors / max(1, total_chars)

    print("=" * 60)
    print("EVALUATION RESULTS")
    print(f"Total Samples: {len(predictions)}")
    print(f"Overall WER:   {overall_wer * 100:.2f}%")
    print(f"Overall CER:   {overall_cer * 100:.2f}%")
    print("=" * 60)

    df_out = pd.DataFrame(predictions)
    df_out.to_csv(args.output_csv, index=False)
    print(f"Detailed predictions written to: {args.output_csv}")


if __name__ == "__main__":
    main()
