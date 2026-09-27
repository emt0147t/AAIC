import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import torch
import soundfile as sf
import pandas as pd
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from vietnamese_asr_app.evaluation.metrics import (
    compute_levenshtein_wer,
    compute_levenshtein_cer,
    normalize_vietnamese_text,
)

def main():
    device = "cpu"
    model_id = "vinai/PhoWhisper-tiny"
    processor = WhisperProcessor.from_pretrained(model_id)
    base_model = WhisperForConditionalGeneration.from_pretrained(model_id)
    
    ckpt_path = "vietnamese_asr_app/scratch/pilot2_checkpoint"
    print(f"Loading Pilot-2 adapter from {ckpt_path}...")
    pilot2_model = PeftModel.from_pretrained(base_model, ckpt_path)
    pilot2_model.eval()

    manifest_df = pd.read_csv("vietnamese_asr_app/manifests/test_manifest.csv")
    prompt_ids = processor.get_decoder_prompt_ids(language="vi", task="transcribe")

    results = []
    print("\n--- Running Pilot-2 Inference on 9-Sample Test Set ---")
    for idx, row in manifest_df.iterrows():
        audio, sr = sf.read(row["audio_path"])
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            gen_ids = pilot2_model.generate(
                inputs.input_features,
                forced_decoder_ids=prompt_ids,
                max_new_tokens=225,
                num_beams=1,
                do_sample=False,
            )
        pred_raw = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        pred_norm = normalize_vietnamese_text(pred_raw)
        ref_norm = normalize_vietnamese_text(row["transcript"])
        
        wer_res = compute_levenshtein_wer(ref_norm, pred_norm)
        cer_res = compute_levenshtein_cer(ref_norm, pred_norm)
        
        n_words = len(ref_norm.split())
        n_chars = len(ref_norm)
        
        wer_err = wer_res.substitutions + wer_res.deletions + wer_res.insertions
        cer_err = cer_res.substitutions + cer_res.deletions + cer_res.insertions
        
        results.append({
            "sample_id": row["sample_id"],
            "ref": ref_norm,
            "pred": pred_norm,
            "n_words": n_words,
            "wer_subs": wer_res.substitutions,
            "wer_dels": wer_res.deletions,
            "wer_ins": wer_res.insertions,
            "wer_errors": wer_err,
            "wer": (wer_err / n_words) * 100,
            "n_chars": n_chars,
            "cer_subs": cer_res.substitutions,
            "cer_dels": cer_res.deletions,
            "cer_ins": cer_res.insertions,
            "cer_errors": cer_err,
            "cer": (cer_err / n_chars) * 100,
        })
        print(f"[{row['sample_id']}] WER={results[-1]['wer']:.2f}% ({wer_err}/{n_words}) | CER={results[-1]['cer']:.2f}% ({cer_err}/{n_chars})")
        print(f"  Ref:  {ref_norm}")
        print(f"  Pred: {pred_norm}\n")

    df_res = pd.DataFrame(results)
    tot_words = int(df_res["n_words"].sum())
    tot_wer_errors = int(df_res["wer_errors"].sum())
    tot_chars = int(df_res["n_chars"].sum())
    tot_cer_errors = int(df_res["cer_errors"].sum())

    micro_wer = (tot_wer_errors / tot_words) * 100
    micro_cer = (tot_cer_errors / tot_chars) * 100

    print("=" * 70)
    print("PILOT-2 TEST EVALUATION ON 9-SAMPLE TEST SET:")
    print(f"Total Words: {tot_words}, Total Word Errors: {tot_wer_errors}")
    print(f"Micro WER: {micro_wer:.2f}% (vs E0: 18.93%, Delta: {micro_wer - 18.93:+.2f} pp)")
    print(f"Total Chars: {tot_chars}, Total Char Errors: {tot_cer_errors}")
    print(f"Micro CER: {micro_cer:.2f}% (vs E0: 11.07%, Delta: {micro_cer - 11.07:+.2f} pp)")
    print("=" * 70)

    # Save to CSV
    output_path = "vietnamese_asr_app/reports/pilot2_test_results.csv"
    df_res.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")

if __name__ == "__main__":
    main()
