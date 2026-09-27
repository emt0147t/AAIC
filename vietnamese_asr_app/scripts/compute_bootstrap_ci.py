import pandas as pd
import numpy as np
import torch
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from vietnamese_asr_app.evaluation.metrics import compute_levenshtein_wer, compute_levenshtein_cer, normalize_vietnamese_text

def main():
    comp_df = pd.read_csv('vietnamese_asr_app/reports/e1_vs_e2_prediction_comparison.csv')
    processor = WhisperProcessor.from_pretrained('vinai/PhoWhisper-tiny')
    model = WhisperForConditionalGeneration.from_pretrained('vinai/PhoWhisper-tiny')
    model.eval()

    manifest_df = pd.read_csv('vietnamese_asr_app/manifests/test_manifest.csv')

    samples = []
    for idx, row in manifest_df.iterrows():
        audio, sr = sf.read(row['audio_path'])
        inputs = processor(audio, sampling_rate=16000, return_tensors='pt')
        with torch.no_grad():
            generated_ids = model.generate(
                inputs.input_features,
                max_new_tokens=225,
                language='vi',
                task='transcribe'
            )
        pred_raw = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        pred_e0 = normalize_vietnamese_text(pred_raw)
        ref = normalize_vietnamese_text(row['transcript'])
        
        e1_row = comp_df[comp_df['sample_id'] == row['sample_id']].iloc[0]
        pred_e1 = normalize_vietnamese_text(str(e1_row['e1_prediction']))
        
        wer_e0 = compute_levenshtein_wer(ref, pred_e0)
        cer_e0 = compute_levenshtein_cer(ref, pred_e0)
        wer_e1 = compute_levenshtein_wer(ref, pred_e1)
        cer_e1 = compute_levenshtein_cer(ref, pred_e1)
        
        n_words = len(ref.split())
        n_chars = len(ref)
        
        samples.append({
            'sample_id': row['sample_id'],
            'n_words': n_words,
            'e0_wer_err': wer_e0.substitutions + wer_e0.deletions + wer_e0.insertions,
            'e1_wer_err': wer_e1.substitutions + wer_e1.deletions + wer_e1.insertions,
            'n_chars': n_chars,
            'e0_cer_err': cer_e0.substitutions + cer_e0.deletions + cer_e0.insertions,
            'e1_cer_err': cer_e1.substitutions + cer_e1.deletions + cer_e1.insertions,
        })

    df_samples = pd.DataFrame(samples)
    tot_w = df_samples['n_words'].sum()
    tot_c = df_samples['n_chars'].sum()
    
    e0_wer = df_samples['e0_wer_err'].sum() / tot_w * 100
    e1_wer = df_samples['e1_wer_err'].sum() / tot_w * 100
    e0_cer = df_samples['e0_cer_err'].sum() / tot_c * 100
    e1_cer = df_samples['e1_cer_err'].sum() / tot_c * 100
    
    print(f"Total Words: {tot_w}, Total Chars: {tot_c}")
    print(f"E0 WER: {e0_wer:.2f}%, E1 WER: {e1_wer:.2f}%")
    print(f"E0 CER: {e0_cer:.2f}%, E1 CER: {e1_cer:.2f}%")

    np.random.seed(42)
    n_samples = len(df_samples)
    n_boot = 1000

    boot_e0_wer, boot_e1_wer = [], []
    boot_e0_cer, boot_e1_cer = [], []

    for _ in range(n_boot):
        idx_resample = np.random.choice(n_samples, size=n_samples, replace=True)
        res_df = df_samples.iloc[idx_resample]
        
        tw = res_df['n_words'].sum()
        tc = res_df['n_chars'].sum()
        
        boot_e0_wer.append(res_df['e0_wer_err'].sum() / tw * 100)
        boot_e1_wer.append(res_df['e1_wer_err'].sum() / tw * 100)
        boot_e0_cer.append(res_df['e0_cer_err'].sum() / tc * 100)
        boot_e1_cer.append(res_df['e1_cer_err'].sum() / tc * 100)

    print("\n--- 1000-BOOTSTRAP 95% CONFIDENCE INTERVALS ---")
    print(f"E0 Baseline WER: {e0_wer:.2f}% [95% CI: {np.percentile(boot_e0_wer, 2.5):.2f}% - {np.percentile(boot_e0_wer, 97.5):.2f}%]")
    print(f"E1 LoRA Real WER: {e1_wer:.2f}% [95% CI: {np.percentile(boot_e1_wer, 2.5):.2f}% - {np.percentile(boot_e1_wer, 97.5):.2f}%]")
    print(f"E2 LoRA Real+Synth WER: {e1_wer:.2f}% [95% CI: {np.percentile(boot_e1_wer, 2.5):.2f}% - {np.percentile(boot_e1_wer, 97.5):.2f}%]")
    print(f"E0 Baseline CER: {e0_cer:.2f}% [95% CI: {np.percentile(boot_e0_cer, 2.5):.2f}% - {np.percentile(boot_e0_cer, 97.5):.2f}%]")
    print(f"E1 LoRA Real CER: {e1_cer:.2f}% [95% CI: {np.percentile(boot_e1_cer, 2.5):.2f}% - {np.percentile(boot_e1_cer, 97.5):.2f}%]")
    print(f"E2 LoRA Real+Synth CER: {e1_cer:.2f}% [95% CI: {np.percentile(boot_e1_cer, 2.5):.2f}% - {np.percentile(boot_e1_cer, 97.5):.2f}%]")

if __name__ == '__main__':
    main()
